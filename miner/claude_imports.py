"""Deterministic primitives for Claude ``@path`` import-syntax analysis.

This module implements the documentation-derived Phase 2r detector.  It is
deliberately separate from :mod:`miner.markdown_references`: Markdown
references and semantic delegation are different constructs from Claude
Code's ``@path`` file-import syntax.

The public Claude Code documentation accessed on 29 July 2026 states that
imports:

* use ``@path/to/import`` and may appear anywhere in ``CLAUDE.md``;
* may name relative or absolute paths and are not limited to Markdown files;
* resolve relative to the file containing the import;
* are skipped inside Markdown code spans and fenced code blocks; and
* may recurse to a maximum depth of four hops.

The documentation does not publish a complete character-level token grammar.
The scanner below therefore exposes a versioned *operational* grammar and
retains source evidence and exclusion reasons.  A detected plain-text token is
an import-syntax candidate, not proof that its target exists, was approved, or
loaded at runtime.
"""
from __future__ import annotations

import bisect
import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath


DETECTOR_VERSION = "phase2r_at_path_v1"
DOCUMENTATION_URL = "https://code.claude.com/docs/en/memory#import-additional-files"
DOCUMENTATION_ACCESSED = "2026-07-29"
MAX_DOCUMENTED_IMPORT_HOPS = 4

FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
HTML_COMMENT_START = "<!--"
HTML_COMMENT_END = "-->"
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

# Whitespace and Markdown/prose delimiters terminate the operational token.
# A period remains available inside a path and is trimmed only when it is the
# final sentence character.  Spaces in filenames are not documented as an
# import quoting form and consequently terminate a token.
TOKEN_TERMINATORS = frozenset("`<>\"'[](),;!|&")
TRAILING_SENTENCE_PUNCTUATION = "."
UNSUPPORTED_PATTERN_CHARACTERS = frozenset("*?#{}")


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class ClaudeImportOccurrence:
    """One operational ``@path`` token, including excluded contexts."""

    line_number: int
    column_number: int
    end_column_number: int
    raw_token: str
    raw_target: str
    context_kind: str
    decision: str
    rule_id: str
    surface_form: str
    path_kind: str
    evidence_context: str


@dataclass(frozen=True)
class ImportTargetResolution:
    """Source-relative lexical interpretation of one import target."""

    normalized_target: str
    resolved_target: str
    path_kind: str
    path_relation: str
    target_basename: str
    target_extension_class: str
    is_self_reference: int


def _fence_open(line: str) -> tuple[str, int] | None:
    match = FENCE_OPEN_RE.match(line)
    if match is None:
        return None
    marker = match.group(1)
    if marker[0] == "`" and "`" in match.group(2):
        return None
    return marker[0], len(marker)


def _fence_close(line: str, character: str, minimum_length: int) -> bool:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    match = re.fullmatch(re.escape(character) + r"+[ \t]*", stripped)
    return match is not None and len(stripped.rstrip(" \t")) >= minimum_length


def _fenced_spans(content: str) -> list[_Span]:
    spans: list[_Span] = []
    fence: tuple[str, int] | None = None
    fence_start = 0
    offset = 0
    for raw_line in content.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if fence is not None:
            if _fence_close(line, fence[0], fence[1]):
                spans.append(_Span(fence_start, offset + len(raw_line), "fenced_code"))
                fence = None
            offset += len(raw_line)
            continue
        opening = _fence_open(line)
        if opening is not None:
            fence = opening
            fence_start = offset
        offset += len(raw_line)
    if fence is not None:
        spans.append(_Span(fence_start, len(content), "fenced_code"))
    return spans


def _html_comment_spans(content: str, *, block_only: bool = False) -> list[_Span]:
    spans: list[_Span] = []
    offset = 0
    while True:
        start = content.find(HTML_COMMENT_START, offset)
        if start < 0:
            break
        closing = content.find(HTML_COMMENT_END, start + len(HTML_COMMENT_START))
        end = len(content) if closing < 0 else closing + len(HTML_COMMENT_END)
        line_start = content.rfind("\n", 0, start) + 1
        prefix = content[line_start:start]
        is_block_start = len(prefix) <= 3 and not prefix.strip(" ")
        if not block_only or is_block_start:
            spans.append(_Span(start, end, "html_comment"))
        offset = end
    return spans


def _blank_line_spans(content: str) -> list[_Span]:
    spans: list[_Span] = []
    offset = 0
    for raw_line in content.splitlines(keepends=True):
        if not raw_line.strip(" \t\r\n"):
            spans.append(_Span(offset, offset + len(raw_line), "blank_line"))
        offset += len(raw_line)
    return spans


def _span_containing(spans: list[_Span], starts: list[int], position: int) -> _Span | None:
    index = bisect.bisect_right(starts, position) - 1
    if index >= 0 and spans[index].start <= position < spans[index].end:
        return spans[index]
    return None


def _preformatted_span_at(
    fenced: list[_Span],
    fenced_starts: list[int],
    comments: list[_Span],
    comment_starts: list[int],
    position: int,
) -> _Span | None:
    # Fences take precedence when a malformed comment happens to cross one.
    return _span_containing(fenced, fenced_starts, position) or _span_containing(
        comments, comment_starts, position
    )


def _inline_code_spans(
    content: str,
    fenced: list[_Span],
    comments: list[_Span],
    additional_barrier_starts: list[int] | None = None,
) -> list[_Span]:
    """Return CommonMark-style equal-backtick-run code spans.

    Code spans may cross physical lines.  An unmatched backtick run remains
    ordinary text, matching CommonMark's basic delimiter behavior.
    """
    fenced_starts = [span.start for span in fenced]
    comment_starts = [span.start for span in comments]
    barrier_starts = sorted(
        [
            *(span.start for span in fenced),
            *(span.start for span in comments),
            *(additional_barrier_starts or []),
        ]
    )
    runs: list[tuple[int, int, int, int]] = []
    position = 0
    while position < len(content):
        start = content.find("`", position)
        if start < 0:
            break
        covered = _preformatted_span_at(
            fenced,
            fenced_starts,
            comments,
            comment_starts,
            start,
        )
        if covered is not None:
            position = covered.end
            continue
        end = start + 1
        while end < len(content) and content[end] == "`":
            end += 1
        segment = bisect.bisect_right(barrier_starts, start)
        runs.append((start, end, end - start, segment))
        position = end

    spans: list[_Span] = []
    index = 0
    while index < len(runs):
        opener = runs[index]
        closing_index = index + 1
        while (
            closing_index < len(runs)
            and runs[closing_index][3] == opener[3]
            and runs[closing_index][2] != opener[2]
        ):
            closing_index += 1
        if closing_index >= len(runs) or runs[closing_index][3] != opener[3]:
            index += 1
            continue
        closer = runs[closing_index]
        spans.append(_Span(opener[0], closer[1], "inline_code"))
        index = closing_index + 1
    return spans


def _line_starts(content: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", content):
        if match.end() < len(content):
            starts.append(match.end())
    return starts


def _line_bounds(content: str, starts: list[int], position: int) -> tuple[int, int, int]:
    index = bisect.bisect_right(starts, position) - 1
    start = starts[index]
    end = content.find("\n", start)
    if end < 0:
        end = len(content)
    if end > start and content[end - 1] == "\r":
        end -= 1
    return index + 1, start, end


def _at_boundary(content: str, position: int) -> bool:
    if position == 0:
        return True
    previous = content[position - 1]
    # This filters email addresses and identifier-internal @ characters while
    # retaining punctuation-delimited imports and npm-like path shapes.
    return not (previous.isalnum() or previous in "_.-@")


def _scan_uri_end(content: str, start: int) -> int:
    end = start
    while end < len(content):
        character = content[end]
        if character.isspace() or character in "`<>\"'[]{}()":
            break
        end += 1
    while end > start and content[end - 1] in ".,;!?":
        end -= 1
    return end


def _scan_target_end(content: str, start: int) -> int:
    if start >= len(content):
        return start
    uri_match = URI_SCHEME_RE.match(content[start:])
    if uri_match is not None:
        return _scan_uri_end(content, start)

    end = start
    while end < len(content):
        character = content[end]
        if character.isspace() or character in TOKEN_TERMINATORS:
            break
        if character == ":":
            is_windows_drive = (
                end == start + 1
                and content[start].isalpha()
                and end + 1 < len(content)
                and content[end + 1] in "\\/"
            )
            if not is_windows_drive:
                break
        end += 1
    while end > start and content[end - 1] == "*":
        end -= 1
    while end > start and content[end - 1] in TRAILING_SENTENCE_PUNCTUATION:
        end -= 1
    return end


def import_path_kind(raw_target: str) -> str:
    """Classify a raw target without assuming the runtime operating system."""
    if URI_SCHEME_RE.match(raw_target):
        return "unsupported_uri"
    if any(character in raw_target for character in UNSUPPORTED_PATTERN_CHARACTERS):
        return "unsupported_pattern"
    if "$" in raw_target or raw_target.startswith("%"):
        return "unsupported_variable"
    if raw_target.startswith("~/") or raw_target.startswith("~\\"):
        return "home_relative"
    if raw_target.startswith("\\\\"):
        return "windows_unc_absolute"
    if WINDOWS_DRIVE_RE.match(raw_target):
        return "windows_absolute"
    if raw_target.startswith("/"):
        return "posix_absolute"
    if "\\" in raw_target:
        return "platform_relative"
    return "relative"


def _file_like_target(raw_target: str) -> bool:
    normalized = raw_target.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return False
    basename = normalized.rsplit("/", 1)[-1]
    return basename not in {"", ".", "..", "~"}


def _surface_form(line: str, raw_token: str) -> str:
    stripped = line.strip()
    if stripped == raw_token:
        return "whole_line"
    list_match = re.match(r"^[ \t]{0,3}(?:[-+*]|\d+[.)])[ \t]+(.*)$", line)
    if list_match is not None and list_match.group(1).strip() == raw_token:
        return "list_item_only"
    residual = line.replace(raw_token, "", 1)
    formatting_only = re.sub(r"[\s*_~>#.+()\[\]{}:;,-]", "", residual)
    if not formatting_only:
        return "standalone_formatted"
    return "prose_embedded"


def _evidence_context(content: str, start: int, end: int, radius: int = 180) -> str:
    excerpt_start = max(0, start - radius)
    excerpt_end = min(len(content), end + radius)
    return content[excerpt_start:excerpt_end].replace("\r", " ").replace("\n", " ")


def extract_claude_import_occurrences(
    content: str,
) -> tuple[ClaudeImportOccurrence, ...]:
    """Extract active and explicitly excluded operational ``@path`` tokens."""
    fenced = _fenced_spans(content)
    block_comments = _html_comment_spans(content, block_only=True)
    inline = _inline_code_spans(
        content,
        fenced,
        block_comments,
        [span.start for span in _blank_line_spans(content)],
    )
    comments = _html_comment_spans(content)
    fenced_starts = [span.start for span in fenced]
    comment_starts = [span.start for span in comments]
    inline_starts = [span.start for span in inline]
    starts = _line_starts(content)

    occurrences: list[ClaudeImportOccurrence] = []
    position = 0
    while True:
        at = content.find("@", position)
        if at < 0:
            break
        position = at + 1
        if not _at_boundary(content, at):
            continue
        target_start = at + 1
        target_end = _scan_target_end(content, target_start)
        if target_end <= target_start:
            continue
        raw_target = content[target_start:target_end]
        if not _file_like_target(raw_target):
            context_kind = "plain_text"
            decision = "excluded_non_file_target"
            rule_id = "non_file_target_exclusion_v1"
        else:
            fenced_span = _span_containing(fenced, fenced_starts, at)
            comment_span = _span_containing(comments, comment_starts, at)
            inline_span = _span_containing(inline, inline_starts, at)
            if fenced_span is not None:
                context_kind = "fenced_code"
                decision = "excluded_fenced_code"
                rule_id = "fenced_code_exclusion_v1"
            elif inline_span is not None:
                context_kind = "inline_code"
                decision = "excluded_inline_code"
                rule_id = "inline_code_exclusion_v1"
            elif comment_span is not None:
                context_kind = "html_comment"
                decision = "excluded_html_comment"
                rule_id = "html_comment_exclusion_v1"
            elif import_path_kind(raw_target) == "unsupported_uri":
                context_kind = "plain_text"
                decision = "excluded_unsupported_uri"
                rule_id = "filesystem_path_only_v1"
            elif import_path_kind(raw_target) == "unsupported_pattern":
                context_kind = "plain_text"
                decision = "excluded_unsupported_pattern"
                rule_id = "undocumented_pattern_exclusion_v1"
            elif import_path_kind(raw_target) == "unsupported_variable":
                context_kind = "plain_text"
                decision = "excluded_unsupported_variable"
                rule_id = "undocumented_variable_exclusion_v1"
            else:
                context_kind = "plain_text"
                decision = "import_candidate"
                rule_id = "documented_at_path_candidate_v1"

        line_number, line_start, line_end = _line_bounds(content, starts, at)
        raw_token = content[at:target_end]
        line = content[line_start:line_end]
        occurrences.append(
            ClaudeImportOccurrence(
                line_number=line_number,
                column_number=at - line_start + 1,
                end_column_number=target_end - line_start + 1,
                raw_token=raw_token,
                raw_target=raw_target,
                context_kind=context_kind,
                decision=decision,
                rule_id=rule_id,
                surface_form=_surface_form(line, raw_token),
                path_kind=import_path_kind(raw_target),
                evidence_context=_evidence_context(content, at, target_end),
            )
        )
        position = target_end
    return tuple(occurrences)


def _target_basename(raw_target: str, path_kind: str) -> str:
    if path_kind in {"windows_absolute", "windows_unc_absolute", "platform_relative"}:
        return PureWindowsPath(raw_target).name
    return PurePosixPath(raw_target).name


def target_extension_class(target_basename: str) -> str:
    folded = target_basename.casefold()
    if folded.endswith(".md"):
        return "markdown"
    if "." in target_basename.lstrip("."):
        return "other_extension"
    return "extensionless"


def target_shape(raw_target: str, target_extension: str) -> str:
    """Separate explicit paths from ambiguous bare extensionless tokens."""
    if target_extension == "markdown":
        return "markdown_file"
    if target_extension == "other_extension":
        return "other_extension_file"
    if "/" in raw_target or "\\" in raw_target:
        return "path_extensionless"
    if raw_target.startswith("."):
        return "dotfile_extensionless"
    return "bare_extensionless"


def _local_path_relation(source_path: str, resolved_target: str) -> str:
    if resolved_target == ".." or resolved_target.startswith("../"):
        return "outside_repository"
    source_directory = PurePosixPath(posixpath.dirname(source_path)).parts
    target_directory = PurePosixPath(posixpath.dirname(resolved_target)).parts
    if source_directory == target_directory:
        return "same_directory"
    if len(target_directory) < len(source_directory) and source_directory[: len(target_directory)] == target_directory:
        return "ancestor_directory"
    if len(target_directory) > len(source_directory) and target_directory[: len(source_directory)] == source_directory:
        return "descendant_directory"
    return "other_directory"


def resolve_import_target(
    source_path: str,
    raw_target: str,
) -> ImportTargetResolution:
    """Resolve an operational target while preserving external path classes."""
    kind = import_path_kind(raw_target)
    basename = _target_basename(raw_target, kind)
    extension_class = target_extension_class(basename)
    normalized = raw_target
    resolved = ""
    relation = "unsupported"

    if kind == "relative":
        normalized = posixpath.normpath(raw_target)
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_path), normalized)
        )
        relation = _local_path_relation(source_path, resolved)
    elif kind == "platform_relative":
        relation = "platform_dependent"
    elif kind == "home_relative":
        relation = "external_home"
    elif kind in {"posix_absolute", "windows_absolute", "windows_unc_absolute"}:
        relation = "external_absolute"
    elif kind in {"unsupported_pattern", "unsupported_variable"}:
        if not raw_target.startswith(("/", "~/", "~\\")) and "\\" not in raw_target:
            normalized = posixpath.normpath(raw_target)
        relation = kind
    elif kind == "unsupported_uri":
        relation = kind

    is_self = int(bool(resolved) and resolved.casefold() == source_path.casefold())
    return ImportTargetResolution(
        normalized_target=normalized,
        resolved_target=resolved,
        path_kind=kind,
        path_relation=relation,
        target_basename=basename,
        target_extension_class=extension_class,
        is_self_reference=is_self,
    )
