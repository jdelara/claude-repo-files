"""Small, dependency-free primitives for local Markdown-reference analysis.

The extractor is intentionally conservative.  It ignores fenced code blocks,
retains source evidence, and distinguishes direct ``@path.md`` inclusions from
Markdown links and other visible path mentions.  Repository-relative target
resolution belongs to the later corpus analysis because it needs the source
file path.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator
from urllib.parse import unquote, urlsplit


FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
DIRECT_INCLUDE_RE = re.compile(
    r"^[ \t]{0,3}@(?P<target>[^\s<>`]+\.md(?:[?#][^\s<>`]*)?)[ \t]*$",
    re.IGNORECASE,
)
COMMENT_INCLUDE_RE = re.compile(
    r"^[ \t]*<!--\s*(?:include|import|source)\s*:\s*@?"
    r"(?P<target>[^\s<>`]+\.md(?:[?#][^\s<>`]*)?)\s*-->[ \t]*$",
    re.IGNORECASE,
)
MARKDOWN_LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]*\]\(\s*<?(?P<target>[^\s)>]+\.md(?:[?#][^\s)>]*)?)>?"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)",
    re.IGNORECASE,
)
LOCAL_MD_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<target>"
    r"(?:@)?(?:\.{1,2}/|/)?"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.md"
    r"(?:[?#][A-Za-z0-9_.~:/?&=%+\-]*)?"
    r")",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?|file)://[^\s<>]+", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
BARE_MD_HEADING_RE = re.compile(
    r"^[ ]{0,3}#{1,6}[ \t]+[`*_~]*@?"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.md"
    r"[`*_~]*[ \t]*#*[ \t]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MarkdownReference:
    line_number: int
    syntax: str
    raw_target: str
    normalized_target: str
    target_basename: str
    source_line: str


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
    return (
        match is not None
        and len(stripped.rstrip(" \t")) >= minimum_length
    )


def iter_unfenced_lines(content: str) -> Iterator[tuple[int, str]]:
    """Yield one-based line numbers and text outside Markdown fences."""
    fence: tuple[str, int] | None = None
    for line_number, line in enumerate(content.splitlines(), 1):
        if fence is not None:
            if _fence_close(line, fence[0], fence[1]):
                fence = None
            continue
        opening = _fence_open(line)
        if opening is not None:
            fence = opening
            continue
        yield line_number, line


def normalize_markdown_target(raw_target: str) -> str:
    """Normalize a local target lexically while retaining ``..`` segments."""
    target = raw_target.strip().lstrip("@").strip("<>")
    path = re.split(r"[?#]", target, maxsplit=1)[0]
    if path.startswith("/"):
        normalized = "/" + posixpath.normpath(path).lstrip("/")
    else:
        normalized = posixpath.normpath(path)
    return normalized


def is_local_markdown_target(raw_target: str) -> bool:
    target = raw_target.strip().lstrip("@").strip("<>")
    return not SCHEME_RE.match(target) and not target.startswith("//")


def markdown_target_basename(normalized_target: str) -> str:
    return PurePosixPath(normalized_target).name.casefold()


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _iter_local_markdown_reference_occurrences(
    content: str,
) -> Iterator[MarkdownReference]:
    for line_number, line in iter_unfenced_lines(content):
        comment_include = COMMENT_INCLUDE_RE.match(line)
        if comment_include is not None:
            raw = comment_include.group("target")
            if not is_local_markdown_target(raw):
                continue
            normalized = normalize_markdown_target(raw)
            yield MarkdownReference(
                line_number=line_number,
                syntax="comment_include",
                raw_target=raw,
                normalized_target=normalized,
                target_basename=markdown_target_basename(normalized),
                source_line=line,
            )
            continue
        direct = DIRECT_INCLUDE_RE.match(line)
        if direct is not None:
            raw = direct.group("target")
            if not is_local_markdown_target(raw):
                continue
            normalized = normalize_markdown_target(raw)
            yield MarkdownReference(
                line_number=line_number,
                syntax="direct_include",
                raw_target=raw,
                normalized_target=normalized,
                target_basename=markdown_target_basename(normalized),
                source_line=line,
            )
            continue

        occupied = [match.span() for match in URL_RE.finditer(line)]
        occupied.extend(match.span() for match in HTML_COMMENT_RE.finditer(line))
        for match in MARKDOWN_LINK_RE.finditer(line):
            raw = match.group("target")
            occupied.append(match.span())
            if not is_local_markdown_target(raw):
                continue
            normalized = normalize_markdown_target(raw)
            yield MarkdownReference(
                line_number=line_number,
                syntax="markdown_link",
                raw_target=raw,
                normalized_target=normalized,
                target_basename=markdown_target_basename(normalized),
                source_line=line,
            )

        if BARE_MD_HEADING_RE.match(line):
            continue
        for match in LOCAL_MD_PATH_RE.finditer(line):
            if _overlaps(match.span(), occupied):
                continue
            raw = match.group("target")
            normalized = normalize_markdown_target(raw)
            yield MarkdownReference(
                line_number=line_number,
                syntax="path_mention",
                raw_target=raw,
                normalized_target=normalized,
                target_basename=markdown_target_basename(normalized),
                source_line=line,
            )


def extract_local_markdown_reference_occurrences(
    content: str,
) -> tuple[MarkdownReference, ...]:
    """Extract every conservative local ``.md`` mention outside fences.

    Unlike :func:`extract_local_markdown_references`, repeated mentions of the
    same target on one line are retained.  This occurrence-preserving form is
    used by corpus analyses; the original deduplicated function remains stable
    for the Phase 1b content classifier.
    """
    return tuple(_iter_local_markdown_reference_occurrences(content))


def extract_local_markdown_references(content: str) -> tuple[MarkdownReference, ...]:
    """Extract unique line/syntax/target local references outside fences."""
    references = extract_local_markdown_reference_occurrences(content)

    unique: dict[tuple[int, str, str], MarkdownReference] = {}
    for reference in references:
        key = (
            reference.line_number,
            reference.syntax,
            reference.normalized_target,
        )
        unique.setdefault(key, reference)
    return tuple(unique.values())


def _trim_external_url(raw_url: str) -> str:
    url = raw_url.rstrip(".,;:!?\"'")
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    while url.endswith("]") and url.count("[") < url.count("]"):
        url = url[:-1]
    return url


def is_external_markdown_url(raw_url: str) -> bool:
    """Return whether an HTTP(S)/file URL has a path ending in ``.md``."""
    try:
        parsed = urlsplit(_trim_external_url(raw_url.strip().strip("<>")))
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https", "file"}
        and unquote(parsed.path).casefold().endswith(".md")
    )


def external_markdown_basename(raw_url: str) -> str:
    parsed = urlsplit(_trim_external_url(raw_url.strip().strip("<>")))
    return PurePosixPath(unquote(parsed.path)).name.casefold()


def extract_external_markdown_references(
    content: str,
) -> tuple[MarkdownReference, ...]:
    """Extract external Markdown links and bare URLs outside fenced code."""
    references: list[MarkdownReference] = []
    for line_number, line in iter_unfenced_lines(content):
        occupied = [match.span() for match in HTML_COMMENT_RE.finditer(line)]
        for match in MARKDOWN_LINK_RE.finditer(line):
            raw = match.group("target")
            occupied.append(match.span())
            if is_local_markdown_target(raw) or not is_external_markdown_url(raw):
                continue
            cleaned = _trim_external_url(raw.strip().strip("<>"))
            references.append(
                MarkdownReference(
                    line_number=line_number,
                    syntax="external_markdown_link",
                    raw_target=cleaned,
                    normalized_target=cleaned,
                    target_basename=external_markdown_basename(cleaned),
                    source_line=line,
                )
            )

        for match in URL_RE.finditer(line):
            if _overlaps(match.span(), occupied):
                continue
            raw = _trim_external_url(match.group(0))
            if not is_external_markdown_url(raw):
                continue
            references.append(
                MarkdownReference(
                    line_number=line_number,
                    syntax="external_url",
                    raw_target=raw,
                    normalized_target=raw,
                    target_basename=external_markdown_basename(raw),
                    source_line=line,
                )
            )
    return tuple(references)
