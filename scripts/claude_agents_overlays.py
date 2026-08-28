#!/usr/bin/env python3
"""Analyze ``CLAUDE.md`` files that import ``AGENTS.md`` and add an overlay.

The analysis reuses the documentation-derived Claude ``@path`` lexer from
``miner.claude_imports``.  It selects active import-syntax candidates whose
case-insensitive target basename is ``AGENTS.md`` and then distinguishes:

* strict shims containing only one or more imports and whitespace;
* files with residual text after the imports are removed;
* files with substantive residual text after the final import; and
* files with a heading that explicitly labels a following section as being
  for Claude, Claude Code, or a Claude-specific adapter.

The final category is a lexical candidate set, not a semantic ground truth.
An optional English directive-cue heuristic and placeholder detector make the
candidate set easier to audit, but may miss non-English or unusually worded
instructions.  Import target existence and runtime loading are not checked.

The SQLite source is opened with ``mode=ro&immutable=1`` and
``PRAGMA query_only=ON``.  The script also records database hashes before and
after the analysis.

Example:

    python scripts/claude_agents_overlays.py --db mined.db \
        --files-output article/claude_agents_overlay_files.csv \
        --sections-output article/claude_agents_overlay_sections.csv \
        --summary-output article/claude_agents_overlay_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miner.claude_imports import (  # noqa: E402
    DETECTOR_VERSION as IMPORT_DETECTOR_VERSION,
    DOCUMENTATION_ACCESSED,
    DOCUMENTATION_URL,
    extract_claude_import_occurrences,
    resolve_import_target,
)
from scripts.markdown_section_stats import parse_headings  # noqa: E402


ANALYSIS_VERSION = "claude_agents_overlay_v1"
POPULATION_DEFINITION = (
    "files whose stored POSIX path has a case-insensitive basename equal to "
    "CLAUDE.md"
)
TARGET_BASENAME = "agents.md"
DEFAULT_SUBSTANTIVE_WORD_THRESHOLD = 10
DEFAULT_MINIMUM_SECTION_BODY_WORDS = 3
DEFAULT_EXCERPT_CHARACTERS = 500
MAX_OVERLAY_HEADING_CHARACTERS = 160

WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
HTML_COMMENT_RE = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)

# The negative lookbehind prevents paths such as ``.claude/rules`` from being
# mistaken for an explicit reference to the Claude product in a heading.
CLAUDE_HEADING_TERM_RE = re.compile(
    r"(?<![\w./-])claude(?:[\s-]+code\b|-(?:specific|only)\b|(?!-[a-z])\b)",
    re.IGNORECASE,
)
OVERLAY_HEADING_CUE_RE = re.compile(
    r"\b(?:"
    r"specific|only|additional|additions?|addenda?|adapter|override|"
    r"orchestration|instructions?|guidance|guidelines?|notes?|rules?|"
    r"workflow|working\s+agreements?|configuration|settings?|preferences?|"
    r"commands?|tips?|behavio[u]?r|context|memory"
    r")\b",
    re.IGNORECASE,
)
EXPLICIT_SPECIFICITY_RE = re.compile(
    r"\b(?:specific|only|additional|additions?|addenda?|adapter|override|"
    r"orchestration)\b",
    re.IGNORECASE,
)
INSTRUCTION_HEADING_RE = re.compile(
    r"\b(?:instructions?|guidance|guidelines?|rules?|workflow|"
    r"working\s+agreements?|commands?|behavio[u]?r|preferences?)\b",
    re.IGNORECASE,
)
ENGLISH_DIRECTIVE_CUE_RE = re.compile(
    r"\b(?:must|should|always|never|use|run|prefer|avoid|ensure|keep|"
    r"do\s+not|don't|ask|invoke|read|write|start|before|after|when|if|"
    r"only|require|required|dispatch|delegate|perform|append|treat|follow|"
    r"call|update|verify|check)\b",
    re.IGNORECASE,
)
COMMON_CLAUDE_BOILERPLATE_RE = re.compile(
    r"(?im)^[ \t]*This file provides guidance to Claude Code "
    r"\(claude\.ai/code\) when working with (?:code in )?this repository\.?"
    r"[ \t]*$"
)
PLACEHOLDER_PATTERNS = (
    re.compile(r"(?is)^\s*[_*(]*\s*none(?:\s+yet)?\b"),
    re.compile(
        r"(?i)\bno\s+(?:claude(?:\s+code)?[- ]specific|claude[- ]only|"
        r"specific)\s+instructions?\b"
    ),
    re.compile(r"(?i)\bkeep\s+this\s+section\s+empty\s+unless\b"),
    re.compile(
        r"(?i)\badd\s+(?:only\s+)?claude(?:\s+code)?[- ]specific\s+"
        r"(?:guidance|instructions?|notes?)\s+here\b"
    ),
    re.compile(r"(?i)\breserved\s+for\s+claude(?:\s+code)?\b"),
)

THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    "plan_mode": re.compile(r"\bplan mode\b", re.IGNORECASE),
    "skills_or_slash_commands": re.compile(
        r"\bskills?\b|(?<!\w)/(?:[a-z][\w-]+)", re.IGNORECASE
    ),
    "hooks": re.compile(r"\bhooks?\b", re.IGNORECASE),
    "subagents_or_delegation": re.compile(
        r"\bsub-?agents?\b|\bdelegat(?:e|es|ed|ing|ion)\b", re.IGNORECASE
    ),
    "mcp": re.compile(r"\bMCP\b", re.IGNORECASE),
    "model_selection": re.compile(
        r"\b(?:opus|sonnet|haiku|model)\b", re.IGNORECASE
    ),
    "claude_configuration_paths": re.compile(
        r"(?:\.claude/|CLAUDE\.local\.md|settings\.json)", re.IGNORECASE
    ),
    "review_commit_or_pull_request": re.compile(
        r"\b(?:review|commit|pull request|PR)\b", re.IGNORECASE
    ),
    "permissions_or_safety": re.compile(
        r"\b(?:permission|sandbox|destructive|safety|safe|approval)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class ImportSpan:
    """One active ``AGENTS.md`` import and its character offsets."""

    start: int
    end: int
    line_number: int
    raw_token: str
    raw_target: str


@dataclass(frozen=True)
class ContentSection:
    """Content-only evidence for an explicitly Claude-labeled section."""

    heading: str
    normalized_heading: str
    heading_level: int
    heading_line: int
    label_kind: str
    body_word_count: int
    semantic_body_word_count: int
    has_english_directive_cue: int
    has_placeholder_language: int
    concrete_directive_candidate: int
    themes: tuple[str, ...]
    evidence_excerpt: str


@dataclass(frozen=True)
class ContentAnalysis:
    """Analysis that depends only on a file's content."""

    import_spans: tuple[ImportSpan, ...]
    exact_literal_at_agents_occurrences: int
    target_form_counts: tuple[tuple[str, int], ...]
    strict_import_only: int
    no_residual_outside_html_comments: int
    residual_word_count: int
    residual_after_final_import_word_count: int
    residual_excerpt: str
    sections: tuple[ContentSection, ...]


@dataclass(frozen=True)
class OverlayFileRow:
    """One exact ``CLAUDE.md`` file with an active ``AGENTS.md`` import."""

    file_id: int
    repo_full_name: str
    source_path: str
    html_url: str
    content_hash: str
    size_bytes: int | None
    repository_stars: int | None
    repository_language: str
    repository_exact_claude_files: int
    exact_population_content_copies: int
    agents_import_occurrences: int
    exact_literal_at_agents_occurrences: int
    has_exact_literal_at_agents_import: int
    target_form_counts_json: str
    first_agents_import_line: int
    last_agents_import_line: int
    strict_import_only: int
    no_residual_outside_html_comments: int
    residual_class: str
    residual_word_count: int
    residual_after_final_import_word_count: int
    substantive_residual: int
    substantive_residual_after_final_import: int
    explicit_claude_labeled_sections_after_final_import: int
    labeled_sections_with_nonempty_body: int
    labeled_sections_with_placeholder_language: int
    concrete_labeled_directive_sections: int
    has_concrete_labeled_directive_candidate: int
    overlay_headings_json: str
    overlay_themes_json: str
    residual_excerpt: str


@dataclass(frozen=True)
class OverlaySectionRow:
    """One explicit Claude-labeled section appearing after the final import."""

    file_id: int
    repo_full_name: str
    source_path: str
    html_url: str
    content_hash: str
    heading: str
    normalized_heading: str
    heading_level: int
    heading_line: int
    label_kind: str
    body_word_count: int
    semantic_body_word_count: int
    has_english_directive_cue: int
    has_placeholder_language: int
    concrete_directive_candidate: int
    themes_json: str
    evidence_excerpt: str


@dataclass(frozen=True)
class DatabaseAnalysis:
    """Complete result and source-population metadata."""

    files: tuple[OverlayFileRow, ...]
    sections: tuple[OverlaySectionRow, ...]
    exact_population_files: int
    exact_population_repositories: int
    exact_population_content_identities: int
    sqlite_total_changes: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _exact_population_predicate(alias: str = "f") -> str:
    prefix = f"{alias}." if alias else ""
    return (
        f"(lower({prefix}path) = 'claude.md' "
        f"OR lower({prefix}path) LIKE '%/claude.md')"
    )


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _import_line_starts(content: str) -> list[int]:
    """Match the newline model used by ``miner.claude_imports``."""
    starts = [0]
    for match in re.finditer("\n", content):
        if match.end() < len(content):
            starts.append(match.end())
    return starts


def _markdown_line_starts(content: str) -> list[int]:
    """Return offsets matching ``str.splitlines`` used by the heading parser."""
    starts: list[int] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    if content and not starts:
        starts.append(0)
    return starts


def _remove_spans(
    content: str,
    spans: Sequence[tuple[int, int]],
    *,
    base_offset: int = 0,
) -> str:
    """Remove global character spans from a string or source substring."""
    local: list[tuple[int, int]] = []
    limit = base_offset + len(content)
    for start, end in spans:
        if start >= limit or end <= base_offset:
            continue
        local.append((max(start, base_offset) - base_offset, min(end, limit) - base_offset))
    result = content
    for start, end in sorted(local, reverse=True):
        result = result[:start] + result[end:]
    return result


def _mask_html_comments(text: str) -> str:
    """Replace HTML comments with spaces while preserving line boundaries."""

    def replacement(match: re.Match[str]) -> str:
        return "".join(char if char in "\r\n" else " " for char in match.group(0))

    return HTML_COMMENT_RE.sub(replacement, text)


def _excerpt(text: str, maximum_characters: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= maximum_characters:
        return compact
    return compact[: maximum_characters - 1].rstrip() + "…"


def _json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_explicit_claude_overlay_heading(normalized_heading: str) -> bool:
    """Return whether a heading explicitly labels Claude-oriented material."""
    return bool(
        len(normalized_heading) <= MAX_OVERLAY_HEADING_CHARACTERS
        and CLAUDE_HEADING_TERM_RE.search(normalized_heading)
        and OVERLAY_HEADING_CUE_RE.search(normalized_heading)
    )


def _heading_label_kind(normalized_heading: str) -> str:
    if EXPLICIT_SPECIFICITY_RE.search(normalized_heading):
        return "explicit_specificity_or_adapter"
    if INSTRUCTION_HEADING_RE.search(normalized_heading):
        return "instruction_or_workflow"
    return "notes_configuration_or_context"


def _has_placeholder_language(text: str) -> bool:
    return any(pattern.search(text) for pattern in PLACEHOLDER_PATTERNS)


def _semantic_section_body(text: str) -> str:
    """Remove a common generated description that is not itself a directive."""
    return COMMON_CLAUDE_BOILERPLATE_RE.sub(" ", text)


def _themes(text: str) -> tuple[str, ...]:
    return tuple(
        name for name, pattern in THEME_PATTERNS.items() if pattern.search(text)
    )


def _active_agents_import_spans(content: str, source_path: str) -> tuple[ImportSpan, ...]:
    line_starts = _import_line_starts(content)
    spans: list[ImportSpan] = []
    for occurrence in extract_claude_import_occurrences(content):
        if occurrence.decision != "import_candidate":
            continue
        resolution = resolve_import_target(source_path, occurrence.raw_target)
        if resolution.target_basename.casefold() != TARGET_BASENAME:
            continue
        base = line_starts[occurrence.line_number - 1]
        spans.append(
            ImportSpan(
                start=base + occurrence.column_number - 1,
                end=base + occurrence.end_column_number - 1,
                line_number=occurrence.line_number,
                raw_token=occurrence.raw_token,
                raw_target=occurrence.raw_target,
            )
        )
    return tuple(spans)


def analyze_content(
    content: str,
    source_path: str,
    *,
    minimum_section_body_words: int = DEFAULT_MINIMUM_SECTION_BODY_WORDS,
    excerpt_characters: int = DEFAULT_EXCERPT_CHARACTERS,
) -> ContentAnalysis | None:
    """Analyze one file, returning ``None`` when no active target is present."""
    if minimum_section_body_words <= 0:
        raise ValueError("minimum_section_body_words must be positive")
    if excerpt_characters <= 0:
        raise ValueError("excerpt_characters must be positive")

    imports = _active_agents_import_spans(content, source_path)
    if not imports:
        return None
    span_pairs = [(item.start, item.end) for item in imports]
    residual = _remove_spans(content, span_pairs)
    residual_without_comments = _mask_html_comments(residual)
    last_import_end = max(item.end for item in imports)
    after_final_import = _mask_html_comments(content[last_import_end:])

    markdown_line_starts = _markdown_line_starts(content)
    headings = parse_headings(_mask_html_comments(content))
    sections: list[ContentSection] = []
    for index, heading in enumerate(headings):
        if not is_explicit_claude_overlay_heading(heading.normalized_name):
            continue
        heading_start = markdown_line_starts[heading.start_line]
        if heading_start <= last_import_end:
            continue

        body_start_line = heading.end_line + 1
        body_start = (
            markdown_line_starts[body_start_line]
            if body_start_line < len(markdown_line_starts)
            else len(content)
        )
        body_end = len(content)
        for following in headings[index + 1 :]:
            if following.level <= heading.level:
                body_end = markdown_line_starts[following.start_line]
                break

        body = content[body_start:body_end]
        body = _remove_spans(body, span_pairs, base_offset=body_start)
        body = _mask_html_comments(body)
        semantic_body = _semantic_section_body(body)
        body_words = _word_count(body)
        semantic_words = _word_count(semantic_body)
        directive = bool(ENGLISH_DIRECTIVE_CUE_RE.search(semantic_body))
        placeholder = _has_placeholder_language(semantic_body)
        concrete = bool(
            semantic_words >= minimum_section_body_words
            and directive
            and not placeholder
        )
        sections.append(
            ContentSection(
                heading=heading.text,
                normalized_heading=heading.normalized_name,
                heading_level=heading.level,
                heading_line=heading.start_line + 1,
                label_kind=_heading_label_kind(heading.normalized_name),
                body_word_count=body_words,
                semantic_body_word_count=semantic_words,
                has_english_directive_cue=int(directive),
                has_placeholder_language=int(placeholder),
                concrete_directive_candidate=int(concrete),
                themes=_themes(semantic_body),
                evidence_excerpt=_excerpt(body, excerpt_characters),
            )
        )

    form_counts = Counter(item.raw_token for item in imports)
    return ContentAnalysis(
        import_spans=imports,
        exact_literal_at_agents_occurrences=sum(
            item.raw_token == "@AGENTS.md" for item in imports
        ),
        target_form_counts=tuple(sorted(form_counts.items())),
        strict_import_only=int(not residual.strip("\ufeff \t\r\n")),
        no_residual_outside_html_comments=int(
            not residual_without_comments.strip("\ufeff \t\r\n")
        ),
        residual_word_count=_word_count(residual_without_comments),
        residual_after_final_import_word_count=_word_count(after_final_import),
        residual_excerpt=_excerpt(residual_without_comments, excerpt_characters),
        sections=tuple(sections),
    )


def _residual_class(analysis: ContentAnalysis, substantive_threshold: int) -> str:
    if analysis.strict_import_only:
        return "strict_import_only"
    if analysis.no_residual_outside_html_comments:
        return "imports_plus_html_comments_only"
    if analysis.residual_word_count == 0:
        return "markup_or_punctuation_only_residual"
    if analysis.residual_word_count < substantive_threshold:
        return "short_residual"
    return "substantive_residual"


def analyze_database(
    db_path: str | Path,
    *,
    substantive_word_threshold: int = DEFAULT_SUBSTANTIVE_WORD_THRESHOLD,
    minimum_section_body_words: int = DEFAULT_MINIMUM_SECTION_BODY_WORDS,
    excerpt_characters: int = DEFAULT_EXCERPT_CHARACTERS,
) -> DatabaseAnalysis:
    """Run the complete read-only database analysis."""
    if min(
        substantive_word_threshold,
        minimum_section_body_words,
        excerpt_characters,
    ) <= 0:
        raise ValueError("analysis thresholds must be positive")

    connection = _readonly_connection(db_path)
    predicate = _exact_population_predicate("f")
    try:
        population = connection.execute(
            f"""
            SELECT COUNT(*) AS files,
                   COUNT(DISTINCT f.repo_full_name) AS repositories,
                   COUNT(DISTINCT COALESCE(
                       f.content_hash, printf('id:%020d', f.id)
                   )) AS content_identities
            FROM files AS f
            WHERE {predicate}
            """
        ).fetchone()
        repo_counts = {
            str(row["repo_full_name"]): int(row["copies"])
            for row in connection.execute(
                f"""
                SELECT f.repo_full_name, COUNT(*) AS copies
                FROM files AS f
                WHERE {predicate}
                GROUP BY f.repo_full_name
                """
            )
        }
        hash_counts = {
            str(row["identity"]): int(row["copies"])
            for row in connection.execute(
                f"""
                SELECT COALESCE(f.content_hash, printf('id:%020d', f.id)) AS identity,
                       COUNT(*) AS copies
                FROM files AS f
                WHERE {predicate}
                GROUP BY identity
                """
            )
        }

        # This SQL is only a lossless ASCII prefilter.  The versioned Python
        # lexer below still decides whether an occurrence is active.
        query = f"""
            SELECT f.id, f.repo_full_name, f.path, f.html_url, f.content_hash,
                   f.size_bytes, f.content, r.stars, r.language
            FROM files AS f
            LEFT JOIN repos AS r ON r.full_name = f.repo_full_name
            WHERE {predicate}
              AND instr(f.content, '@') > 0
              AND instr(lower(f.content), 'agents.md') > 0
            ORDER BY lower(f.repo_full_name), f.repo_full_name,
                     lower(f.path), f.path, f.id
        """
        file_rows: list[OverlayFileRow] = []
        section_rows: list[OverlaySectionRow] = []
        for row in connection.execute(query):
            source_path = str(row["path"])
            content = str(row["content"] or "")
            analysis = analyze_content(
                content,
                source_path,
                minimum_section_body_words=minimum_section_body_words,
                excerpt_characters=excerpt_characters,
            )
            if analysis is None:
                continue

            identity = str(row["content_hash"] or f"id:{int(row['id']):020d}")
            headings = [item.heading for item in analysis.sections]
            themes = sorted(
                {theme for item in analysis.sections for theme in item.themes}
            )
            nonempty_sections = sum(
                item.semantic_body_word_count >= minimum_section_body_words
                for item in analysis.sections
            )
            placeholder_sections = sum(
                item.has_placeholder_language for item in analysis.sections
            )
            concrete_sections = sum(
                item.concrete_directive_candidate for item in analysis.sections
            )
            file_id = int(row["id"])
            repo = str(row["repo_full_name"])
            url = str(row["html_url"] or "")
            file_rows.append(
                OverlayFileRow(
                    file_id=file_id,
                    repo_full_name=repo,
                    source_path=source_path,
                    html_url=url,
                    content_hash=identity,
                    size_bytes=(
                        int(row["size_bytes"])
                        if row["size_bytes"] is not None
                        else None
                    ),
                    repository_stars=(
                        int(row["stars"]) if row["stars"] is not None else None
                    ),
                    repository_language=str(row["language"] or "Unknown"),
                    repository_exact_claude_files=repo_counts[repo],
                    exact_population_content_copies=hash_counts[identity],
                    agents_import_occurrences=len(analysis.import_spans),
                    exact_literal_at_agents_occurrences=(
                        analysis.exact_literal_at_agents_occurrences
                    ),
                    has_exact_literal_at_agents_import=int(
                        analysis.exact_literal_at_agents_occurrences > 0
                    ),
                    target_form_counts_json=_json_compact(
                        [
                            {"raw_token": token, "occurrences": count}
                            for token, count in analysis.target_form_counts
                        ]
                    ),
                    first_agents_import_line=min(
                        item.line_number for item in analysis.import_spans
                    ),
                    last_agents_import_line=max(
                        item.line_number for item in analysis.import_spans
                    ),
                    strict_import_only=analysis.strict_import_only,
                    no_residual_outside_html_comments=(
                        analysis.no_residual_outside_html_comments
                    ),
                    residual_class=_residual_class(
                        analysis, substantive_word_threshold
                    ),
                    residual_word_count=analysis.residual_word_count,
                    residual_after_final_import_word_count=(
                        analysis.residual_after_final_import_word_count
                    ),
                    substantive_residual=int(
                        analysis.residual_word_count
                        >= substantive_word_threshold
                    ),
                    substantive_residual_after_final_import=int(
                        analysis.residual_after_final_import_word_count
                        >= substantive_word_threshold
                    ),
                    explicit_claude_labeled_sections_after_final_import=len(
                        analysis.sections
                    ),
                    labeled_sections_with_nonempty_body=nonempty_sections,
                    labeled_sections_with_placeholder_language=(
                        placeholder_sections
                    ),
                    concrete_labeled_directive_sections=concrete_sections,
                    has_concrete_labeled_directive_candidate=int(
                        concrete_sections > 0
                    ),
                    overlay_headings_json=_json_compact(headings),
                    overlay_themes_json=_json_compact(themes),
                    residual_excerpt=analysis.residual_excerpt,
                )
            )
            for section in analysis.sections:
                section_rows.append(
                    OverlaySectionRow(
                        file_id=file_id,
                        repo_full_name=repo,
                        source_path=source_path,
                        html_url=url,
                        content_hash=identity,
                        heading=section.heading,
                        normalized_heading=section.normalized_heading,
                        heading_level=section.heading_level,
                        heading_line=section.heading_line,
                        label_kind=section.label_kind,
                        body_word_count=section.body_word_count,
                        semantic_body_word_count=section.semantic_body_word_count,
                        has_english_directive_cue=(
                            section.has_english_directive_cue
                        ),
                        has_placeholder_language=(
                            section.has_placeholder_language
                        ),
                        concrete_directive_candidate=(
                            section.concrete_directive_candidate
                        ),
                        themes_json=_json_compact(list(section.themes)),
                        evidence_excerpt=section.evidence_excerpt,
                    )
                )

        changes = connection.total_changes
        return DatabaseAnalysis(
            files=tuple(file_rows),
            sections=tuple(section_rows),
            exact_population_files=int(population["files"]),
            exact_population_repositories=int(population["repositories"]),
            exact_population_content_identities=int(
                population["content_identities"]
            ),
            sqlite_total_changes=changes,
        )
    finally:
        connection.close()


def _rounded(value: float) -> float:
    return round(value, 6)


def _share(count: int, denominator: int) -> float:
    return _rounded(count / denominator) if denominator else 0.0


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("cannot describe an empty population")
    ordered = sorted(values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": min(values),
        "p25": _nearest_rank(values, 0.25),
        "median": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "maximum": max(values),
        "mean": _rounded(sum(values) / len(values)),
    }


def _cohort(
    rows: Sequence[OverlayFileRow],
    predicate: Callable[[OverlayFileRow], bool],
    *,
    candidate_denominator: int,
    exact_denominator: int,
) -> dict[str, int | float]:
    selected = [row for row in rows if predicate(row)]
    return {
        "files": len(selected),
        "repositories": len({row.repo_full_name for row in selected}),
        "content_identities": len({row.content_hash for row in selected}),
        "file_share_of_agents_import_files": _share(
            len(selected), candidate_denominator
        ),
        "file_share_of_exact_claude_population": _share(
            len(selected), exact_denominator
        ),
    }


def build_summary(
    analysis: DatabaseAnalysis,
    *,
    database_path: str | Path,
    database_sha256_before: str,
    database_sha256_after: str,
    substantive_word_threshold: int = DEFAULT_SUBSTANTIVE_WORD_THRESHOLD,
    minimum_section_body_words: int = DEFAULT_MINIMUM_SECTION_BODY_WORDS,
) -> dict[str, object]:
    """Build a deterministic, JSON-ready summary."""
    rows = analysis.files
    sections = analysis.sections
    candidate_count = len(rows)

    def cohort(predicate: Callable[[OverlayFileRow], bool]) -> dict[str, int | float]:
        return _cohort(
            rows,
            predicate,
            candidate_denominator=candidate_count,
            exact_denominator=analysis.exact_population_files,
        )

    target_forms: Counter[str] = Counter()
    for row in rows:
        for item in json.loads(row.target_form_counts_json):
            target_forms[str(item["raw_token"])] += int(item["occurrences"])
    heading_counts = Counter(section.normalized_heading for section in sections)
    label_kind_counts = Counter(section.label_kind for section in sections)

    theme_summary: dict[str, object] = {}
    for theme in THEME_PATTERNS:
        matching_sections = [
            section
            for section in sections
            if theme in json.loads(section.themes_json)
        ]
        theme_summary[theme] = {
            "sections": len(matching_sections),
            "files": len({section.file_id for section in matching_sections}),
            "repositories": len(
                {section.repo_full_name for section in matching_sections}
            ),
        }

    strict = sum(row.strict_import_only for row in rows)
    comments_only = sum(
        row.no_residual_outside_html_comments and not row.strict_import_only
        for row in rows
    )
    word_residual = sum(row.residual_word_count > 0 for row in rows)
    markup_only = candidate_count - strict - comments_only - word_residual

    classifications = {
        "strict_import_only": cohort(lambda row: bool(row.strict_import_only)),
        "no_residual_outside_html_comments": cohort(
            lambda row: bool(row.no_residual_outside_html_comments)
        ),
        "at_least_one_residual_word": cohort(
            lambda row: row.residual_word_count > 0
        ),
        "substantive_residual": cohort(
            lambda row: bool(row.substantive_residual)
        ),
        "substantive_residual_after_final_import": cohort(
            lambda row: bool(row.substantive_residual_after_final_import)
        ),
        "explicit_claude_labeled_section_after_final_import": cohort(
            lambda row: row.explicit_claude_labeled_sections_after_final_import > 0
        ),
        "labeled_section_with_nonempty_body": cohort(
            lambda row: row.labeled_sections_with_nonempty_body > 0
        ),
        "labeled_section_with_placeholder_language": cohort(
            lambda row: row.labeled_sections_with_placeholder_language > 0
        ),
        "concrete_labeled_directive_candidate": cohort(
            lambda row: bool(row.has_concrete_labeled_directive_candidate)
        ),
    }

    literal_rows = [row for row in rows if row.has_exact_literal_at_agents_import]
    literal_denominator = len(literal_rows)

    def literal_cohort(
        predicate: Callable[[OverlayFileRow], bool]
    ) -> dict[str, int | float]:
        selected = [row for row in literal_rows if predicate(row)]
        return {
            "files": len(selected),
            "repositories": len({row.repo_full_name for row in selected}),
            "content_identities": len({row.content_hash for row in selected}),
            "file_share_of_literal_subset": _share(
                len(selected), literal_denominator
            ),
        }

    summary: dict[str, object] = {
        "analysis": "CLAUDE.md AGENTS.md interoperability shims and overlays",
        "analysis_version": ANALYSIS_VERSION,
        "import_detector_version": IMPORT_DETECTOR_VERSION,
        "population_definition": POPULATION_DEFINITION,
        "database": {
            "path": Path(database_path).as_posix(),
            "sha256_before": database_sha256_before,
            "sha256_after": database_sha256_after,
            "sha256_unchanged": database_sha256_before
            == database_sha256_after,
            "sqlite_total_changes": analysis.sqlite_total_changes,
            "sqlite_library_version": sqlite3.sqlite_version,
        },
        "source_population": {
            "files": analysis.exact_population_files,
            "repositories": analysis.exact_population_repositories,
            "content_identities": analysis.exact_population_content_identities,
        },
        "agents_target_basename_candidates": {
            **cohort(lambda row: True),
            "occurrences": sum(row.agents_import_occurrences for row in rows),
            "target_form_counts": [
                {"raw_token": token, "occurrences": count}
                for token, count in sorted(
                    target_forms.items(),
                    key=lambda item: (-item[1], item[0].casefold(), item[0]),
                )
            ],
            "residual_word_count_distribution": _distribution(
                [row.residual_word_count for row in rows]
            ),
            "residual_word_count_distribution_when_positive": _distribution(
                [row.residual_word_count for row in rows if row.residual_word_count]
            ),
            "residual_after_final_import_word_count_distribution": _distribution(
                [row.residual_after_final_import_word_count for row in rows]
            ),
        },
        "exact_literal_at_agents_subset": {
            "occurrences": sum(
                row.exact_literal_at_agents_occurrences for row in literal_rows
            ),
            **literal_cohort(lambda row: True),
            "strict_import_only": literal_cohort(
                lambda row: bool(row.strict_import_only)
            ),
            "substantive_residual": literal_cohort(
                lambda row: bool(row.substantive_residual)
            ),
            "substantive_residual_after_final_import": literal_cohort(
                lambda row: bool(row.substantive_residual_after_final_import)
            ),
            "explicit_claude_labeled_section_after_final_import": literal_cohort(
                lambda row: row.explicit_claude_labeled_sections_after_final_import
                > 0
            ),
            "concrete_labeled_directive_candidate": literal_cohort(
                lambda row: bool(row.has_concrete_labeled_directive_candidate)
            ),
        },
        "classification_cohorts": classifications,
        "exclusive_residual_partition": {
            "strict_import_only": strict,
            "imports_plus_html_comments_only": comments_only,
            "markup_or_punctuation_only_residual": markup_only,
            "at_least_one_residual_word": word_residual,
            "sum": strict + comments_only + markup_only + word_residual,
        },
        "explicit_overlay_sections": {
            "sections": len(sections),
            "files": len({section.file_id for section in sections}),
            "repositories": len(
                {section.repo_full_name for section in sections}
            ),
            "content_identities": len(
                {section.content_hash for section in sections}
            ),
            "concrete_directive_candidate_sections": sum(
                section.concrete_directive_candidate for section in sections
            ),
            "placeholder_language_sections": sum(
                section.has_placeholder_language for section in sections
            ),
            "label_kind_counts": dict(sorted(label_kind_counts.items())),
            "leading_normalized_headings": [
                {"heading": heading, "sections": count}
                for heading, count in heading_counts.most_common(50)
            ],
            "overlapping_lexical_theme_signals": theme_summary,
        },
        "parameters": {
            "substantive_word_threshold": substantive_word_threshold,
            "minimum_section_body_words": minimum_section_body_words,
            "word_definition": (
                "Unicode alphanumeric runs, optionally joined internally by "
                "apostrophes or hyphens; underscores are separators"
            ),
            "exact_literal_definition": "raw active token exactly @AGENTS.md",
        },
        "operational_definitions": {
            "agents_import": (
                "Phase 2r active @path syntax candidate whose case-insensitive "
                "target basename is AGENTS.md"
            ),
            "strict_import_only": (
                "removing all active AGENTS.md import tokens leaves only a BOM "
                "and whitespace"
            ),
            "no_residual_outside_html_comments": (
                "after removing active imports and HTML comments, only a BOM "
                "and whitespace remain"
            ),
            "substantive_residual": (
                f"at least {substantive_word_threshold} residual word tokens "
                "after active imports and HTML comments are removed"
            ),
            "substantive_residual_after_final_import": (
                f"at least {substantive_word_threshold} word tokens occur after "
                "the final active AGENTS.md import, excluding HTML comments"
            ),
            "explicit_claude_labeled_section_after_final_import": (
                "an ATX or Setext heading after the final import contains an "
                "explicit Claude/Claude Code term and an overlay cue such as "
                "specific, instructions, workflow, notes, adapter, or context"
            ),
            "concrete_labeled_directive_candidate": (
                "an explicit labeled section has the configured minimum body "
                "words, an English directive cue, and no recognized placeholder"
            ),
        },
        "documentation_basis": {
            "url": DOCUMENTATION_URL,
            "accessed": DOCUMENTATION_ACCESSED,
        },
        "artifact_rows": {
            "files": len(rows),
            "sections": len(sections),
        },
        "invariants": {
            "every_output_file_has_an_active_agents_target": all(
                row.agents_import_occurrences > 0 for row in rows
            ),
            "exclusive_residual_partition_matches_candidate_files": (
                strict + comments_only + markup_only + word_residual
                == candidate_count
            ),
            "section_rows_match_file_section_counts": len(sections)
            == sum(
                row.explicit_claude_labeled_sections_after_final_import
                for row in rows
            ),
        },
        "limitations": [
            "An import-syntax candidate does not establish that AGENTS.md exists, was approved, recursively expanded, or loaded at runtime.",
            "The ten-word substantive threshold is an operational sensitivity cutoff, not a semantic boundary.",
            "HTML comments are excluded only for this residual-text measure; the analysis does not claim that comments are absent from model context.",
            "The explicit-heading detector is lexical and primarily English-oriented; unlabeled and non-English Claude-specific overlays can be missed.",
            "The directive and placeholder rules are audit aids rather than validated semantic labels and can produce false positives or false negatives.",
            "File counts are copy-weighted; repository and content-identity counts are reported alongside them to expose duplication.",
        ],
    }
    return summary


def _write_dataclass_csv(
    rows: Sequence[object],
    row_type: type,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(row_type)]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return output


def write_files_csv(rows: Sequence[OverlayFileRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, OverlayFileRow, output_path)


def write_sections_csv(
    rows: Sequence[OverlaySectionRow], output_path: str | Path
) -> Path:
    return _write_dataclass_csv(rows, OverlaySectionRow, output_path)


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mined.db")
    parser.add_argument(
        "--files-output",
        default="article/claude_agents_overlay_files.csv",
    )
    parser.add_argument(
        "--sections-output",
        default="article/claude_agents_overlay_sections.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="article/claude_agents_overlay_summary.json",
    )
    parser.add_argument(
        "--substantive-word-threshold",
        type=positive_int,
        default=DEFAULT_SUBSTANTIVE_WORD_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-section-body-words",
        type=positive_int,
        default=DEFAULT_MINIMUM_SECTION_BODY_WORDS,
    )
    parser.add_argument(
        "--excerpt-characters",
        type=positive_int,
        default=DEFAULT_EXCERPT_CHARACTERS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    database_hash_before = _sha256_file(arguments.db)
    analysis = analyze_database(
        arguments.db,
        substantive_word_threshold=arguments.substantive_word_threshold,
        minimum_section_body_words=arguments.minimum_section_body_words,
        excerpt_characters=arguments.excerpt_characters,
    )
    database_hash_after = _sha256_file(arguments.db)
    summary = build_summary(
        analysis,
        database_path=arguments.db,
        database_sha256_before=database_hash_before,
        database_sha256_after=database_hash_after,
        substantive_word_threshold=arguments.substantive_word_threshold,
        minimum_section_body_words=arguments.minimum_section_body_words,
    )

    files_path = write_files_csv(analysis.files, arguments.files_output)
    sections_path = write_sections_csv(analysis.sections, arguments.sections_output)
    summary_path = write_summary_json(summary, arguments.summary_output)
    concrete = summary["classification_cohorts"][  # type: ignore[index]
        "concrete_labeled_directive_candidate"
    ]["files"]  # type: ignore[index]
    print(
        f"Wrote {len(analysis.files):,} file rows to {files_path}, "
        f"{len(analysis.sections):,} labeled section rows to {sections_path}, "
        f"and {summary_path}."
    )
    print(
        f"Concrete labeled directive candidates: {concrete:,}; "
        f"SQLite changes: {analysis.sqlite_total_changes}; database SHA-256 "
        f"unchanged: {database_hash_before == database_hash_after}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
