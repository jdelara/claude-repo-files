"""Analyze Markdown section structure in ``mined.db`` without modifying it.

The default population is files whose case-insensitive basename is exactly
``CLAUDE.md``. Content is parsed once per SHA-256 hash, then weighted by its
number of stored copies to produce both full-population and content-deduplicated
statistics.

The self-contained block parser recognizes ATX and Setext headings while
ignoring heading-like lines inside backtick or tilde fenced code blocks. A
section length is the non-overlapping content segment after one heading and
before the next heading of any level.

Example:

    python scripts/markdown_section_stats.py \
        --db mined.db \
        --scope exact-claude \
        --families-output article/section_structure_families.csv \
        --headings-output article/common_heading_names.csv \
        --styles-output article/section_content_styles.csv \
        --list-distribution-output article/section_list_item_distribution.csv \
        --summary-output article/section_structure_summary.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCOPES = ("exact-claude", "markdown", "all")
ATX_HEADING_RE = re.compile(r"^[ ]{0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
FENCE_OPEN_RE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
SETEXT_UNDERLINE_RE = re.compile(r"^[ ]{0,3}(=+|-+)[ \t]*$")
LIST_ITEM_RE = re.compile(
    r"^(?P<indent> *)(?P<marker>[-+*]|\d{1,9}[.)])(?P<spacing>[ \t]+)(?P<body>.*)$"
)
TASK_LIST_ITEM_RE = re.compile(r"^\[(?: |x|X)\](?:[ \t]+|$)")
HORIZONTAL_RULE_RE = re.compile(
    r"^[ ]{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
)
LINK_DEFINITION_RE = re.compile(r"^[ ]{0,3}\[[^]]+\]:[ \t]*\S+")
HTML_BLOCK_LINE_RE = re.compile(
    r"^[ ]{0,3}</?(?:address|article|aside|base|basefont|blockquote|body|caption|"
    r"center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|"
    r"figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|"
    r"legend|li|link|main|menu|menuitem|nav|noframes|ol|optgroup|option|p|"
    r"param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|"
    r"track|ul)(?:[ \t]|/?>|$)",
    re.IGNORECASE,
)
BLOCKQUOTE_PREFIX_RE = re.compile(r"^[ ]{0,3}>[ ]?")
MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
NUMBERING_PREFIX_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)(?:[.)\-:]?)|(?:[ivxlcdm]+|[a-z])[.)\-:])[ \t]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    normalized_name: str
    start_line: int
    end_line: int


CONTENT_STYLES = (
    "empty",
    "prose_only",
    "list_only",
    "other_only",
    "list_and_prose",
    "prose_and_other",
    "list_and_other",
    "list_prose_and_other",
)


@dataclass(frozen=True)
class SectionBodyMetrics:
    line_count: int
    word_count: int
    nonblank_line_count: int
    list_item_count: int
    unordered_list_item_count: int
    ordered_list_item_count: int
    task_list_item_count: int
    standalone_prose_block_count: int
    list_content_line_count: int
    prose_content_line_count: int
    other_content_line_count: int
    content_style: str


@dataclass(frozen=True)
class DocumentStructure:
    headings: tuple[Heading, ...]
    top_level_sections: int
    maximum_relative_depth: int
    skipped_level_jumps: int
    starts_below_h1: bool
    h1_count: int
    section_line_counts: tuple[int, ...]
    section_word_counts: tuple[int, ...]
    section_bodies: tuple[SectionBodyMetrics, ...]
    empty_sections: int


@dataclass(frozen=True)
class ContentFamilyStructure:
    content_hash: str
    copies: int
    representative_repo: str
    representative_path: str
    size_bytes: int | None
    section_count: int
    top_level_sections: int
    maximum_relative_depth: int
    skipped_level_jumps: int
    starts_below_h1: bool
    h1_count: int
    empty_sections: int
    sections_with_list_items: int
    sections_with_standalone_prose: int
    total_list_items: int
    unordered_list_items: int
    ordered_list_items: int
    task_list_items: int
    total_standalone_prose_blocks: int
    list_content_lines: int
    prose_content_lines: int
    other_content_lines: int
    list_only_sections: int
    prose_only_sections: int
    list_and_prose_sections: int
    other_only_sections: int
    sections_with_other_content: int
    median_list_items_per_section: int | None
    mean_list_items_per_section: float | None
    maximum_list_items_per_section: int | None
    median_section_lines: int | None
    mean_section_lines: float | None
    maximum_section_lines: int | None
    median_section_words: int | None
    mean_section_words: float | None
    maximum_section_words: int | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _scope_predicate(scope: str, *, alias: str = "") -> str:
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope!r}")
    prefix = f"{alias}." if alias else ""
    if scope == "exact-claude":
        return (
            f"(lower({prefix}path) = 'claude.md' "
            f"OR lower({prefix}path) LIKE '%/claude.md')"
        )
    if scope == "markdown":
        return f"lower({prefix}path) LIKE '%.md'"
    return "1 = 1"


def _scope_definition(scope: str) -> str:
    return {
        "exact-claude": "case-insensitive basename equal to CLAUDE.md",
        "markdown": "case-insensitive paths ending in .md",
        "all": "all stored file records",
    }[scope]


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def normalize_heading_name(value: str) -> str:
    """Conservatively normalize a heading for frequency analysis."""
    text = html.unescape(value)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = re.sub(r"[`*_~]", "", text)
    text = NUMBERING_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t:;,.!?-–—")
    return text.casefold() or "(empty)"


def _atx_heading(line: str) -> tuple[int, str] | None:
    match = ATX_HEADING_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    text = match.group(2).strip()
    text = re.sub(r"[ \t]+#+[ \t]*$", "", text).strip()
    return level, text


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
    indentation = len(line) - len(stripped)
    if indentation > 3:
        return False
    match = re.fullmatch(re.escape(character) + r"+([ \t]*)", stripped)
    return match is not None and len(stripped.rstrip(" \t")) >= minimum_length


def _can_be_setext_text(line: str) -> bool:
    if not line.strip() or len(line) - len(line.lstrip(" ")) > 3:
        return False
    stripped = line.lstrip(" ")
    return not re.match(r"(?:>|[-+*][ \t]+|\d+[.)][ \t]+)", stripped)


def _strip_blockquote_prefixes(line: str) -> str:
    """Remove Markdown blockquote containers before classifying their contents."""
    remainder = line
    while True:
        match = BLOCKQUOTE_PREFIX_RE.match(remainder)
        if match is None:
            return remainder
        remainder = remainder[match.end() :]


def _is_other_block_line(line: str) -> bool:
    indentation = len(line) - len(line.lstrip(" "))
    return bool(
        indentation >= 4
        or line.count("|") >= 2
        or LINK_DEFINITION_RE.match(line)
        or HORIZONTAL_RULE_RE.match(line)
        or HTML_BLOCK_LINE_RE.match(line)
    )


def _section_content_style(
    *, list_lines: int, prose_lines: int, other_lines: int
) -> str:
    if not (list_lines or prose_lines or other_lines):
        return "empty"
    if list_lines and prose_lines and other_lines:
        return "list_prose_and_other"
    if list_lines and prose_lines:
        return "list_and_prose"
    if list_lines and other_lines:
        return "list_and_other"
    if prose_lines and other_lines:
        return "prose_and_other"
    if list_lines:
        return "list_only"
    if prose_lines:
        return "prose_only"
    return "other_only"


def _measure_section_body(lines: Sequence[str]) -> SectionBodyMetrics:
    """Classify direct section-body lines with a lightweight Markdown scanner.

    List continuation lines, nested items, and fenced blocks indented under an
    active list are allocated to list content. Paragraphs nested in list items
    are intentionally not counted as standalone prose blocks.
    """
    list_item_count = 0
    unordered_list_item_count = 0
    ordered_list_item_count = 0
    task_list_item_count = 0
    standalone_prose_block_count = 0
    list_content_line_count = 0
    prose_content_line_count = 0
    other_content_line_count = 0

    active_list_content_indent: int | None = None
    list_lazy_continuation = False
    fence_character: str | None = None
    fence_length = 0
    fence_kind: str | None = None
    fence_list_indent = 0
    previous_kind: str | None = None

    def record(kind: str) -> None:
        nonlocal standalone_prose_block_count
        nonlocal list_content_line_count
        nonlocal prose_content_line_count
        nonlocal other_content_line_count
        nonlocal previous_kind
        if kind == "list":
            list_content_line_count += 1
        elif kind == "prose":
            prose_content_line_count += 1
            if previous_kind != "prose":
                standalone_prose_block_count += 1
        elif kind == "other":
            other_content_line_count += 1
        else:  # pragma: no cover - internal invariant
            raise AssertionError(f"unsupported section content kind: {kind!r}")
        previous_kind = kind

    for raw_line in lines:
        line = _strip_blockquote_prefixes(raw_line.expandtabs(4))
        if fence_character is not None:
            if line.strip():
                assert fence_kind is not None
                record(fence_kind)
            else:
                previous_kind = None
            close_candidate = line
            if fence_kind == "list" and line.startswith(" " * fence_list_indent):
                close_candidate = line[fence_list_indent:]
            if _fence_close(close_candidate, fence_character, fence_length):
                fence_character = None
                fence_length = 0
                if fence_kind == "list":
                    list_lazy_continuation = True
                fence_kind = None
                fence_list_indent = 0
            continue

        if not line.strip():
            previous_kind = None
            list_lazy_continuation = False
            continue

        indentation = len(line) - len(line.lstrip(" "))
        horizontal_rule = HORIZONTAL_RULE_RE.match(line) is not None
        item = None if horizontal_rule else LIST_ITEM_RE.match(line)
        if item is not None:
            item_indent = len(item.group("indent"))
            item_belongs_to_active_list = (
                active_list_content_indent is not None
                and item_indent >= active_list_content_indent
            )
            if item_indent <= 3 or item_belongs_to_active_list:
                marker = item.group("marker")
                if marker[0] in "-+*":
                    unordered_list_item_count += 1
                else:
                    ordered_list_item_count += 1
                list_item_count += 1
                body = item.group("body")
                if TASK_LIST_ITEM_RE.match(body):
                    task_list_item_count += 1

                item_content_indent = (
                    item_indent + len(marker) + len(item.group("spacing"))
                )
                if not item_belongs_to_active_list:
                    active_list_content_indent = item_content_indent
                record("list")
                list_lazy_continuation = True

                opening = _fence_open(body)
                if opening is not None:
                    fence_character, fence_length = opening
                    fence_kind = "list"
                    fence_list_indent = active_list_content_indent
                continue

        if active_list_content_indent is not None and indentation >= active_list_content_indent:
            list_candidate = line[active_list_content_indent:]
            opening = _fence_open(list_candidate)
            record("list")
            list_lazy_continuation = True
            if opening is not None:
                fence_character, fence_length = opening
                fence_kind = "list"
                fence_list_indent = active_list_content_indent
            continue

        opening = _fence_open(line)
        if opening is not None:
            active_list_content_indent = None
            list_lazy_continuation = False
            record("other")
            fence_character, fence_length = opening
            fence_kind = "other"
            continue

        if _is_other_block_line(line):
            active_list_content_indent = None
            list_lazy_continuation = False
            record("other")
            continue

        if active_list_content_indent is not None and list_lazy_continuation:
            record("list")
            continue

        active_list_content_indent = None
        list_lazy_continuation = False
        record("prose")

    nonblank_line_count = (
        list_content_line_count + prose_content_line_count + other_content_line_count
    )
    style = _section_content_style(
        list_lines=list_content_line_count,
        prose_lines=prose_content_line_count,
        other_lines=other_content_line_count,
    )
    assert list_item_count == unordered_list_item_count + ordered_list_item_count
    assert task_list_item_count <= list_item_count
    assert style in CONTENT_STYLES
    return SectionBodyMetrics(
        line_count=len(lines),
        word_count=len("\n".join(lines).split()),
        nonblank_line_count=nonblank_line_count,
        list_item_count=list_item_count,
        unordered_list_item_count=unordered_list_item_count,
        ordered_list_item_count=ordered_list_item_count,
        task_list_item_count=task_list_item_count,
        standalone_prose_block_count=standalone_prose_block_count,
        list_content_line_count=list_content_line_count,
        prose_content_line_count=prose_content_line_count,
        other_content_line_count=other_content_line_count,
        content_style=style,
    )


def parse_headings(content: str) -> tuple[Heading, ...]:
    """Parse ATX/Setext headings outside fenced code blocks."""
    lines = content.splitlines()
    headings: list[Heading] = []
    fence_character: str | None = None
    fence_length = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        if fence_character is not None:
            if _fence_close(line, fence_character, fence_length):
                fence_character = None
                fence_length = 0
            index += 1
            continue

        opening = _fence_open(line)
        if opening is not None:
            fence_character, fence_length = opening
            index += 1
            continue

        atx = _atx_heading(line)
        if atx is not None:
            level, text = atx
            headings.append(
                Heading(
                    level=level,
                    text=text,
                    normalized_name=normalize_heading_name(text),
                    start_line=index,
                    end_line=index,
                )
            )
            index += 1
            continue

        if index + 1 < len(lines) and _can_be_setext_text(line):
            underline = SETEXT_UNDERLINE_RE.match(lines[index + 1])
            if underline is not None:
                text = line.strip()
                level = 1 if underline.group(1).startswith("=") else 2
                headings.append(
                    Heading(
                        level=level,
                        text=text,
                        normalized_name=normalize_heading_name(text),
                        start_line=index,
                        end_line=index + 1,
                    )
                )
                index += 2
                continue

        index += 1

    return tuple(headings)


def parse_document_structure(content: str) -> DocumentStructure:
    lines = content.splitlines()
    headings = parse_headings(content)
    if not headings:
        return DocumentStructure(
            headings=(),
            top_level_sections=0,
            maximum_relative_depth=0,
            skipped_level_jumps=0,
            starts_below_h1=False,
            h1_count=0,
            section_line_counts=(),
            section_word_counts=(),
            section_bodies=(),
            empty_sections=0,
        )

    stack: list[Heading] = []
    top_level_sections = 0
    maximum_depth = 0
    for heading in headings:
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        if not stack:
            top_level_sections += 1
        stack.append(heading)
        maximum_depth = max(maximum_depth, len(stack))

    skipped_jumps = sum(
        current.level > previous.level + 1
        for previous, current in zip(headings, headings[1:])
    )
    section_line_counts: list[int] = []
    section_word_counts: list[int] = []
    section_bodies: list[SectionBodyMetrics] = []
    for index, heading in enumerate(headings):
        start = heading.end_line + 1
        stop = headings[index + 1].start_line if index + 1 < len(headings) else len(lines)
        section_lines = lines[start:stop]
        body = _measure_section_body(section_lines)
        section_bodies.append(body)
        section_line_counts.append(body.line_count)
        section_word_counts.append(body.word_count)

    return DocumentStructure(
        headings=headings,
        top_level_sections=top_level_sections,
        maximum_relative_depth=maximum_depth,
        skipped_level_jumps=skipped_jumps,
        starts_below_h1=headings[0].level > 1,
        h1_count=sum(heading.level == 1 for heading in headings),
        section_line_counts=tuple(section_line_counts),
        section_word_counts=tuple(section_word_counts),
        section_bodies=tuple(section_bodies),
        empty_sections=sum(value == 0 for value in section_word_counts),
    )


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _family_value_summary(values: Sequence[int]) -> tuple[int, float, int] | tuple[None, None, None]:
    if not values:
        return None, None, None
    return _nearest_rank(values, 0.50), sum(values) / len(values), max(values)


def _describe_weighted(counter: Counter[int]) -> dict[str, int | float] | None:
    count = sum(counter.values())
    if count == 0:
        return None
    total = sum(value * weight for value, weight in counter.items())

    def quantile(probability: float) -> int:
        target = max(1, math.ceil(probability * count))
        cumulative = 0
        for value in sorted(counter):
            cumulative += counter[value]
            if cumulative >= target:
                return value
        raise AssertionError("weighted quantile target was not reached")

    return {
        "count": count,
        "total": total,
        "min": min(counter),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "mean": total / count,
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": max(counter),
        "zero": counter.get(0, 0),
    }


class PopulationAccumulator:
    def __init__(self) -> None:
        self.files = 0
        self.files_with_headings = 0
        self.files_with_skipped_jumps = 0
        self.files_starting_below_h1 = 0
        self.files_without_h1 = 0
        self.headed_files_without_h1 = 0
        self.files_with_multiple_h1 = 0
        self.files_with_empty_sections = 0
        self.files_with_list_bearing_sections = 0
        self.files_with_standalone_prose_sections = 0
        self.total_sections = 0
        self.total_skipped_jumps = 0
        self.total_empty_sections = 0
        self.total_list_items = 0
        self.total_unordered_list_items = 0
        self.total_ordered_list_items = 0
        self.total_task_list_items = 0
        self.total_standalone_prose_blocks = 0
        self.total_list_content_lines = 0
        self.total_prose_content_lines = 0
        self.total_other_content_lines = 0
        self.section_counts: Counter[int] = Counter()
        self.top_level_counts: Counter[int] = Counter()
        self.depth_counts: Counter[int] = Counter()
        self.headed_section_counts: Counter[int] = Counter()
        self.headed_top_level_counts: Counter[int] = Counter()
        self.headed_depth_counts: Counter[int] = Counter()
        self.jump_counts: Counter[int] = Counter()
        self.section_line_counts: Counter[int] = Counter()
        self.section_word_counts: Counter[int] = Counter()
        self.list_item_counts: Counter[int] = Counter()
        self.list_item_counts_with_items: Counter[int] = Counter()
        self.standalone_prose_block_counts: Counter[int] = Counter()
        self.content_style_counts: Counter[str] = Counter()

    def add(self, structure: DocumentStructure, weight: int) -> None:
        section_count = len(structure.headings)
        self.files += weight
        self.files_with_headings += weight if section_count else 0
        self.files_with_skipped_jumps += weight if structure.skipped_level_jumps else 0
        self.files_starting_below_h1 += weight if structure.starts_below_h1 else 0
        self.files_without_h1 += weight if structure.h1_count == 0 else 0
        self.headed_files_without_h1 += (
            weight if section_count and structure.h1_count == 0 else 0
        )
        self.files_with_multiple_h1 += weight if structure.h1_count > 1 else 0
        self.files_with_empty_sections += weight if structure.empty_sections else 0
        self.files_with_list_bearing_sections += (
            weight if any(body.list_item_count for body in structure.section_bodies) else 0
        )
        self.files_with_standalone_prose_sections += (
            weight
            if any(body.standalone_prose_block_count for body in structure.section_bodies)
            else 0
        )
        self.total_sections += section_count * weight
        self.total_skipped_jumps += structure.skipped_level_jumps * weight
        self.total_empty_sections += structure.empty_sections * weight
        self.section_counts[section_count] += weight
        self.top_level_counts[structure.top_level_sections] += weight
        self.depth_counts[structure.maximum_relative_depth] += weight
        if section_count:
            self.headed_section_counts[section_count] += weight
            self.headed_top_level_counts[structure.top_level_sections] += weight
            self.headed_depth_counts[structure.maximum_relative_depth] += weight
        self.jump_counts[structure.skipped_level_jumps] += weight
        for value in structure.section_line_counts:
            self.section_line_counts[value] += weight
        for value in structure.section_word_counts:
            self.section_word_counts[value] += weight
        for body in structure.section_bodies:
            self.total_list_items += body.list_item_count * weight
            self.total_unordered_list_items += body.unordered_list_item_count * weight
            self.total_ordered_list_items += body.ordered_list_item_count * weight
            self.total_task_list_items += body.task_list_item_count * weight
            self.total_standalone_prose_blocks += (
                body.standalone_prose_block_count * weight
            )
            self.total_list_content_lines += body.list_content_line_count * weight
            self.total_prose_content_lines += body.prose_content_line_count * weight
            self.total_other_content_lines += body.other_content_line_count * weight
            self.list_item_counts[body.list_item_count] += weight
            if body.list_item_count:
                self.list_item_counts_with_items[body.list_item_count] += weight
            self.standalone_prose_block_counts[
                body.standalone_prose_block_count
            ] += weight
            self.content_style_counts[body.content_style] += weight

    def as_dict(self) -> dict[str, object]:
        headed = self.files_with_headings
        classified_lines = (
            self.total_list_content_lines
            + self.total_prose_content_lines
            + self.total_other_content_lines
        )
        sections_with_lists = self.total_sections - self.list_item_counts.get(0, 0)
        sections_with_prose = self.total_sections - self.standalone_prose_block_counts.get(
            0, 0
        )
        sections_with_other = sum(
            count
            for style, count in self.content_style_counts.items()
            if "other" in style
        )
        style_summary = {
            style: {
                "sections": self.content_style_counts.get(style, 0),
                "share": (
                    self.content_style_counts.get(style, 0) / self.total_sections
                    if self.total_sections
                    else 0.0
                ),
            }
            for style in CONTENT_STYLES
        }
        return {
            "files": self.files,
            "files_with_headings": headed,
            "files_with_heading_share": headed / self.files if self.files else 0.0,
            "files_without_headings": self.files - headed,
            "explicit_sections": self.total_sections,
            "files_with_skipped_level_jumps": self.files_with_skipped_jumps,
            "files_with_skipped_level_jump_share": (
                self.files_with_skipped_jumps / headed if headed else 0.0
            ),
            "total_skipped_level_jumps": self.total_skipped_jumps,
            "files_starting_below_h1": self.files_starting_below_h1,
            "files_starting_below_h1_share": (
                self.files_starting_below_h1 / headed if headed else 0.0
            ),
            "files_without_h1": self.files_without_h1,
            "headed_files_without_h1": self.headed_files_without_h1,
            "headed_files_without_h1_share": (
                self.headed_files_without_h1 / headed if headed else 0.0
            ),
            "files_with_multiple_h1": self.files_with_multiple_h1,
            "files_with_multiple_h1_share": (
                self.files_with_multiple_h1 / headed if headed else 0.0
            ),
            "files_with_empty_sections": self.files_with_empty_sections,
            "files_with_empty_section_share": (
                self.files_with_empty_sections / headed if headed else 0.0
            ),
            "empty_sections": self.total_empty_sections,
            "sections_per_file": _describe_weighted(self.section_counts),
            "sections_per_headed_file": _describe_weighted(self.headed_section_counts),
            "top_level_sections_per_file": _describe_weighted(self.top_level_counts),
            "top_level_sections_per_headed_file": _describe_weighted(
                self.headed_top_level_counts
            ),
            "maximum_relative_depth_per_file": _describe_weighted(self.depth_counts),
            "maximum_relative_depth_per_headed_file": _describe_weighted(
                self.headed_depth_counts
            ),
            "skipped_level_jumps_per_file": _describe_weighted(self.jump_counts),
            "section_lines": _describe_weighted(self.section_line_counts),
            "section_words": _describe_weighted(self.section_word_counts),
            "section_content": {
                "files_with_list_bearing_sections": self.files_with_list_bearing_sections,
                "files_with_list_bearing_section_share": (
                    self.files_with_list_bearing_sections / headed if headed else 0.0
                ),
                "files_with_standalone_prose_sections": (
                    self.files_with_standalone_prose_sections
                ),
                "files_with_standalone_prose_section_share": (
                    self.files_with_standalone_prose_sections / headed
                    if headed
                    else 0.0
                ),
                "sections_with_list_items": sections_with_lists,
                "sections_with_list_item_share": (
                    sections_with_lists / self.total_sections
                    if self.total_sections
                    else 0.0
                ),
                "sections_with_standalone_prose": sections_with_prose,
                "sections_with_standalone_prose_share": (
                    sections_with_prose / self.total_sections
                    if self.total_sections
                    else 0.0
                ),
                "sections_with_other_content": sections_with_other,
                "sections_with_other_content_share": (
                    sections_with_other / self.total_sections
                    if self.total_sections
                    else 0.0
                ),
                "styles": style_summary,
                "list_items": {
                    "total": self.total_list_items,
                    "unordered": self.total_unordered_list_items,
                    "ordered": self.total_ordered_list_items,
                    "task": self.total_task_list_items,
                },
                "list_items_per_section": _describe_weighted(self.list_item_counts),
                "list_items_per_list_bearing_section": _describe_weighted(
                    self.list_item_counts_with_items
                ),
                "list_item_count_distribution": {
                    str(value): count
                    for value, count in sorted(self.list_item_counts.items())
                },
                "standalone_prose_blocks": self.total_standalone_prose_blocks,
                "standalone_prose_blocks_per_section": _describe_weighted(
                    self.standalone_prose_block_counts
                ),
                "content_lines": {
                    "classified_nonblank": classified_lines,
                    "list": self.total_list_content_lines,
                    "list_share": (
                        self.total_list_content_lines / classified_lines
                        if classified_lines
                        else 0.0
                    ),
                    "prose": self.total_prose_content_lines,
                    "prose_share": (
                        self.total_prose_content_lines / classified_lines
                        if classified_lines
                        else 0.0
                    ),
                    "other": self.total_other_content_lines,
                    "other_share": (
                        self.total_other_content_lines / classified_lines
                        if classified_lines
                        else 0.0
                    ),
                },
            },
        }


def analyze_markdown_sections(
    db_path: str | Path,
    *,
    scope: str = "exact-claude",
) -> tuple[
    dict[str, object],
    list[ContentFamilyStructure],
    list[dict[str, int | float | str]],
]:
    """Analyze section structure once per content hash and weight all copies."""
    predicate = _scope_predicate(scope, alias="f")
    connection = _readonly_connection(db_path)
    try:
        counts = connection.execute(
            f"""
            SELECT count(*) AS scoped_files,
                   count(content_hash) AS hashed_files,
                   count(DISTINCT content_hash) AS content_families
            FROM files AS f
            WHERE {predicate}
            """
        ).fetchone()
        assert counts is not None
        rows = connection.execute(
            f"""
            WITH ranked AS (
                SELECT f.content_hash, f.repo_full_name, f.path, f.size_bytes,
                       f.content,
                       count(*) OVER (PARTITION BY f.content_hash) AS copies,
                       row_number() OVER (
                           PARTITION BY f.content_hash
                           ORDER BY lower(f.repo_full_name), lower(f.path),
                                    f.repo_full_name, f.path, f.id
                       ) AS family_row
                FROM files AS f
                WHERE {predicate}
                  AND f.content_hash IS NOT NULL
            )
            SELECT content_hash, repo_full_name, path, size_bytes, content, copies
            FROM ranked
            WHERE family_row = 1
            ORDER BY content_hash
            """
        )

        full = PopulationAccumulator()
        unique = PopulationAccumulator()
        family_rows: list[ContentFamilyStructure] = []
        heading_occurrences_full: Counter[str] = Counter()
        heading_files_full: Counter[str] = Counter()
        heading_occurrences_unique: Counter[str] = Counter()
        heading_files_unique: Counter[str] = Counter()

        for row in rows:
            copies = int(row["copies"])
            structure = parse_document_structure(str(row["content"]))
            full.add(structure, copies)
            unique.add(structure, 1)

            per_content_names = Counter(
                heading.normalized_name for heading in structure.headings
            )
            for name, occurrences in per_content_names.items():
                heading_occurrences_full[name] += occurrences * copies
                heading_files_full[name] += copies
                heading_occurrences_unique[name] += occurrences
                heading_files_unique[name] += 1

            median_lines, mean_lines, maximum_lines = _family_value_summary(
                structure.section_line_counts
            )
            median_words, mean_words, maximum_words = _family_value_summary(
                structure.section_word_counts
            )
            list_item_values = tuple(
                body.list_item_count for body in structure.section_bodies
            )
            median_list_items, mean_list_items, maximum_list_items = (
                _family_value_summary(list_item_values)
            )
            family_styles = Counter(
                body.content_style for body in structure.section_bodies
            )
            family_rows.append(
                ContentFamilyStructure(
                    content_hash=str(row["content_hash"]),
                    copies=copies,
                    representative_repo=str(row["repo_full_name"]),
                    representative_path=str(row["path"]),
                    size_bytes=(
                        int(row["size_bytes"])
                        if row["size_bytes"] is not None
                        else None
                    ),
                    section_count=len(structure.headings),
                    top_level_sections=structure.top_level_sections,
                    maximum_relative_depth=structure.maximum_relative_depth,
                    skipped_level_jumps=structure.skipped_level_jumps,
                    starts_below_h1=structure.starts_below_h1,
                    h1_count=structure.h1_count,
                    empty_sections=structure.empty_sections,
                    sections_with_list_items=sum(
                        bool(body.list_item_count) for body in structure.section_bodies
                    ),
                    sections_with_standalone_prose=sum(
                        bool(body.standalone_prose_block_count)
                        for body in structure.section_bodies
                    ),
                    total_list_items=sum(
                        body.list_item_count for body in structure.section_bodies
                    ),
                    unordered_list_items=sum(
                        body.unordered_list_item_count
                        for body in structure.section_bodies
                    ),
                    ordered_list_items=sum(
                        body.ordered_list_item_count for body in structure.section_bodies
                    ),
                    task_list_items=sum(
                        body.task_list_item_count for body in structure.section_bodies
                    ),
                    total_standalone_prose_blocks=sum(
                        body.standalone_prose_block_count
                        for body in structure.section_bodies
                    ),
                    list_content_lines=sum(
                        body.list_content_line_count for body in structure.section_bodies
                    ),
                    prose_content_lines=sum(
                        body.prose_content_line_count for body in structure.section_bodies
                    ),
                    other_content_lines=sum(
                        body.other_content_line_count for body in structure.section_bodies
                    ),
                    list_only_sections=family_styles["list_only"],
                    prose_only_sections=family_styles["prose_only"],
                    list_and_prose_sections=family_styles["list_and_prose"],
                    other_only_sections=family_styles["other_only"],
                    sections_with_other_content=sum(
                        count
                        for style, count in family_styles.items()
                        if "other" in style
                    ),
                    median_list_items_per_section=median_list_items,
                    mean_list_items_per_section=mean_list_items,
                    maximum_list_items_per_section=maximum_list_items,
                    median_section_lines=median_lines,
                    mean_section_lines=mean_lines,
                    maximum_section_lines=maximum_lines,
                    median_section_words=median_words,
                    mean_section_words=mean_words,
                    maximum_section_words=maximum_words,
                )
            )
    finally:
        connection.close()

    family_rows.sort(key=lambda item: item.content_hash)
    heading_names = [
        {
            "normalized_heading": name,
            "files_full": heading_files_full[name],
            "occurrences_full": heading_occurrences_full[name],
            "unique_contents": heading_files_unique[name],
            "occurrences_unique": heading_occurrences_unique[name],
        }
        for name in heading_occurrences_full
    ]
    heading_names.sort(
        key=lambda item: (
            -int(item["files_full"]),
            -int(item["occurrences_full"]),
            str(item["normalized_heading"]),
        )
    )

    summary: dict[str, object] = {
        "schema_version": 2,
        "scope": scope,
        "scope_definition": _scope_definition(scope),
        "scoped_files": int(counts["scoped_files"]),
        "hashed_files": int(counts["hashed_files"]),
        "unhashed_files": int(counts["scoped_files"]) - int(counts["hashed_files"]),
        "content_families": int(counts["content_families"]),
        "distinct_normalized_heading_names": len(heading_names),
        "parser": {
            "headings": "ATX and Setext headings outside backtick/tilde fences",
            "section_length": "non-overlapping lines/words after a heading and before the next heading",
            "list_items": (
                "unordered (-, +, *), ordered (1. or 1)), and task-list markers "
                "outside fenced/indented code; nested items included"
            ),
            "list_content": (
                "item-marker lines plus indented or lazy continuation lines and "
                "fenced blocks contained by an active list"
            ),
            "standalone_prose": (
                "contiguous paragraph-like lines outside list content, fenced/"
                "indented code, tables, link definitions, thematic breaks, and "
                "recognized HTML block lines"
            ),
            "content_styles": list(CONTENT_STYLES),
            "blockquote_handling": "classify the content after blockquote markers",
            "top_level": "headings with no preceding open heading of a lower numeric level",
            "relative_depth": "maximum depth in the heading stack",
            "hierarchy_violation": "adjacent downward transition skipping one or more heading levels",
            "quantiles": "empirical nearest-rank",
        },
        "full_population": full.as_dict(),
        "unique_content_population": unique.as_dict(),
    }
    return summary, family_rows, heading_names


def write_family_csv(
    families: Sequence[ContentFamilyStructure],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(ContentFamilyStructure.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in families:
            writer.writerow({field: getattr(family, field) for field in fields})
    return output


def write_heading_csv(
    headings: Sequence[dict[str, int | float | str]],
    output_path: str | Path,
    *,
    limit: int,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "rank",
        "normalized_heading",
        "files_full",
        "occurrences_full",
        "unique_contents",
        "occurrences_unique",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(headings[:limit], 1):
            writer.writerow({"rank": rank, **item})
    return output


def _population_section_content(
    summary: dict[str, object], population_key: str
) -> tuple[dict[str, object], dict[str, object]]:
    population = summary[population_key]
    assert isinstance(population, dict)
    section_content = population["section_content"]
    assert isinstance(section_content, dict)
    return population, section_content


def write_content_style_csv(
    summary: dict[str, object], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    full, full_content = _population_section_content(summary, "full_population")
    unique, unique_content = _population_section_content(
        summary, "unique_content_population"
    )
    full_styles = full_content["styles"]
    unique_styles = unique_content["styles"]
    assert isinstance(full_styles, dict) and isinstance(unique_styles, dict)
    fields = (
        "content_style",
        "sections_full",
        "share_full",
        "sections_unique",
        "share_unique",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for style in CONTENT_STYLES:
            full_item = full_styles[style]
            unique_item = unique_styles[style]
            assert isinstance(full_item, dict) and isinstance(unique_item, dict)
            writer.writerow(
                {
                    "content_style": style,
                    "sections_full": full_item["sections"],
                    "share_full": full_item["share"],
                    "sections_unique": unique_item["sections"],
                    "share_unique": unique_item["share"],
                }
            )
    assert int(full["explicit_sections"]) == sum(
        int(full_styles[style]["sections"]) for style in CONTENT_STYLES
    )
    assert int(unique["explicit_sections"]) == sum(
        int(unique_styles[style]["sections"]) for style in CONTENT_STYLES
    )
    return output


def write_list_item_distribution_csv(
    summary: dict[str, object], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    full, full_content = _population_section_content(summary, "full_population")
    unique, unique_content = _population_section_content(
        summary, "unique_content_population"
    )
    full_distribution = full_content["list_item_count_distribution"]
    unique_distribution = unique_content["list_item_count_distribution"]
    assert isinstance(full_distribution, dict) and isinstance(unique_distribution, dict)
    values = sorted({int(value) for value in full_distribution | unique_distribution})
    full_sections = int(full["explicit_sections"])
    unique_sections = int(unique["explicit_sections"])
    full_list_sections = int(full_content["sections_with_list_items"])
    unique_list_sections = int(unique_content["sections_with_list_items"])
    fields = (
        "list_item_count",
        "sections_full",
        "share_all_sections_full",
        "share_list_bearing_sections_full",
        "sections_unique",
        "share_all_sections_unique",
        "share_list_bearing_sections_unique",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in values:
            full_count = int(full_distribution.get(str(value), 0))
            unique_count = int(unique_distribution.get(str(value), 0))
            writer.writerow(
                {
                    "list_item_count": value,
                    "sections_full": full_count,
                    "share_all_sections_full": (
                        full_count / full_sections if full_sections else 0.0
                    ),
                    "share_list_bearing_sections_full": (
                        full_count / full_list_sections
                        if value and full_list_sections
                        else 0.0
                    ),
                    "sections_unique": unique_count,
                    "share_all_sections_unique": (
                        unique_count / unique_sections if unique_sections else 0.0
                    ),
                    "share_list_bearing_sections_unique": (
                        unique_count / unique_list_sections
                        if value and unique_list_sections
                        else 0.0
                    ),
                }
            )
    return output


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read mined.db without modifying it and export Markdown section "
            "hierarchy statistics."
        )
    )
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="exact-claude",
        help="population to analyze (default: exact-claude)",
    )
    parser.add_argument(
        "--families-output",
        default="section_structure_families.csv",
        help="one-row-per-content-family CSV output",
    )
    parser.add_argument(
        "--headings-output",
        default="common_heading_names.csv",
        help="ranked normalized-heading CSV output",
    )
    parser.add_argument(
        "--styles-output",
        default="section_content_styles.csv",
        help="exclusive per-section content-style CSV output",
    )
    parser.add_argument(
        "--list-distribution-output",
        default="section_list_item_distribution.csv",
        help="exact per-section list-item-count distribution CSV output",
    )
    parser.add_argument(
        "--summary-output",
        default="section_structure_summary.json",
        help="machine-readable JSON summary output",
    )
    parser.add_argument(
        "--heading-limit",
        type=positive_int,
        default=1000,
        help="maximum heading-name rows to export (default: 1000)",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=20,
        help="number of common headings to print (default: 20)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, families, heading_names = analyze_markdown_sections(
            args.db,
            scope=args.scope,
        )
        family_path = write_family_csv(families, args.families_output)
        heading_path = write_heading_csv(
            heading_names,
            args.headings_output,
            limit=args.heading_limit,
        )
        styles_path = write_content_style_csv(summary, args.styles_output)
        list_distribution_path = write_list_item_distribution_csv(
            summary, args.list_distribution_output
        )
        summary_path = write_summary_json(summary, args.summary_output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    full = summary["full_population"]
    unique = summary["unique_content_population"]
    assert isinstance(full, dict) and isinstance(unique, dict)
    print(f"Scope: {summary['scope']} ({summary['scope_definition']})")
    print(f"Files: {full['files']:,}; unique contents: {unique['files']:,}")
    print(
        f"Files with headings: {full['files_with_headings']:,} "
        f"({full['files_with_heading_share']:.2%})"
    )
    print(f"Explicit sections: {full['explicit_sections']:,}")
    full_content = full["section_content"]
    assert isinstance(full_content, dict)
    print(
        f"Sections with list items: {full_content['sections_with_list_items']:,} "
        f"({full_content['sections_with_list_item_share']:.2%})"
    )
    print(
        "Files with skipped heading levels: "
        f"{full['files_with_skipped_level_jumps']:,} "
        f"({full['files_with_skipped_level_jump_share']:.2%} of headed files)"
    )
    print("Common normalized headings:")
    for rank, item in enumerate(heading_names[: args.top], 1):
        print(
            f"  {rank:>3}. {str(item['normalized_heading']):30s} "
            f"{int(item['files_full']):>7,} files; "
            f"{int(item['occurrences_full']):>8,} occurrences"
        )
    print(f"Family CSV: {family_path.resolve()}")
    print(f"Heading CSV: {heading_path.resolve()}")
    print(f"Content-style CSV: {styles_path.resolve()}")
    print(f"List-item distribution CSV: {list_distribution_path.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
