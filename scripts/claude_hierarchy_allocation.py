"""Analyze how content is allocated across observed ``CLAUDE.md`` hierarchies.

The population is all repositories with more than one exact case-insensitive
``CLAUDE.md`` basename.  The analysis measures repository-level root/nested
content allocation and compares every nearest-ancestor parent/child path pair.
It also tests whether local Markdown references lexically resolve to an
observed parent or child path.

Example:

    python scripts/claude_hierarchy_allocation.py \
        --db mined.db \
        --repositories-output article/claude_hierarchy_allocation_repositories.csv \
        --edges-output article/claude_hierarchy_allocation_edges.csv \
        --headings-output article/claude_hierarchy_heading_comparison.csv \
        --summary-output article/claude_hierarchy_allocation_summary.json

SQLite is opened with ``mode=ro&immutable=1`` and ``query_only=ON``.  The
database is hashed before and after the analysis.  Only local output artifacts
are written.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.claude_markdown_references import (  # noqa: E402
    POINTER_MAX_RESIDUAL_WORDS,
    _parse_document,
    classify_local_intent,
    local_path_relation,
    resolve_local_target,
    target_category,
)
from scripts.claude_repository_structure import (  # noqa: E402
    FileRecord,
    HierarchyEdge,
    _build_hierarchy,
    _normalize_lines,
)
from scripts.markdown_section_stats import (  # noqa: E402
    _fence_close,
    _fence_open,
    parse_document_structure,
)


ANALYSIS_VERSION = "claude-hierarchy-allocation-v1"
EXACT_CLAUDE_SCOPE = "case-insensitive basename equal to CLAUDE.md"
BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


@dataclass(frozen=True)
class ContentMetrics:
    line_count: int
    word_count: int
    section_count: int
    top_level_sections: int
    maximum_section_depth: int
    code_block_count: int
    bullet_count: int
    link_count: int
    heading_names: frozenset[str]
    normalized_lines: frozenset[str]


@dataclass(frozen=True)
class FileMetrics:
    repo_full_name: str
    path: str
    content_hash: str
    size_bytes: int
    directory_depth: int
    line_count: int
    word_count: int
    section_count: int
    top_level_sections: int
    maximum_section_depth: int
    code_block_count: int
    bullet_count: int
    link_count: int
    heading_names: frozenset[str]
    normalized_lines: frozenset[str]
    local_nonself_reference_occurrences: int
    high_confidence_local_instructional_occurrences: int
    instructional_fan_out: int
    document_form: str
    literal_local_targets: frozenset[str]
    high_confidence_local_targets: frozenset[str]


@dataclass(frozen=True)
class RepositoryAllocationRow:
    repo_full_name: str
    exact_files: int
    root_files: int
    nested_files: int
    location_configuration: str
    hierarchy_edges: int
    total_size_bytes: int
    root_size_bytes: int
    nested_size_bytes: int
    root_byte_share: float
    nested_byte_share: float
    largest_file_size_bytes: int
    largest_file_byte_share: float
    size_hhi: float
    total_words: int
    total_sections: int
    total_code_blocks: int
    total_bullets: int
    total_links: int
    files_with_high_confidence_local_instructional_candidate: int
    high_confidence_local_instructional_occurrences: int
    pointer_only_files: int
    within_repository_exact_duplicate_groups: int
    files_in_within_repository_exact_duplicate_groups: int
    parent_to_child_literal_reference_edges: int
    child_to_parent_literal_reference_edges: int
    either_direction_literal_reference_edges: int
    bidirectional_literal_reference_edges: int
    parent_to_child_high_confidence_edges: int
    child_to_parent_high_confidence_edges: int
    either_direction_high_confidence_edges: int
    bidirectional_high_confidence_edges: int


@dataclass(frozen=True)
class HierarchyAllocationEdgeRow:
    repo_full_name: str
    parent_path: str
    child_path: str
    parent_directory_depth: int
    child_directory_depth: int
    directory_distance: int
    parent_candidates_at_nearest_scope: int
    parent_size_bytes: int
    child_size_bytes: int
    child_minus_parent_size_bytes: int
    child_parent_size_ratio: float | None
    parent_word_count: int
    child_word_count: int
    child_minus_parent_words: int
    child_parent_word_ratio: float | None
    parent_section_count: int
    child_section_count: int
    child_minus_parent_sections: int
    parent_code_block_count: int
    child_code_block_count: int
    child_minus_parent_code_blocks: int
    parent_bullet_count: int
    child_bullet_count: int
    child_minus_parent_bullets: int
    parent_link_count: int
    child_link_count: int
    child_minus_parent_links: int
    parent_high_confidence_local_instructional_occurrences: int
    child_high_confidence_local_instructional_occurrences: int
    parent_instructional_fan_out: int
    child_instructional_fan_out: int
    parent_document_form: str
    child_document_form: str
    same_content_hash: int
    same_normalized_line_set: int
    parent_normalized_lines: int
    child_normalized_lines: int
    shared_normalized_lines: int
    child_unique_normalized_lines: int
    jaccard_similarity: float
    parent_line_coverage: float
    child_line_coverage: float
    smaller_document_containment: float
    parent_references_child_literal: int
    child_references_parent_literal: int
    parent_references_child_high_confidence: int
    child_references_parent_high_confidence: int


@dataclass(frozen=True)
class HeadingComparisonRow:
    rank: int
    normalized_heading: str
    all_multi_file_documents: int
    all_multi_file_document_share: float
    root_documents: int
    root_document_share: float
    nested_documents: int
    nested_document_share: float
    parent_role_documents: int
    parent_role_document_share: float
    child_role_documents: int
    child_role_document_share: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _share(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _exact_predicate(*, alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return (
        f"(lower({prefix}path) = 'claude.md' "
        f"OR lower({prefix}path) LIKE '%/claude.md')"
    )


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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty population")
    rank = max(1, math.ceil(probability * len(ordered)))
    return float(ordered[rank - 1])


def _numeric_distribution(values: Iterable[int | float]) -> dict[str, int | float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p25": _nearest_rank(ordered, 0.25),
        "median": _nearest_rank(ordered, 0.50),
        "p75": _nearest_rank(ordered, 0.75),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _count_fenced_blocks(content: str) -> int:
    fence_character: str | None = None
    fence_length = 0
    blocks = 0
    for line in content.splitlines():
        if fence_character is not None:
            if _fence_close(line, fence_character, fence_length):
                fence_character = None
                fence_length = 0
            continue
        opening = _fence_open(line)
        if opening is not None:
            fence_character, fence_length = opening
            blocks += 1
    return blocks


def _measure_content(content: str) -> ContentMetrics:
    structure = parse_document_structure(content)
    return ContentMetrics(
        line_count=len(content.splitlines()),
        word_count=len(content.split()),
        section_count=len(structure.headings),
        top_level_sections=structure.top_level_sections,
        maximum_section_depth=structure.maximum_relative_depth,
        code_block_count=_count_fenced_blocks(content),
        bullet_count=len(BULLET_RE.findall(content)),
        link_count=len(LINK_RE.findall(content)),
        heading_names=frozenset(
            heading.normalized_name for heading in structure.headings
        ),
        normalized_lines=_normalize_lines(content),
    )


def _measure_file_references(
    *,
    content: str,
    source_path: str,
    parsed_document: object,
) -> tuple[int, int, int, str, frozenset[str], frozenset[str]]:
    parsed = parsed_document
    literal_targets: set[str] = set()
    high_targets: set[str] = set()
    high_occurrences = 0
    local_nonself_occurrences = 0

    for reference in parsed.local_references:  # type: ignore[attr-defined]
        resolved = resolve_local_target(source_path, reference.normalized_target)
        relation = local_path_relation(source_path, resolved)
        self_reference = resolved.casefold() == source_path.casefold()
        if not self_reference and relation != "outside_repository":
            local_nonself_occurrences += 1
            literal_targets.add(resolved.casefold())
        context = parsed.contexts.get(  # type: ignore[attr-defined]
            reference.line_number,
            reference.source_line,
        )
        category = target_category(reference.target_basename, resolved)
        _intent, _confidence, _rule, high = classify_local_intent(
            reference,
            category,
            context,
        )
        if self_reference or relation == "outside_repository":
            high = 0
        if high:
            high_occurrences += 1
            high_targets.add(resolved.casefold())

    if not parsed.local_references and not parsed.external_references:  # type: ignore[attr-defined]
        document_form = "no_detected_markdown_reference"
    elif (
        high_occurrences
        and not parsed.has_fenced_block  # type: ignore[attr-defined]
        and parsed.residual_word_count <= POINTER_MAX_RESIDUAL_WORDS  # type: ignore[attr-defined]
    ):
        document_form = "pointer_only"
    elif high_occurrences:
        document_form = "instructions_plus_delegation"
    elif local_nonself_occurrences:
        document_form = "contextual_local_references_only"
    elif parsed.external_references:  # type: ignore[attr-defined]
        document_form = "external_references_only"
    else:
        document_form = "self_references_only"

    return (
        local_nonself_occurrences,
        high_occurrences,
        len(high_targets),
        document_form,
        frozenset(literal_targets),
        frozenset(high_targets),
    )


def _file_metrics(
    row: sqlite3.Row,
    *,
    content_cache: dict[str, tuple[ContentMetrics, object]],
) -> FileMetrics:
    content = str(row["content"] or "")
    content_hash = str(row["content_hash"] or "")
    cache_key = content_hash or f"id:{int(row['id'])}"
    cached = content_cache.get(cache_key)
    if cached is None:
        cached = (_measure_content(content), _parse_document(content))
        content_cache[cache_key] = cached
    content_metrics, parsed_document = cached
    source_path = str(row["path"])
    (
        local_occurrences,
        high_occurrences,
        fan_out,
        document_form,
        literal_targets,
        high_targets,
    ) = _measure_file_references(
        content=content,
        source_path=source_path,
        parsed_document=parsed_document,
    )
    size_bytes = (
        int(row["size_bytes"])
        if row["size_bytes"] is not None
        else len(content.encode("utf-8"))
    )
    return FileMetrics(
        repo_full_name=str(row["repo_full_name"]),
        path=source_path,
        content_hash=content_hash,
        size_bytes=size_bytes,
        directory_depth=max(0, len(source_path.split("/")) - 1),
        line_count=content_metrics.line_count,
        word_count=content_metrics.word_count,
        section_count=content_metrics.section_count,
        top_level_sections=content_metrics.top_level_sections,
        maximum_section_depth=content_metrics.maximum_section_depth,
        code_block_count=content_metrics.code_block_count,
        bullet_count=content_metrics.bullet_count,
        link_count=content_metrics.link_count,
        heading_names=content_metrics.heading_names,
        normalized_lines=content_metrics.normalized_lines,
        local_nonself_reference_occurrences=local_occurrences,
        high_confidence_local_instructional_occurrences=high_occurrences,
        instructional_fan_out=fan_out,
        document_form=document_form,
        literal_local_targets=literal_targets,
        high_confidence_local_targets=high_targets,
    )


def _path_sort_key(path: str) -> tuple[str, str]:
    return path.casefold(), path


def _location_configuration(root_files: int, nested_files: int) -> str:
    if root_files and nested_files:
        return "root_and_nested"
    if root_files:
        return "root_only"
    return "nested_only"


def _overlap_values(
    parent: frozenset[str],
    child: frozenset[str],
) -> tuple[int, int, float, float, float, float]:
    shared = len(parent & child)
    union = len(parent | child)
    smaller = min(len(parent), len(child))
    return (
        shared,
        len(child - parent),
        _share(shared, union),
        _share(shared, len(parent)),
        _share(shared, len(child)),
        _share(shared, smaller),
    )


def _edge_row(
    edge: HierarchyEdge,
    files_by_path: dict[str, FileMetrics],
) -> HierarchyAllocationEdgeRow:
    parent = files_by_path[edge.parent_path]
    child = files_by_path[edge.child_path]
    (
        shared,
        child_unique,
        jaccard,
        parent_coverage,
        child_coverage,
        containment,
    ) = _overlap_values(parent.normalized_lines, child.normalized_lines)
    child_key = child.path.casefold()
    parent_key = parent.path.casefold()
    return HierarchyAllocationEdgeRow(
        repo_full_name=edge.repo_full_name,
        parent_path=edge.parent_path,
        child_path=edge.child_path,
        parent_directory_depth=edge.parent_directory_depth,
        child_directory_depth=edge.child_directory_depth,
        directory_distance=edge.directory_distance,
        parent_candidates_at_nearest_scope=edge.parent_candidates_at_nearest_scope,
        parent_size_bytes=parent.size_bytes,
        child_size_bytes=child.size_bytes,
        child_minus_parent_size_bytes=child.size_bytes - parent.size_bytes,
        child_parent_size_ratio=_ratio(child.size_bytes, parent.size_bytes),
        parent_word_count=parent.word_count,
        child_word_count=child.word_count,
        child_minus_parent_words=child.word_count - parent.word_count,
        child_parent_word_ratio=_ratio(child.word_count, parent.word_count),
        parent_section_count=parent.section_count,
        child_section_count=child.section_count,
        child_minus_parent_sections=child.section_count - parent.section_count,
        parent_code_block_count=parent.code_block_count,
        child_code_block_count=child.code_block_count,
        child_minus_parent_code_blocks=(
            child.code_block_count - parent.code_block_count
        ),
        parent_bullet_count=parent.bullet_count,
        child_bullet_count=child.bullet_count,
        child_minus_parent_bullets=child.bullet_count - parent.bullet_count,
        parent_link_count=parent.link_count,
        child_link_count=child.link_count,
        child_minus_parent_links=child.link_count - parent.link_count,
        parent_high_confidence_local_instructional_occurrences=(
            parent.high_confidence_local_instructional_occurrences
        ),
        child_high_confidence_local_instructional_occurrences=(
            child.high_confidence_local_instructional_occurrences
        ),
        parent_instructional_fan_out=parent.instructional_fan_out,
        child_instructional_fan_out=child.instructional_fan_out,
        parent_document_form=parent.document_form,
        child_document_form=child.document_form,
        same_content_hash=int(
            bool(parent.content_hash)
            and parent.content_hash == child.content_hash
        ),
        same_normalized_line_set=int(
            bool(parent.normalized_lines)
            and parent.normalized_lines == child.normalized_lines
        ),
        parent_normalized_lines=len(parent.normalized_lines),
        child_normalized_lines=len(child.normalized_lines),
        shared_normalized_lines=shared,
        child_unique_normalized_lines=child_unique,
        jaccard_similarity=jaccard,
        parent_line_coverage=parent_coverage,
        child_line_coverage=child_coverage,
        smaller_document_containment=containment,
        parent_references_child_literal=int(
            child_key in parent.literal_local_targets
        ),
        child_references_parent_literal=int(
            parent_key in child.literal_local_targets
        ),
        parent_references_child_high_confidence=int(
            child_key in parent.high_confidence_local_targets
        ),
        child_references_parent_high_confidence=int(
            parent_key in child.high_confidence_local_targets
        ),
    )


def _repository_row(
    repo_full_name: str,
    files: Sequence[FileMetrics],
    edges: Sequence[HierarchyAllocationEdgeRow],
) -> RepositoryAllocationRow:
    root_files = [file for file in files if file.directory_depth == 0]
    nested_files = [file for file in files if file.directory_depth > 0]
    total_size = sum(file.size_bytes for file in files)
    root_size = sum(file.size_bytes for file in root_files)
    nested_size = total_size - root_size
    largest_size = max((file.size_bytes for file in files), default=0)
    hash_counts = Counter(
        file.content_hash for file in files if file.content_hash
    )
    duplicate_counts = [count for count in hash_counts.values() if count > 1]
    literal_parent_child = sum(
        edge.parent_references_child_literal for edge in edges
    )
    literal_child_parent = sum(
        edge.child_references_parent_literal for edge in edges
    )
    high_parent_child = sum(
        edge.parent_references_child_high_confidence for edge in edges
    )
    high_child_parent = sum(
        edge.child_references_parent_high_confidence for edge in edges
    )
    return RepositoryAllocationRow(
        repo_full_name=repo_full_name,
        exact_files=len(files),
        root_files=len(root_files),
        nested_files=len(nested_files),
        location_configuration=_location_configuration(
            len(root_files),
            len(nested_files),
        ),
        hierarchy_edges=len(edges),
        total_size_bytes=total_size,
        root_size_bytes=root_size,
        nested_size_bytes=nested_size,
        root_byte_share=_share(root_size, total_size),
        nested_byte_share=_share(nested_size, total_size),
        largest_file_size_bytes=largest_size,
        largest_file_byte_share=_share(largest_size, total_size),
        size_hhi=sum(
            _share(file.size_bytes, total_size) ** 2 for file in files
        ),
        total_words=sum(file.word_count for file in files),
        total_sections=sum(file.section_count for file in files),
        total_code_blocks=sum(file.code_block_count for file in files),
        total_bullets=sum(file.bullet_count for file in files),
        total_links=sum(file.link_count for file in files),
        files_with_high_confidence_local_instructional_candidate=sum(
            file.high_confidence_local_instructional_occurrences > 0
            for file in files
        ),
        high_confidence_local_instructional_occurrences=sum(
            file.high_confidence_local_instructional_occurrences
            for file in files
        ),
        pointer_only_files=sum(
            file.document_form == "pointer_only" for file in files
        ),
        within_repository_exact_duplicate_groups=len(duplicate_counts),
        files_in_within_repository_exact_duplicate_groups=sum(duplicate_counts),
        parent_to_child_literal_reference_edges=literal_parent_child,
        child_to_parent_literal_reference_edges=literal_child_parent,
        either_direction_literal_reference_edges=sum(
            bool(
                edge.parent_references_child_literal
                or edge.child_references_parent_literal
            )
            for edge in edges
        ),
        bidirectional_literal_reference_edges=sum(
            bool(
                edge.parent_references_child_literal
                and edge.child_references_parent_literal
            )
            for edge in edges
        ),
        parent_to_child_high_confidence_edges=high_parent_child,
        child_to_parent_high_confidence_edges=high_child_parent,
        either_direction_high_confidence_edges=sum(
            bool(
                edge.parent_references_child_high_confidence
                or edge.child_references_parent_high_confidence
            )
            for edge in edges
        ),
        bidirectional_high_confidence_edges=sum(
            bool(
                edge.parent_references_child_high_confidence
                and edge.child_references_parent_high_confidence
            )
            for edge in edges
        ),
    )


def _comparison_counts(
    rows: Sequence[HierarchyAllocationEdgeRow],
    parent_field: str,
    child_field: str,
) -> dict[str, int | float]:
    less = equal = greater = 0
    for row in rows:
        parent = int(getattr(row, parent_field))
        child = int(getattr(row, child_field))
        if child < parent:
            less += 1
        elif child > parent:
            greater += 1
        else:
            equal += 1
    total = len(rows)
    return {
        "child_lower": less,
        "child_equal": equal,
        "child_higher": greater,
        "child_lower_share": _share(less, total),
        "child_equal_share": _share(equal, total),
        "child_higher_share": _share(greater, total),
    }


def _odds_ratio(a: int, b: int, c: int, d: int) -> float | None:
    denominator = b * c
    return (a * d) / denominator if denominator else None


def _odds_ratio_interval(
    a: int,
    b: int,
    c: int,
    d: int,
) -> dict[str, float | None]:
    ratio = _odds_ratio(a, b, c, d)
    if ratio is None or min(a, b, c, d) <= 0:
        return {"estimate": ratio, "lower_95": None, "upper_95": None}
    standard_error = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_ratio = math.log(ratio)
    return {
        "estimate": ratio,
        "lower_95": math.exp(log_ratio - 1.96 * standard_error),
        "upper_95": math.exp(log_ratio + 1.96 * standard_error),
    }


def _multiplicity_band(file_count: int) -> str:
    if file_count == 2:
        return "2"
    if file_count == 3:
        return "3"
    if file_count <= 5:
        return "4-5"
    if file_count <= 10:
        return "6-10"
    if file_count <= 20:
        return "11-20"
    return ">20"


def _allocation_strata(
    repositories: Sequence[RepositoryAllocationRow],
    *,
    dimension: str,
) -> list[dict[str, object]]:
    if dimension == "observed_file_count":
        order = ("2", "3", "4-5", "6-10", "11-20", ">20")
        value_for = lambda row: _multiplicity_band(row.exact_files)
    elif dimension == "location_configuration":
        order = ("root_only", "nested_only", "root_and_nested")
        value_for = lambda row: row.location_configuration
    else:
        raise ValueError(f"unsupported allocation stratum: {dimension}")

    rows: list[dict[str, object]] = []
    for value in order:
        selected = [row for row in repositories if value_for(row) == value]
        if not selected:
            continue
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "repositories": len(selected),
                "repository_share": _share(len(selected), len(repositories)),
                "files": sum(row.exact_files for row in selected),
                "total_size_bytes": sum(row.total_size_bytes for row in selected),
                "largest_file_byte_share": _numeric_distribution(
                    row.largest_file_byte_share for row in selected
                ),
                "size_hhi": _numeric_distribution(row.size_hhi for row in selected),
                "repositories_with_hierarchy": sum(
                    row.hierarchy_edges > 0 for row in selected
                ),
                "hierarchy_repository_share": _share(
                    sum(row.hierarchy_edges > 0 for row in selected),
                    len(selected),
                ),
                "repositories_with_exact_reuse": sum(
                    row.within_repository_exact_duplicate_groups > 0
                    for row in selected
                ),
                "exact_reuse_repository_share": _share(
                    sum(
                        row.within_repository_exact_duplicate_groups > 0
                        for row in selected
                    ),
                    len(selected),
                ),
            }
        )
    return rows


def _heading_rows(
    heading_counts: dict[str, Counter[str]],
    denominators: dict[str, int],
    *,
    limit: int,
) -> list[HeadingComparisonRow]:
    headings = set().union(*(counter.keys() for counter in heading_counts.values()))
    ordered = sorted(
        headings,
        key=lambda heading: (-heading_counts["all"][heading], heading),
    )[:limit]
    return [
        HeadingComparisonRow(
            rank=rank,
            normalized_heading=heading,
            all_multi_file_documents=heading_counts["all"][heading],
            all_multi_file_document_share=_share(
                heading_counts["all"][heading], denominators["all"]
            ),
            root_documents=heading_counts["root"][heading],
            root_document_share=_share(
                heading_counts["root"][heading], denominators["root"]
            ),
            nested_documents=heading_counts["nested"][heading],
            nested_document_share=_share(
                heading_counts["nested"][heading], denominators["nested"]
            ),
            parent_role_documents=heading_counts["parent"][heading],
            parent_role_document_share=_share(
                heading_counts["parent"][heading], denominators["parent"]
            ),
            child_role_documents=heading_counts["child"][heading],
            child_role_document_share=_share(
                heading_counts["child"][heading], denominators["child"]
            ),
        )
        for rank, heading in enumerate(ordered, 1)
    ]


def _build_summary(
    repositories: Sequence[RepositoryAllocationRow],
    edges: Sequence[HierarchyAllocationEdgeRow],
    headings: Sequence[HeadingComparisonRow],
    *,
    files: Sequence[FileMetrics],
    database_sha256_before: str,
    database_sha256_after: str,
    database_changes: int,
    heading_limit: int,
) -> dict[str, object]:
    hierarchy_repositories = [row for row in repositories if row.hierarchy_edges]
    nonhierarchy_repositories = [row for row in repositories if not row.hierarchy_edges]
    root_and_nested = [
        row
        for row in repositories
        if row.location_configuration == "root_and_nested"
    ]
    hierarchy_with_candidate = sum(
        row.files_with_high_confidence_local_instructional_candidate > 0
        for row in hierarchy_repositories
    )
    nonhierarchy_with_candidate = sum(
        row.files_with_high_confidence_local_instructional_candidate > 0
        for row in nonhierarchy_repositories
    )
    hierarchy_without_candidate = (
        len(hierarchy_repositories) - hierarchy_with_candidate
    )
    nonhierarchy_without_candidate = (
        len(nonhierarchy_repositories) - nonhierarchy_with_candidate
    )
    literal_either = sum(
        bool(
            edge.parent_references_child_literal
            or edge.child_references_parent_literal
        )
        for edge in edges
    )
    high_either = sum(
        bool(
            edge.parent_references_child_high_confidence
            or edge.child_references_parent_high_confidence
        )
        for edge in edges
    )
    configuration_counts = Counter(
        row.location_configuration for row in repositories
    )
    different_hash_edges = [edge for edge in edges if not edge.same_content_hash]
    parent_role_documents = {
        (edge.repo_full_name, edge.parent_path) for edge in edges
    }
    child_role_documents = {
        (edge.repo_full_name, edge.child_path) for edge in edges
    }
    return {
        "analysis": "content allocation across observed exact CLAUDE.md hierarchies",
        "analysis_version": ANALYSIS_VERSION,
        "scope": "exact-claude-multi-file-repositories",
        "scope_definition": (
            f"{EXACT_CLAUDE_SCOPE}; repositories with at least two observed files"
        ),
        "parameters": {
            "heading_output_limit": heading_limit,
            "pointer_max_residual_words": POINTER_MAX_RESIDUAL_WORDS,
            "quantiles": "empirical nearest-rank",
        },
        "definitions": {
            "nearest_ancestor_edge": (
                "child path joined to every exact CLAUDE.md at the closest strict "
                "ancestor directory containing an observed exact file"
            ),
            "root_byte_share": (
                "bytes in exact root CLAUDE.md files divided by all exact "
                "CLAUDE.md bytes in the repository"
            ),
            "largest_file_byte_share": (
                "bytes in the largest observed exact file divided by repository "
                "exact-file bytes"
            ),
            "size_hhi": (
                "sum of squared per-file byte shares; higher values indicate "
                "greater within-repository concentration"
            ),
            "normalized_line": (
                "non-empty line after Unicode NFKC normalization, trimming, "
                "whitespace collapse, and case folding; duplicates form a set"
            ),
            "literal_edge_reference": (
                "an unfenced local Markdown reference lexically resolves to the "
                "observed path at the other end of a nearest-ancestor edge"
            ),
            "high_confidence_edge_reference": (
                "a literal edge reference also satisfies the Phase 2 direct-include "
                "or bounded instructional-cue rules"
            ),
            "heading_prevalence": (
                "documents containing at least one occurrence of an exact "
                "conservatively normalized heading name"
            ),
        },
        "population": {
            "repositories": len(repositories),
            "files": len(files),
            "root_files": sum(file.directory_depth == 0 for file in files),
            "nested_files": sum(file.directory_depth > 0 for file in files),
            "nearest_ancestor_edges": len(edges),
            "repositories_with_nearest_ancestor_edges": len(
                hierarchy_repositories
            ),
            "repositories_without_nearest_ancestor_edges": len(
                nonhierarchy_repositories
            ),
            "location_configurations": {
                key: configuration_counts[key]
                for key in ("root_only", "nested_only", "root_and_nested")
            },
        },
        "content_allocation": {
            "total_size_bytes": sum(row.total_size_bytes for row in repositories),
            "root_size_bytes": sum(row.root_size_bytes for row in repositories),
            "nested_size_bytes": sum(
                row.nested_size_bytes for row in repositories
            ),
            "root_byte_share_of_multi_file_population": _share(
                sum(row.root_size_bytes for row in repositories),
                sum(row.total_size_bytes for row in repositories),
            ),
            "root_byte_share_in_root_and_nested_repositories": (
                _numeric_distribution(row.root_byte_share for row in root_and_nested)
            ),
            "largest_file_byte_share": _numeric_distribution(
                row.largest_file_byte_share for row in repositories
            ),
            "size_hhi": _numeric_distribution(
                row.size_hhi for row in repositories
            ),
            "repositories_with_largest_file_at_least_half": sum(
                row.largest_file_byte_share >= 0.5 for row in repositories
            ),
            "repository_share_with_largest_file_at_least_half": _share(
                sum(row.largest_file_byte_share >= 0.5 for row in repositories),
                len(repositories),
            ),
            "repositories_with_largest_file_at_least_80_percent": sum(
                row.largest_file_byte_share >= 0.8 for row in repositories
            ),
            "repository_share_with_largest_file_at_least_80_percent": _share(
                sum(row.largest_file_byte_share >= 0.8 for row in repositories),
                len(repositories),
            ),
            "repositories_with_within_repository_exact_reuse": sum(
                row.within_repository_exact_duplicate_groups > 0
                for row in repositories
            ),
            "files_in_within_repository_exact_duplicate_groups": sum(
                row.files_in_within_repository_exact_duplicate_groups
                for row in repositories
            ),
            "pointer_only_candidate_files": sum(
                row.pointer_only_files for row in repositories
            ),
            "by_observed_file_count": _allocation_strata(
                repositories,
                dimension="observed_file_count",
            ),
            "by_location_configuration": _allocation_strata(
                repositories,
                dimension="location_configuration",
            ),
        },
        "parent_child_comparison": {
            "edges": len(edges),
            "size_bytes": {
                "parent": _numeric_distribution(
                    edge.parent_size_bytes for edge in edges
                ),
                "child": _numeric_distribution(
                    edge.child_size_bytes for edge in edges
                ),
                "child_minus_parent": _numeric_distribution(
                    edge.child_minus_parent_size_bytes for edge in edges
                ),
                "child_parent_ratio_nonzero_parent": _numeric_distribution(
                    edge.child_parent_size_ratio
                    for edge in edges
                    if edge.child_parent_size_ratio is not None
                ),
                "comparison": _comparison_counts(
                    edges,
                    "parent_size_bytes",
                    "child_size_bytes",
                ),
            },
            "words": {
                "parent": _numeric_distribution(
                    edge.parent_word_count for edge in edges
                ),
                "child": _numeric_distribution(
                    edge.child_word_count for edge in edges
                ),
                "child_minus_parent": _numeric_distribution(
                    edge.child_minus_parent_words for edge in edges
                ),
                "comparison": _comparison_counts(
                    edges,
                    "parent_word_count",
                    "child_word_count",
                ),
            },
            "sections": {
                "parent": _numeric_distribution(
                    edge.parent_section_count for edge in edges
                ),
                "child": _numeric_distribution(
                    edge.child_section_count for edge in edges
                ),
                "child_minus_parent": _numeric_distribution(
                    edge.child_minus_parent_sections for edge in edges
                ),
                "comparison": _comparison_counts(
                    edges,
                    "parent_section_count",
                    "child_section_count",
                ),
            },
            "code_blocks": {
                "parent": _numeric_distribution(
                    edge.parent_code_block_count for edge in edges
                ),
                "child": _numeric_distribution(
                    edge.child_code_block_count for edge in edges
                ),
                "child_minus_parent": _numeric_distribution(
                    edge.child_minus_parent_code_blocks for edge in edges
                ),
                "comparison": _comparison_counts(
                    edges,
                    "parent_code_block_count",
                    "child_code_block_count",
                ),
            },
            "exact_content_hash_edges": sum(
                edge.same_content_hash for edge in edges
            ),
            "exact_content_hash_edge_share": _share(
                sum(edge.same_content_hash for edge in edges), len(edges)
            ),
            "same_normalized_line_set_edges": sum(
                edge.same_normalized_line_set for edge in edges
            ),
            "same_normalized_line_set_edge_share": _share(
                sum(edge.same_normalized_line_set for edge in edges), len(edges)
            ),
            "different_hash_edges": len(different_hash_edges),
            "different_hash_same_normalized_line_set_edges": sum(
                edge.same_normalized_line_set for edge in different_hash_edges
            ),
            "jaccard_similarity": _numeric_distribution(
                edge.jaccard_similarity for edge in edges
            ),
            "jaccard_similarity_different_hash_edges": _numeric_distribution(
                edge.jaccard_similarity for edge in different_hash_edges
            ),
            "smaller_document_containment": _numeric_distribution(
                edge.smaller_document_containment for edge in edges
            ),
            "smaller_document_containment_different_hash_edges": (
                _numeric_distribution(
                    edge.smaller_document_containment
                    for edge in different_hash_edges
                )
            ),
            "parent_line_coverage": _numeric_distribution(
                edge.parent_line_coverage for edge in edges
            ),
            "child_line_coverage": _numeric_distribution(
                edge.child_line_coverage for edge in edges
            ),
            "child_unique_normalized_lines": _numeric_distribution(
                edge.child_unique_normalized_lines for edge in edges
            ),
            "child_unique_normalized_lines_different_hash_edges": (
                _numeric_distribution(
                    edge.child_unique_normalized_lines
                    for edge in different_hash_edges
                )
            ),
            "edges_with_no_child_unique_normalized_lines": sum(
                edge.child_unique_normalized_lines == 0 for edge in edges
            ),
            "edge_share_with_no_child_unique_normalized_lines": _share(
                sum(edge.child_unique_normalized_lines == 0 for edge in edges),
                len(edges),
            ),
        },
        "explicit_hierarchy_references": {
            "literal": {
                "parent_to_child_edges": sum(
                    edge.parent_references_child_literal for edge in edges
                ),
                "child_to_parent_edges": sum(
                    edge.child_references_parent_literal for edge in edges
                ),
                "either_direction_edges": literal_either,
                "either_direction_edge_share": _share(literal_either, len(edges)),
                "bidirectional_edges": sum(
                    edge.parent_references_child_literal
                    and edge.child_references_parent_literal
                    for edge in edges
                ),
            },
            "high_confidence_candidate": {
                "parent_to_child_edges": sum(
                    edge.parent_references_child_high_confidence for edge in edges
                ),
                "child_to_parent_edges": sum(
                    edge.child_references_parent_high_confidence for edge in edges
                ),
                "either_direction_edges": high_either,
                "either_direction_edge_share": _share(high_either, len(edges)),
                "bidirectional_edges": sum(
                    edge.parent_references_child_high_confidence
                    and edge.child_references_parent_high_confidence
                    for edge in edges
                ),
                "repositories_with_at_least_one_edge": sum(
                    row.either_direction_high_confidence_edges > 0
                    for row in repositories
                ),
            },
            "candidate_presence_by_path_hierarchy": {
                "hierarchy_repositories": len(hierarchy_repositories),
                "hierarchy_repositories_with_candidate_file": (
                    hierarchy_with_candidate
                ),
                "hierarchy_repository_candidate_share": _share(
                    hierarchy_with_candidate, len(hierarchy_repositories)
                ),
                "nonhierarchy_repositories": len(nonhierarchy_repositories),
                "nonhierarchy_repositories_with_candidate_file": (
                    nonhierarchy_with_candidate
                ),
                "nonhierarchy_repository_candidate_share": _share(
                    nonhierarchy_with_candidate, len(nonhierarchy_repositories)
                ),
                "unadjusted_odds_ratio_95_interval": _odds_ratio_interval(
                    hierarchy_with_candidate,
                    hierarchy_without_candidate,
                    nonhierarchy_with_candidate,
                    nonhierarchy_without_candidate,
                ),
            },
        },
        "headings": {
            "document_denominators": {
                "all_multi_file_documents": len(files),
                "root_documents": sum(
                    file.directory_depth == 0 for file in files
                ),
                "nested_documents": sum(
                    file.directory_depth > 0 for file in files
                ),
                "parent_role_documents": len(parent_role_documents),
                "child_role_documents": len(child_role_documents),
            },
            "output_rows": len(headings),
            "output_limit": heading_limit,
            "top_20": [asdict(row) for row in headings[:20]],
        },
        "validity": {
            "path_hierarchy_is_not_runtime_loading": True,
            "references_are_lexically_resolved_without_frozen_tree_verification": True,
            "high_confidence_reference_intent_is_not_independently_validated": True,
            "pointer_only_is_an_operational_candidate_label": True,
            "normalized_line_sets_discard_order_repetition_case_and_formatting": True,
            "repositories_may_be_incomplete_or_observed_at_different_revisions": True,
            "manual_dataset_revision_required": False,
        },
        "database": {
            "sha256_before": database_sha256_before,
            "sha256_after": database_sha256_after,
            "total_changes": database_changes,
            "unchanged": (
                database_sha256_before == database_sha256_after
                and database_changes == 0
            ),
        },
    }


def analyze_hierarchy_allocation(
    db_path: str | Path,
    *,
    heading_limit: int = 1000,
) -> tuple[
    dict[str, object],
    list[RepositoryAllocationRow],
    list[HierarchyAllocationEdgeRow],
    list[HeadingComparisonRow],
]:
    """Return deterministic hierarchy-allocation results without DB writes."""
    if heading_limit <= 0:
        raise ValueError("heading_limit must be positive")
    database_sha256_before = _sha256_file(db_path)
    connection = _readonly_connection(db_path)
    predicate = _exact_predicate()
    aliased_predicate = _exact_predicate(alias="f")
    query = f"""
        WITH multi_repositories AS (
            SELECT repo_full_name
            FROM files
            WHERE {predicate}
            GROUP BY repo_full_name
            HAVING COUNT(*) > 1
        )
        SELECT f.id, f.repo_full_name, f.path, f.content_hash,
               f.size_bytes, f.content
        FROM files AS f
        INNER JOIN multi_repositories AS m
                ON m.repo_full_name = f.repo_full_name
        WHERE {aliased_predicate}
        ORDER BY lower(f.repo_full_name), f.repo_full_name,
                 lower(f.path), f.path
    """
    content_cache: dict[str, tuple[ContentMetrics, object]] = {}
    repositories: list[RepositoryAllocationRow] = []
    edge_rows: list[HierarchyAllocationEdgeRow] = []
    all_files: list[FileMetrics] = []
    heading_counts = {
        dimension: Counter()
        for dimension in ("all", "root", "nested", "parent", "child")
    }
    heading_denominators = Counter()

    current_repository: str | None = None
    current_files: list[FileMetrics] = []

    def flush_repository() -> None:
        nonlocal current_files
        if current_repository is None:
            return
        current_files.sort(key=lambda item: _path_sort_key(item.path))
        file_records = [
            FileRecord(
                repo_full_name=file.repo_full_name,
                path=file.path,
                content_hash=file.content_hash or None,
                size_bytes=file.size_bytes,
            )
            for file in current_files
        ]
        hierarchy, _children, _parents, _chain, _same_directory = (
            _build_hierarchy(current_repository, file_records)
        )
        by_path = {file.path: file for file in current_files}
        repository_edges = [_edge_row(edge, by_path) for edge in hierarchy]
        edge_rows.extend(repository_edges)
        repositories.append(
            _repository_row(
                current_repository,
                current_files,
                repository_edges,
            )
        )

        parent_paths = {edge.parent_path for edge in hierarchy}
        child_paths = {edge.child_path for edge in hierarchy}
        for file in current_files:
            roles = ["all", "root" if file.directory_depth == 0 else "nested"]
            if file.path in parent_paths:
                roles.append("parent")
            if file.path in child_paths:
                roles.append("child")
            for role in roles:
                heading_denominators[role] += 1
                for heading in file.heading_names:
                    heading_counts[role][heading] += 1
        all_files.extend(current_files)
        current_files = []

    try:
        for row in connection.execute(query):
            repo_full_name = str(row["repo_full_name"])
            if (
                current_repository is not None
                and repo_full_name != current_repository
            ):
                flush_repository()
            current_repository = repo_full_name
            current_files.append(
                _file_metrics(row, content_cache=content_cache)
            )
        flush_repository()
        database_changes = connection.total_changes
    finally:
        connection.close()

    repositories.sort(
        key=lambda row: (row.repo_full_name.casefold(), row.repo_full_name)
    )
    edge_rows.sort(
        key=lambda row: (
            row.repo_full_name.casefold(),
            row.repo_full_name,
            *_path_sort_key(row.parent_path),
            *_path_sort_key(row.child_path),
        )
    )
    heading_rows = _heading_rows(
        heading_counts,
        dict(heading_denominators),
        limit=heading_limit,
    )
    database_sha256_after = _sha256_file(db_path)
    summary = _build_summary(
        repositories,
        edge_rows,
        heading_rows,
        files=all_files,
        database_sha256_before=database_sha256_before,
        database_sha256_after=database_sha256_after,
        database_changes=database_changes,
        heading_limit=heading_limit,
    )
    return summary, repositories, edge_rows, heading_rows


def _write_dataclass_csv(
    rows: Sequence[object],
    output_path: str | Path,
    *,
    fields: Sequence[str],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = asdict(row)  # type: ignore[arg-type]
            for key, value in data.items():
                if isinstance(value, float):
                    data[key] = f"{value:.6f}"
                elif value is None:
                    data[key] = ""
            writer.writerow(data)
    return output


def write_repository_csv(
    rows: Sequence[RepositoryAllocationRow],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        rows,
        output_path,
        fields=tuple(RepositoryAllocationRow.__dataclass_fields__),
    )


def write_edge_csv(
    rows: Sequence[HierarchyAllocationEdgeRow],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        rows,
        output_path,
        fields=tuple(HierarchyAllocationEdgeRow.__dataclass_fields__),
    )


def write_heading_csv(
    rows: Sequence[HeadingComparisonRow],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        rows,
        output_path,
        fields=tuple(HeadingComparisonRow.__dataclass_fields__),
    )


def write_summary_json(
    summary: dict[str, object],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--repositories-output",
        default="article/claude_hierarchy_allocation_repositories.csv",
        help="one-row-per-multi-file-repository CSV output",
    )
    parser.add_argument(
        "--edges-output",
        default="article/claude_hierarchy_allocation_edges.csv",
        help="one-row-per-nearest-ancestor-edge CSV output",
    )
    parser.add_argument(
        "--headings-output",
        default="article/claude_hierarchy_heading_comparison.csv",
        help="ranked root/nested and parent/child heading CSV output",
    )
    parser.add_argument(
        "--summary-output",
        default="article/claude_hierarchy_allocation_summary.json",
        help="machine-readable summary JSON output",
    )
    parser.add_argument(
        "--heading-limit",
        type=positive_int,
        default=1000,
        help="maximum heading rows to export (default: 1000)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary, repositories, edges, headings = analyze_hierarchy_allocation(
            arguments.db,
            heading_limit=arguments.heading_limit,
        )
        repository_path = write_repository_csv(
            repositories,
            arguments.repositories_output,
        )
        edge_path = write_edge_csv(edges, arguments.edges_output)
        heading_path = write_heading_csv(headings, arguments.headings_output)
        summary_path = write_summary_json(summary, arguments.summary_output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    population = summary["population"]
    allocation = summary["content_allocation"]
    references = summary["explicit_hierarchy_references"]
    database = summary["database"]
    assert isinstance(population, dict)
    assert isinstance(allocation, dict)
    assert isinstance(references, dict)
    assert isinstance(database, dict)
    high = references["high_confidence_candidate"]
    assert isinstance(high, dict)

    print(f"Scope: {summary['scope']}")
    print(
        f"Repositories / files: {population['repositories']:,} / "
        f"{population['files']:,}"
    )
    print(f"Nearest-ancestor edges: {population['nearest_ancestor_edges']:,}")
    print(
        "Repositories with largest-file byte share >= 0.80: "
        f"{allocation['repositories_with_largest_file_at_least_80_percent']:,}"
    )
    print(
        "High-confidence hierarchy-linked edges: "
        f"{high['either_direction_edges']:,}"
    )
    print(f"Repositories CSV: {repository_path.resolve()}")
    print(f"Edges CSV: {edge_path.resolve()}")
    print(f"Headings CSV: {heading_path.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print(f"Database unchanged: {database['unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
