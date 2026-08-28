"""Analyze repository-level placement of exact ``CLAUDE.md`` files.

The script reads ``mined.db`` without modifying it.  Its population is fixed
to files whose case-insensitive basename is exactly ``CLAUDE.md``.  It reports
file location, repository multiplicity, nearest-ancestor relationships between
instruction-file directories, within-repository exact reuse, and exploratory
normalized-line overlap between non-identical files.

Example:

    python scripts/claude_repository_structure.py \\
        --db mined.db \\
        --repositories-output article/claude_repository_structure.csv \\
        --components-output article/claude_path_components.csv \\
        --hierarchy-output article/claude_hierarchy_edges.csv \\
        --overlap-output article/claude_intra_repo_line_overlap.csv \\
        --summary-output article/claude_repository_structure_summary.json

SQLite is opened with ``mode=ro&immutable=1`` and ``PRAGMA query_only=ON``.
Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


EXACT_CLAUDE_SCOPE = (
    "case-insensitive basename equal to CLAUDE.md; stored GitHub paths use "
    "POSIX '/' separators"
)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FileRecord:
    repo_full_name: str
    path: str
    content_hash: str | None
    size_bytes: int | None

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    @property
    def directory_parts(self) -> tuple[str, ...]:
        return self.parts[:-1]

    @property
    def directory_depth(self) -> int:
        return len(self.directory_parts)

    @property
    def placement_form(self) -> str:
        """Return the documented direct or hidden-directory placement form."""
        if (
            self.directory_parts
            and self.directory_parts[-1].casefold() == ".claude"
        ):
            return "hidden_claude_directory"
        return "direct"

    @property
    def logical_scope_parts(self) -> tuple[str, ...]:
        """Return the directory to which the instruction file is scoped.

        Anthropic documents ``D/CLAUDE.md`` and ``D/.claude/CLAUDE.md`` as
        alternative project placements.  The latter is therefore normalized
        to scope directory ``D`` instead of the physical ``D/.claude``
        directory.  This is a repository-path normalization, not evidence
        that a particular Claude Code session loaded the file.
        """
        if self.placement_form == "hidden_claude_directory":
            return self.directory_parts[:-1]
        return self.directory_parts

    @property
    def logical_scope_depth(self) -> int:
        return len(self.logical_scope_parts)


@dataclass(frozen=True)
class ContentRecord:
    path: str
    content_hash: str | None
    size_bytes: int | None
    normalized_lines: frozenset[str]


@dataclass
class RepositoryStructure:
    repo_full_name: str
    exact_files: int
    root_files: int
    nested_files: int
    maximum_directory_depth: int
    location_configuration: str
    hierarchy_edges: int = 0
    files_with_nearest_ancestor: int = 0
    files_serving_as_nearest_ancestor: int = 0
    maximum_instruction_scope_chain: int = 1
    same_directory_file_groups: int = 0
    exact_duplicate_groups: int = 0
    files_in_exact_duplicate_groups: int = 0
    repeated_exact_instances: int = 0
    exact_duplicate_pairs: int = 0
    normalized_equivalent_pairs: int = 0
    high_jaccard_pairs: int = 0
    high_containment_pairs: int = 0
    files_in_nonexact_overlap_pairs: int = 0


@dataclass(frozen=True)
class HierarchyEdge:
    repo_full_name: str
    parent_path: str
    child_path: str
    parent_directory_depth: int
    child_directory_depth: int
    directory_distance: int
    parent_candidates_at_nearest_scope: int


@dataclass(frozen=True)
class LineOverlapPair:
    repo_full_name: str
    path_a: str
    path_b: str
    relation: str
    size_bytes_a: int | None
    size_bytes_b: int | None
    normalized_lines_a: int
    normalized_lines_b: int
    shared_normalized_lines: int
    jaccard_similarity: float
    smaller_document_containment: float


@dataclass
class ComponentAccumulator:
    file_count: int = 0
    occurrence_count: int = 0
    repositories: set[str] = field(default_factory=set)
    example_paths: list[str] = field(default_factory=list)

    def observe(
        self,
        *,
        repo_full_name: str,
        path: str,
        occurrences: int = 1,
    ) -> None:
        self.file_count += 1
        self.occurrence_count += occurrences
        self.repositories.add(repo_full_name)
        if len(self.example_paths) < 3:
            self.example_paths.append(f"{repo_full_name}::{path}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


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
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _nearest_rank(sorted_values: Sequence[int], probability: float) -> int:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty population")
    rank = max(1, math.ceil(probability * len(sorted_values)))
    return int(sorted_values[rank - 1])


def _integer_distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": 0,
            "p25": 0,
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "maximum": 0,
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


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _location_configuration(root_files: int, nested_files: int) -> str:
    if root_files and nested_files:
        return "root_and_nested"
    if root_files:
        return "root_only"
    return "nested_only"


def _nearest_ancestor_directory(
    directory: tuple[str, ...],
    available_directories: set[tuple[str, ...]],
) -> tuple[str, ...] | None:
    for length in range(len(directory) - 1, -1, -1):
        candidate = directory[:length]
        if candidate in available_directories:
            return candidate
    return None


def _instruction_scope_chain_length(
    directory: tuple[str, ...],
    available_directories: set[tuple[str, ...]],
    memo: dict[tuple[str, ...], int],
) -> int:
    cached = memo.get(directory)
    if cached is not None:
        return cached
    parent = _nearest_ancestor_directory(directory, available_directories)
    length = (
        1
        if parent is None
        else 1
        + _instruction_scope_chain_length(parent, available_directories, memo)
    )
    memo[directory] = length
    return length


def _normalize_lines(content: str | None) -> frozenset[str]:
    if not content:
        return frozenset()
    normalized: set[str] = set()
    for raw_line in content.splitlines():
        line = unicodedata.normalize("NFKC", raw_line)
        line = WHITESPACE_RE.sub(" ", line.strip()).casefold()
        if line:
            normalized.add(line)
    return frozenset(normalized)


def _path_sort_key(path: str) -> tuple[str, str]:
    return path.casefold(), path


def _repository_sort_key(name: str) -> tuple[str, str]:
    return name.casefold(), name


def _build_hierarchy(
    repo_full_name: str,
    files: Sequence[FileRecord],
) -> tuple[list[HierarchyEdge], int, int, int, int]:
    by_directory: dict[tuple[str, ...], list[FileRecord]] = defaultdict(list)
    for file in files:
        by_directory[file.directory_parts].append(file)
    for members in by_directory.values():
        members.sort(key=lambda member: _path_sort_key(member.path))

    directories = set(by_directory)
    edges: list[HierarchyEdge] = []
    child_paths: set[str] = set()
    parent_paths: set[str] = set()
    for child in files:
        parent_directory = _nearest_ancestor_directory(
            child.directory_parts,
            directories,
        )
        if parent_directory is None:
            continue
        candidates = by_directory[parent_directory]
        child_paths.add(child.path)
        for parent in candidates:
            parent_paths.add(parent.path)
            edges.append(
                HierarchyEdge(
                    repo_full_name=repo_full_name,
                    parent_path=parent.path,
                    child_path=child.path,
                    parent_directory_depth=parent.directory_depth,
                    child_directory_depth=child.directory_depth,
                    directory_distance=(
                        child.directory_depth - parent.directory_depth
                    ),
                    parent_candidates_at_nearest_scope=len(candidates),
                )
            )

    memo: dict[tuple[str, ...], int] = {}
    maximum_chain = max(
        (
            _instruction_scope_chain_length(directory, directories, memo)
            for directory in directories
        ),
        default=0,
    )
    same_directory_groups = sum(
        1 for members in by_directory.values() if len(members) > 1
    )
    return (
        edges,
        len(child_paths),
        len(parent_paths),
        maximum_chain,
        same_directory_groups,
    )


def _documented_scope_mapping(
    files_by_repository: dict[str, list[FileRecord]],
) -> dict[str, object]:
    """Summarize documented placement forms and normalized logical scopes.

    The original Phase 1 hierarchy remains a literal stored-directory
    topology.  This companion summary case-folds a terminal ``.claude``
    directory and maps it back to its containing scope so that physical
    placement is not confused with Anthropic's documented project-location
    semantics. Exact-case and casing-variant counts remain separate.
    """
    exact_files = sum(len(files) for files in files_by_repository.values())
    direct_files = 0
    hidden_files = 0
    hidden_exact_case_files = 0
    hidden_case_variant_files = 0
    project_root_scope_files = 0
    nonroot_scope_files = 0
    project_root_direct_files = 0
    project_root_hidden_files = 0
    project_root_hidden_exact_case_files = 0
    project_root_hidden_case_variant_files = 0
    direct_placement_repositories: set[str] = set()
    hidden_placement_repositories: set[str] = set()
    project_root_direct_repositories: set[str] = set()
    project_root_hidden_repositories: set[str] = set()
    alternative_scope_repositories: set[str] = set()
    alternative_scope_groups = 0
    alternative_scope_file_pairs = 0
    alternative_scope_same_hash_pairs = 0
    project_root_alternative_groups = 0
    nonroot_alternative_groups = 0
    logical_scope_edges = 0
    logical_scope_edge_repositories = 0
    maximum_logical_scope_chain = 0
    unique_logical_scopes = 0
    nonroot_scope_depths: list[int] = []
    configuration_accumulator: dict[str, dict[str, int]] = {
        "project_root_only": {"repositories": 0, "files": 0},
        "nonroot_only": {"repositories": 0, "files": 0},
        "project_root_and_nonroot": {"repositories": 0, "files": 0},
    }

    for repo_full_name, files in files_by_repository.items():
        by_scope: dict[tuple[str, ...], list[FileRecord]] = defaultdict(list)
        for file in files:
            by_scope[file.logical_scope_parts].append(file)
            if file.placement_form == "hidden_claude_directory":
                hidden_files += 1
                hidden_placement_repositories.add(repo_full_name)
                if file.directory_parts[-1] == ".claude":
                    hidden_exact_case_files += 1
                else:
                    hidden_case_variant_files += 1
            else:
                direct_files += 1
                direct_placement_repositories.add(repo_full_name)
            if file.logical_scope_parts:
                nonroot_scope_files += 1
                nonroot_scope_depths.append(file.logical_scope_depth)
            else:
                project_root_scope_files += 1
                if file.placement_form == "hidden_claude_directory":
                    project_root_hidden_files += 1
                    project_root_hidden_repositories.add(repo_full_name)
                    if file.directory_parts[-1] == ".claude":
                        project_root_hidden_exact_case_files += 1
                    else:
                        project_root_hidden_case_variant_files += 1
                else:
                    project_root_direct_files += 1
                    project_root_direct_repositories.add(repo_full_name)

        unique_logical_scopes += len(by_scope)
        has_project_root = () in by_scope
        has_nonroot = any(scope for scope in by_scope)
        if has_project_root and has_nonroot:
            configuration = "project_root_and_nonroot"
        elif has_project_root:
            configuration = "project_root_only"
        else:
            configuration = "nonroot_only"
        configuration_accumulator[configuration]["repositories"] += 1
        configuration_accumulator[configuration]["files"] += len(files)

        for scope, members in by_scope.items():
            direct = [
                member for member in members if member.placement_form == "direct"
            ]
            hidden = [
                member
                for member in members
                if member.placement_form == "hidden_claude_directory"
            ]
            if not direct or not hidden:
                continue
            alternative_scope_groups += 1
            alternative_scope_repositories.add(repo_full_name)
            if scope:
                nonroot_alternative_groups += 1
            else:
                project_root_alternative_groups += 1
            for direct_file in direct:
                for hidden_file in hidden:
                    alternative_scope_file_pairs += 1
                    if (
                        direct_file.content_hash is not None
                        and direct_file.content_hash == hidden_file.content_hash
                    ):
                        alternative_scope_same_hash_pairs += 1

        scopes = set(by_scope)
        repository_edges = 0
        memo: dict[tuple[str, ...], int] = {}
        for scope in scopes:
            if _nearest_ancestor_directory(scope, scopes) is not None:
                repository_edges += 1
        logical_scope_edges += repository_edges
        logical_scope_edge_repositories += repository_edges > 0
        maximum_logical_scope_chain = max(
            maximum_logical_scope_chain,
            max(
                (
                    _instruction_scope_chain_length(scope, scopes, memo)
                    for scope in scopes
                ),
                default=0,
            ),
        )

    configurations = {
        name: {
            **values,
            "repository_share": _share(
                values["repositories"],
                len(files_by_repository),
            ),
            "file_share": _share(values["files"], exact_files),
        }
        for name, values in configuration_accumulator.items()
    }
    repositories_with_either_project_root_form = (
        project_root_direct_repositories | project_root_hidden_repositories
    )
    repositories_with_both_project_root_forms = (
        project_root_direct_repositories & project_root_hidden_repositories
    )

    return {
        "status": (
            "documented-location normalization of stored paths; not an observed "
            "Claude Code session or runtime load graph"
        ),
        "placement_forms": {
            "direct_files": direct_files,
            "direct_repositories": len(direct_placement_repositories),
            "hidden_claude_directory_files": hidden_files,
            "hidden_claude_directory_exact_case_files": hidden_exact_case_files,
            "hidden_claude_directory_case_variant_files": (
                hidden_case_variant_files
            ),
            "hidden_claude_directory_repositories": len(
                hidden_placement_repositories
            ),
            "hidden_claude_directory_file_share": _share(
                hidden_files,
                exact_files,
            ),
        },
        "logical_file_location": {
            "project_root_scope_files": project_root_scope_files,
            "project_root_scope_file_share": _share(
                project_root_scope_files,
                exact_files,
            ),
            "nonroot_scope_files": nonroot_scope_files,
            "nonroot_scope_file_share": _share(
                nonroot_scope_files,
                exact_files,
            ),
            "project_root_direct_files": project_root_direct_files,
            "project_root_hidden_files": project_root_hidden_files,
            "project_root_hidden_exact_case_files": (
                project_root_hidden_exact_case_files
            ),
            "project_root_hidden_case_variant_files": (
                project_root_hidden_case_variant_files
            ),
            "nonroot_scope_depth_distribution": _integer_distribution(
                nonroot_scope_depths
            ),
        },
        "logical_location_configurations": configurations,
        "project_root_placement_repositories": {
            "direct_form": len(project_root_direct_repositories),
            "hidden_form": len(project_root_hidden_repositories),
            "either_form": len(repositories_with_either_project_root_form),
            "both_forms": len(repositories_with_both_project_root_forms),
            "hidden_form_without_direct_form": len(
                project_root_hidden_repositories
                - project_root_direct_repositories
            ),
        },
        "same_logical_scope_alternative_placements": {
            "scope_groups": alternative_scope_groups,
            "repositories": len(alternative_scope_repositories),
            "project_root_scope_groups": project_root_alternative_groups,
            "nonroot_scope_groups": nonroot_alternative_groups,
            "file_pairs": alternative_scope_file_pairs,
            "same_content_hash_file_pairs": alternative_scope_same_hash_pairs,
            "different_content_hash_file_pairs": (
                alternative_scope_file_pairs - alternative_scope_same_hash_pairs
            ),
        },
        "logical_scope_topology": {
            "unique_repository_scopes": unique_logical_scopes,
            "nearest_ancestor_scope_edges": logical_scope_edges,
            "repositories_with_nearest_ancestor_scope_edges": (
                logical_scope_edge_repositories
            ),
            "maximum_scope_chain": maximum_logical_scope_chain,
        },
    }


def _exact_reuse(files: Sequence[FileRecord]) -> tuple[int, int, int, int]:
    counts = Counter(
        file.content_hash for file in files if file.content_hash is not None
    )
    repeated = [count for count in counts.values() if count > 1]
    return (
        len(repeated),
        sum(repeated),
        sum(count - 1 for count in repeated),
        sum(math.comb(count, 2) for count in repeated),
    )


def _process_overlap_repository(
    repo_full_name: str,
    files: Sequence[ContentRecord],
    repository: RepositoryStructure,
    *,
    minimum_shared_lines: int,
    jaccard_threshold: float,
    containment_threshold: float,
) -> tuple[list[LineOverlapPair], int, int]:
    pairs: list[LineOverlapPair] = []
    evaluated_nonexact_pairs = 0
    involved_paths: set[str] = set()

    for left, right in itertools.combinations(files, 2):
        if (
            left.content_hash is not None
            and left.content_hash == right.content_hash
        ):
            continue
        evaluated_nonexact_pairs += 1
        if not left.normalized_lines or not right.normalized_lines:
            continue

        shared = len(left.normalized_lines & right.normalized_lines)
        union = len(left.normalized_lines | right.normalized_lines)
        smaller = min(len(left.normalized_lines), len(right.normalized_lines))
        jaccard = shared / union
        containment = shared / smaller

        relation: str | None = None
        if left.normalized_lines == right.normalized_lines:
            relation = "normalized_equivalent"
        elif shared >= minimum_shared_lines and jaccard >= jaccard_threshold:
            relation = "high_jaccard"
        elif (
            shared >= minimum_shared_lines
            and containment >= containment_threshold
        ):
            relation = "high_containment"

        if relation is None:
            continue
        involved_paths.update((left.path, right.path))
        if relation == "normalized_equivalent":
            repository.normalized_equivalent_pairs += 1
        elif relation == "high_jaccard":
            repository.high_jaccard_pairs += 1
        else:
            repository.high_containment_pairs += 1
        pairs.append(
            LineOverlapPair(
                repo_full_name=repo_full_name,
                path_a=left.path,
                path_b=right.path,
                relation=relation,
                size_bytes_a=left.size_bytes,
                size_bytes_b=right.size_bytes,
                normalized_lines_a=len(left.normalized_lines),
                normalized_lines_b=len(right.normalized_lines),
                shared_normalized_lines=shared,
                jaccard_similarity=jaccard,
                smaller_document_containment=containment,
            )
        )

    repository.files_in_nonexact_overlap_pairs = len(involved_paths)
    return pairs, math.comb(len(files), 2), evaluated_nonexact_pairs


def _component_rows(
    statistics: dict[str, dict[str, ComponentAccumulator]],
    *,
    nested_files: int,
    repositories_with_nested_files: int,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    dimension_order = ("first_directory", "immediate_parent", "any_directory")
    for dimension in dimension_order:
        ordered = sorted(
            statistics[dimension].items(),
            key=lambda item: (-item[1].file_count, item[0]),
        )
        for rank, (component, accumulator) in enumerate(ordered, 1):
            rows.append(
                {
                    "dimension": dimension,
                    "rank": rank,
                    "component_casefolded": component,
                    "file_count": accumulator.file_count,
                    "file_share_of_nested_files": _share(
                        accumulator.file_count,
                        nested_files,
                    ),
                    "repository_count": len(accumulator.repositories),
                    "repository_share_of_repositories_with_nested_files": _share(
                        len(accumulator.repositories),
                        repositories_with_nested_files,
                    ),
                    "occurrence_count": accumulator.occurrence_count,
                    "example_paths": " | ".join(accumulator.example_paths),
                }
            )
    return rows


def _multiplicity_bins(counts: Sequence[int]) -> list[dict[str, int | float | str]]:
    definitions = (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5", 5, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        (">100", 101, None),
    )
    rows: list[dict[str, int | float | str]] = []
    for label, lower, upper in definitions:
        repositories = sum(
            1
            for count in counts
            if count >= lower and (upper is None or count <= upper)
        )
        files = sum(
            count
            for count in counts
            if count >= lower and (upper is None or count <= upper)
        )
        rows.append(
            {
                "file_count_bin": label,
                "repositories": repositories,
                "repository_share": _share(repositories, len(counts)),
                "files": files,
                "file_share": _share(files, sum(counts)),
            }
        )
    return rows


def analyze_repository_structure(
    db_path: str | Path,
    *,
    minimum_shared_lines: int = 5,
    jaccard_threshold: float = 0.80,
    containment_threshold: float = 0.90,
) -> tuple[
    dict[str, object],
    list[RepositoryStructure],
    list[dict[str, int | float | str]],
    list[HierarchyEdge],
    list[LineOverlapPair],
]:
    """Return deterministic repository-placement and overlap results."""
    if minimum_shared_lines <= 0:
        raise ValueError("minimum_shared_lines must be positive")
    for name, value in (
        ("jaccard_threshold", jaccard_threshold),
        ("containment_threshold", containment_threshold),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1")

    connection = _readonly_connection(db_path)
    predicate = _exact_predicate()
    files_by_repository: dict[str, list[FileRecord]] = defaultdict(list)
    path_anomalies = Counter()
    component_statistics: dict[str, dict[str, ComponentAccumulator]] = {
        dimension: defaultdict(ComponentAccumulator)
        for dimension in ("first_directory", "immediate_parent", "any_directory")
    }
    basename_casing = Counter()

    try:
        rows = connection.execute(
            f"""
            SELECT repo_full_name, path, content_hash, size_bytes
            FROM files
            WHERE {predicate}
            ORDER BY lower(repo_full_name), repo_full_name, lower(path), path
            """
        )
        for row in rows:
            record = FileRecord(
                repo_full_name=str(row["repo_full_name"]),
                path=str(row["path"]),
                content_hash=row["content_hash"],
                size_bytes=(
                    int(row["size_bytes"])
                    if row["size_bytes"] is not None
                    else None
                ),
            )
            files_by_repository[record.repo_full_name].append(record)
            basename_casing[record.parts[-1]] += 1
            if record.path.startswith("/"):
                path_anomalies["leading_slash"] += 1
            if "//" in record.path:
                path_anomalies["empty_component"] += 1
            if "\\" in record.path:
                path_anomalies["backslash"] += 1
            if any(part in (".", "..") for part in record.parts):
                path_anomalies["dot_component"] += 1

            if not record.directory_parts:
                continue
            folded = [part.casefold() for part in record.directory_parts]
            observations = {
                "first_directory": Counter((folded[0],)),
                "immediate_parent": Counter((folded[-1],)),
                "any_directory": Counter(folded),
            }
            for dimension, counts in observations.items():
                for component, occurrences in counts.items():
                    component_statistics[dimension][component].observe(
                        repo_full_name=record.repo_full_name,
                        path=record.path,
                        occurrences=occurrences,
                    )

        repositories: list[RepositoryStructure] = []
        hierarchy_edges: list[HierarchyEdge] = []
        for repo_full_name in sorted(
            files_by_repository,
            key=_repository_sort_key,
        ):
            files = files_by_repository[repo_full_name]
            files.sort(key=lambda file: _path_sort_key(file.path))
            root_files = sum(file.directory_depth == 0 for file in files)
            nested_files = len(files) - root_files
            repository = RepositoryStructure(
                repo_full_name=repo_full_name,
                exact_files=len(files),
                root_files=root_files,
                nested_files=nested_files,
                maximum_directory_depth=max(
                    (file.directory_depth for file in files),
                    default=0,
                ),
                location_configuration=_location_configuration(
                    root_files,
                    nested_files,
                ),
            )
            (
                edges,
                repository.files_with_nearest_ancestor,
                repository.files_serving_as_nearest_ancestor,
                repository.maximum_instruction_scope_chain,
                repository.same_directory_file_groups,
            ) = _build_hierarchy(repo_full_name, files)
            repository.hierarchy_edges = len(edges)
            hierarchy_edges.extend(edges)
            (
                repository.exact_duplicate_groups,
                repository.files_in_exact_duplicate_groups,
                repository.repeated_exact_instances,
                repository.exact_duplicate_pairs,
            ) = _exact_reuse(files)
            repositories.append(repository)

        repository_by_name = {
            repository.repo_full_name: repository for repository in repositories
        }
        overlap_pairs: list[LineOverlapPair] = []
        candidate_pairs = 0
        evaluated_nonexact_pairs = 0
        missing_content_files = 0
        aliased_predicate = _exact_predicate(alias="f")
        content_rows = connection.execute(
            f"""
            WITH multi_repositories AS (
                SELECT repo_full_name
                FROM files
                WHERE {predicate}
                GROUP BY repo_full_name
                HAVING count(*) > 1
            )
            SELECT f.repo_full_name, f.path, f.content_hash, f.size_bytes,
                   f.content
            FROM files AS f
            INNER JOIN multi_repositories AS m
                    ON m.repo_full_name = f.repo_full_name
            WHERE {aliased_predicate}
            ORDER BY lower(f.repo_full_name), f.repo_full_name,
                     lower(f.path), f.path
            """
        )

        current_repository: str | None = None
        content_group: list[ContentRecord] = []

        def flush_content_group() -> None:
            nonlocal candidate_pairs, evaluated_nonexact_pairs
            if current_repository is None:
                return
            pairs, candidates, evaluated = _process_overlap_repository(
                current_repository,
                content_group,
                repository_by_name[current_repository],
                minimum_shared_lines=minimum_shared_lines,
                jaccard_threshold=jaccard_threshold,
                containment_threshold=containment_threshold,
            )
            overlap_pairs.extend(pairs)
            candidate_pairs += candidates
            evaluated_nonexact_pairs += evaluated

        for row in content_rows:
            repo_full_name = str(row["repo_full_name"])
            if current_repository is not None and repo_full_name != current_repository:
                flush_content_group()
                content_group = []
            current_repository = repo_full_name
            content = row["content"]
            if content is None:
                missing_content_files += 1
            content_group.append(
                ContentRecord(
                    path=str(row["path"]),
                    content_hash=row["content_hash"],
                    size_bytes=(
                        int(row["size_bytes"])
                        if row["size_bytes"] is not None
                        else None
                    ),
                    normalized_lines=_normalize_lines(content),
                )
            )
        flush_content_group()
    finally:
        connection.close()

    hierarchy_edges.sort(
        key=lambda edge: (
            *_repository_sort_key(edge.repo_full_name),
            *_path_sort_key(edge.parent_path),
            *_path_sort_key(edge.child_path),
        )
    )
    overlap_pairs.sort(
        key=lambda pair: (
            *_repository_sort_key(pair.repo_full_name),
            pair.relation,
            *_path_sort_key(pair.path_a),
            *_path_sort_key(pair.path_b),
        )
    )

    exact_files = sum(repository.exact_files for repository in repositories)
    root_files = sum(repository.root_files for repository in repositories)
    nested_files = exact_files - root_files
    multi_repositories = [
        repository for repository in repositories if repository.exact_files > 1
    ]
    repositories_with_nested = sum(
        repository.nested_files > 0 for repository in repositories
    )
    component_rows = _component_rows(
        component_statistics,
        nested_files=nested_files,
        repositories_with_nested_files=repositories_with_nested,
    )
    configurations = {}
    for configuration in ("root_only", "nested_only", "root_and_nested"):
        selected = [
            repository
            for repository in repositories
            if repository.location_configuration == configuration
        ]
        configurations[configuration] = {
            "repositories": len(selected),
            "repository_share": _share(len(selected), len(repositories)),
            "files": sum(repository.exact_files for repository in selected),
            "file_share": _share(
                sum(repository.exact_files for repository in selected),
                exact_files,
            ),
        }

    overlap_relation_counts = Counter(pair.relation for pair in overlap_pairs)
    overlap_relation_repositories = {
        relation: len(
            {
                pair.repo_full_name
                for pair in overlap_pairs
                if pair.relation == relation
            }
        )
        for relation in (
            "normalized_equivalent",
            "high_jaccard",
            "high_containment",
        )
    }
    top_components: dict[str, list[dict[str, int | float | str]]] = {}
    for dimension in ("first_directory", "immediate_parent", "any_directory"):
        top_components[dimension] = [
            row
            for row in component_rows
            if row["dimension"] == dimension and int(row["rank"]) <= 20
        ]

    depth_counts = Counter(
        file.directory_depth
        for files in files_by_repository.values()
        for file in files
    )
    multiplicity_counts = [repository.exact_files for repository in repositories]
    hierarchy_repositories = [
        repository for repository in repositories if repository.hierarchy_edges > 0
    ]
    exact_reuse_repositories = [
        repository
        for repository in repositories
        if repository.exact_duplicate_groups > 0
    ]
    nonexact_overlap_repositories = [
        repository
        for repository in repositories
        if (
            repository.normalized_equivalent_pairs
            + repository.high_jaccard_pairs
            + repository.high_containment_pairs
        )
        > 0
    ]

    summary: dict[str, object] = {
        "analysis": "repository-level placement of exact CLAUDE.md files",
        "scope": "exact-claude",
        "scope_definition": EXACT_CLAUDE_SCOPE,
        "parameters": {
            "minimum_shared_normalized_lines": minimum_shared_lines,
            "jaccard_threshold": jaccard_threshold,
            "containment_threshold": containment_threshold,
        },
        "definitions": {
            "root_file": "stored path has no '/' directory separator",
            "physical_nested_file": (
                "stored path has at least one '/' directory separator; this is "
                "a path property and not necessarily a non-root instruction scope"
            ),
            "placement_form": (
                "direct D/CLAUDE.md or case-insensitive hidden "
                "D/.claude/CLAUDE.md candidate form; exact-case counts retained"
            ),
            "logical_scope": (
                "D for either D/CLAUDE.md or D/.claude/CLAUDE.md; an "
                "operational mapping from documented locations, not an observed load"
            ),
            "directory_depth": "number of POSIX path components before the basename",
            "nearest_ancestor": (
                "closest strict ancestor directory that also contains an exact "
                "CLAUDE.md; all files at an ambiguous nearest scope produce edges"
            ),
            "instruction_scope_chain": (
                "number of distinct CLAUDE.md-containing directory scopes along "
                "the nearest-ancestor chain, including the current scope"
            ),
            "normalized_line": (
                "non-empty line after Unicode NFKC normalization, trimming, "
                "internal-whitespace collapse, and case folding; duplicates form a set"
            ),
            "normalized_equivalent": (
                "different content hashes but identical non-empty normalized-line sets"
            ),
            "high_jaccard": (
                "different content hashes, required shared-line minimum, and "
                "normalized-line-set Jaccard similarity at or above the threshold"
            ),
            "high_containment": (
                "different content hashes, required shared-line minimum, below the "
                "Jaccard threshold, and intersection/minimum-set-size at or above "
                "the containment threshold"
            ),
            "quantiles": "empirical nearest-rank",
        },
        "population": {
            "files": exact_files,
            "repositories": len(repositories),
            "root_files": root_files,
            "root_file_share": _share(root_files, exact_files),
            "nested_files": nested_files,
            "nested_file_share": _share(nested_files, exact_files),
            "repositories_with_root_files": sum(
                repository.root_files > 0 for repository in repositories
            ),
            "repositories_with_nested_files": repositories_with_nested,
            "single_file_repositories": len(repositories) - len(multi_repositories),
            "multi_file_repositories": len(multi_repositories),
            "multi_file_repository_share": _share(
                len(multi_repositories),
                len(repositories),
            ),
            "files_in_multi_file_repositories": sum(
                repository.exact_files for repository in multi_repositories
            ),
            "files_in_multi_file_repository_share": _share(
                sum(repository.exact_files for repository in multi_repositories),
                exact_files,
            ),
            "null_content_hash_files": sum(
                file.content_hash is None
                for files in files_by_repository.values()
                for file in files
            ),
            "missing_content_files_in_overlap_population": missing_content_files,
        },
        "documented_scope_mapping": _documented_scope_mapping(
            files_by_repository
        ),
        "location_configurations": configurations,
        "directory_depth": {
            "distribution": _integer_distribution(depth_counts.elements()),
            "nested_file_distribution": _integer_distribution(
                depth
                for depth, count in depth_counts.items()
                if depth > 0
                for _ in range(count)
            ),
            "counts": [
                {
                    "directory_depth": depth,
                    "files": count,
                    "file_share": _share(count, exact_files),
                }
                for depth, count in sorted(depth_counts.items())
            ],
        },
        "repository_multiplicity": {
            "distribution": _integer_distribution(multiplicity_counts),
            "bins": _multiplicity_bins(multiplicity_counts),
            "top_repositories": [
                {
                    "repo_full_name": repository.repo_full_name,
                    "exact_files": repository.exact_files,
                    "root_files": repository.root_files,
                    "nested_files": repository.nested_files,
                    "maximum_directory_depth": repository.maximum_directory_depth,
                    "location_configuration": repository.location_configuration,
                }
                for repository in sorted(
                    repositories,
                    key=lambda item: (
                        -item.exact_files,
                        *_repository_sort_key(item.repo_full_name),
                    ),
                )[:20]
            ],
        },
        "path_components": {
            "casefolded": True,
            "distinct_first_directories": len(
                component_statistics["first_directory"]
            ),
            "distinct_immediate_parents": len(
                component_statistics["immediate_parent"]
            ),
            "distinct_directory_components": len(
                component_statistics["any_directory"]
            ),
            "top_20": top_components,
        },
        "hierarchy": {
            "nearest_ancestor_edges": len(hierarchy_edges),
            "root_parent_edges": sum(
                edge.parent_directory_depth == 0 for edge in hierarchy_edges
            ),
            "nested_parent_edges": sum(
                edge.parent_directory_depth > 0 for edge in hierarchy_edges
            ),
            "repositories_with_nearest_ancestor_edges": len(
                hierarchy_repositories
            ),
            "repository_share_with_nearest_ancestor_edges": _share(
                len(hierarchy_repositories),
                len(repositories),
            ),
            "multi_file_repositories_with_nearest_ancestor_edges": len(
                hierarchy_repositories
            ),
            "multi_file_repository_share_with_nearest_ancestor_edges": _share(
                len(hierarchy_repositories),
                len(multi_repositories),
            ),
            "multi_file_repositories_without_ancestor_relation": (
                len(multi_repositories) - len(hierarchy_repositories)
            ),
            "child_files_with_nearest_ancestor": sum(
                repository.files_with_nearest_ancestor
                for repository in repositories
            ),
            "child_file_share_of_nested_files": _share(
                sum(
                    repository.files_with_nearest_ancestor
                    for repository in repositories
                ),
                nested_files,
            ),
            "files_serving_as_nearest_ancestor": sum(
                repository.files_serving_as_nearest_ancestor
                for repository in repositories
            ),
            "same_directory_file_groups": sum(
                repository.same_directory_file_groups
                for repository in repositories
            ),
            "repositories_with_same_directory_file_groups": sum(
                repository.same_directory_file_groups > 0
                for repository in repositories
            ),
            "directory_distance": _integer_distribution(
                edge.directory_distance for edge in hierarchy_edges
            ),
            "maximum_instruction_scope_chain": _integer_distribution(
                repository.maximum_instruction_scope_chain
                for repository in repositories
            ),
            "maximum_observed_instruction_scope_chain": max(
                (
                    repository.maximum_instruction_scope_chain
                    for repository in repositories
                ),
                default=0,
            ),
            "instruction_scope_chain_counts": [
                {
                    "maximum_chain": chain,
                    "repositories": count,
                    "repository_share": _share(count, len(repositories)),
                }
                for chain, count in sorted(
                    Counter(
                        repository.maximum_instruction_scope_chain
                        for repository in repositories
                    ).items()
                )
            ],
            "repositories_with_edges_by_location_configuration": {
                configuration: sum(
                    repository.location_configuration == configuration
                    for repository in hierarchy_repositories
                )
                for configuration in (
                    "root_only",
                    "nested_only",
                    "root_and_nested",
                )
            },
        },
        "within_repository_exact_reuse": {
            "repositories_with_exact_duplicate_groups": len(
                exact_reuse_repositories
            ),
            "repository_share_with_exact_duplicate_groups": _share(
                len(exact_reuse_repositories),
                len(repositories),
            ),
            "multi_file_repository_share_with_exact_duplicate_groups": _share(
                len(exact_reuse_repositories),
                len(multi_repositories),
            ),
            "exact_duplicate_groups": sum(
                repository.exact_duplicate_groups for repository in repositories
            ),
            "files_in_exact_duplicate_groups": sum(
                repository.files_in_exact_duplicate_groups
                for repository in repositories
            ),
            "file_share_in_exact_duplicate_groups": _share(
                sum(
                    repository.files_in_exact_duplicate_groups
                    for repository in repositories
                ),
                exact_files,
            ),
            "multi_file_population_share_in_exact_duplicate_groups": _share(
                sum(
                    repository.files_in_exact_duplicate_groups
                    for repository in repositories
                ),
                sum(
                    repository.exact_files for repository in multi_repositories
                ),
            ),
            "repeated_exact_instances": sum(
                repository.repeated_exact_instances
                for repository in repositories
            ),
            "repeated_exact_instance_share": _share(
                sum(
                    repository.repeated_exact_instances
                    for repository in repositories
                ),
                exact_files,
            ),
            "exact_duplicate_pairs": sum(
                repository.exact_duplicate_pairs for repository in repositories
            ),
        },
        "within_repository_normalized_line_overlap": {
            "candidate_file_pairs": candidate_pairs,
            "evaluated_different_hash_pairs": evaluated_nonexact_pairs,
            "reported_pairs": len(overlap_pairs),
            "repositories_with_reported_pairs": len(
                nonexact_overlap_repositories
            ),
            "multi_file_repository_share_with_reported_pairs": _share(
                len(nonexact_overlap_repositories),
                len(multi_repositories),
            ),
            "files_in_reported_pairs": sum(
                repository.files_in_nonexact_overlap_pairs
                for repository in repositories
            ),
            "multi_file_population_share_in_reported_pairs": _share(
                sum(
                    repository.files_in_nonexact_overlap_pairs
                    for repository in repositories
                ),
                sum(
                    repository.exact_files for repository in multi_repositories
                ),
            ),
            "pair_counts_by_relation": {
                relation: overlap_relation_counts[relation]
                for relation in (
                    "normalized_equivalent",
                    "high_jaccard",
                    "high_containment",
                )
            },
            "repository_counts_by_relation": overlap_relation_repositories,
            "interpretation": (
                "exploratory lexical-overlap indicators; not validated semantic "
                "classifications of duplication, inheritance, or override behavior"
            ),
        },
        "path_quality": {
            "anomalies": dict(sorted(path_anomalies.items())),
            "basename_casing": [
                {"basename": basename, "files": count}
                for basename, count in sorted(
                    basename_casing.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
    }
    return (
        summary,
        repositories,
        component_rows,
        hierarchy_edges,
        overlap_pairs,
    )


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
            writer.writerow(data)
    return output


def write_repository_csv(
    repositories: Sequence[RepositoryStructure],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        repositories,
        output_path,
        fields=tuple(RepositoryStructure.__dataclass_fields__),
    )


def write_component_csv(
    rows: Sequence[dict[str, int | float | str]],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dimension",
        "rank",
        "component_casefolded",
        "file_count",
        "file_share_of_nested_files",
        "repository_count",
        "repository_share_of_repositories_with_nested_files",
        "occurrence_count",
        "example_paths",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in formatted.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.6f}"
            writer.writerow(formatted)
    return output


def write_hierarchy_csv(
    edges: Sequence[HierarchyEdge],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        edges,
        output_path,
        fields=tuple(HierarchyEdge.__dataclass_fields__),
    )


def write_overlap_csv(
    pairs: Sequence[LineOverlapPair],
    output_path: str | Path,
) -> Path:
    return _write_dataclass_csv(
        pairs,
        output_path,
        fields=tuple(LineOverlapPair.__dataclass_fields__),
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
    parser = argparse.ArgumentParser(
        description=(
            "Read mined.db without modifying it and export repository-level "
            "placement statistics for exact CLAUDE.md files."
        )
    )
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--repositories-output",
        default="claude_repository_structure.csv",
        help="one-row-per-repository CSV output",
    )
    parser.add_argument(
        "--components-output",
        default="claude_path_components.csv",
        help="ranked directory-component CSV output",
    )
    parser.add_argument(
        "--hierarchy-output",
        default="claude_hierarchy_edges.csv",
        help="nearest-ancestor edge CSV output",
    )
    parser.add_argument(
        "--overlap-output",
        default="claude_intra_repo_line_overlap.csv",
        help="reported non-exact normalized-line-overlap pairs",
    )
    parser.add_argument(
        "--summary-output",
        default="claude_repository_structure_summary.json",
        help="machine-readable JSON summary output",
    )
    parser.add_argument(
        "--minimum-shared-lines",
        type=positive_int,
        default=5,
        help="minimum shared normalized lines for thresholded pairs (default: 5)",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=unit_interval,
        default=0.80,
        help="high-Jaccard threshold (default: 0.80)",
    )
    parser.add_argument(
        "--containment-threshold",
        type=unit_interval,
        default=0.90,
        help="smaller-document containment threshold (default: 0.90)",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=10,
        help="number of leading repositories to print (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, repositories, components, edges, pairs = (
            analyze_repository_structure(
                args.db,
                minimum_shared_lines=args.minimum_shared_lines,
                jaccard_threshold=args.jaccard_threshold,
                containment_threshold=args.containment_threshold,
            )
        )
        repository_path = write_repository_csv(
            repositories,
            args.repositories_output,
        )
        component_path = write_component_csv(components, args.components_output)
        hierarchy_path = write_hierarchy_csv(edges, args.hierarchy_output)
        overlap_path = write_overlap_csv(pairs, args.overlap_output)
        summary_path = write_summary_json(summary, args.summary_output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    population = summary["population"]
    hierarchy = summary["hierarchy"]
    exact_reuse = summary["within_repository_exact_reuse"]
    line_overlap = summary["within_repository_normalized_line_overlap"]
    assert isinstance(population, dict)
    assert isinstance(hierarchy, dict)
    assert isinstance(exact_reuse, dict)
    assert isinstance(line_overlap, dict)

    print(f"Scope: exact-claude ({EXACT_CLAUDE_SCOPE})")
    print(
        f"Files: {population['files']:,}; repositories: "
        f"{population['repositories']:,}"
    )
    print(
        f"Root / nested files: {population['root_files']:,} / "
        f"{population['nested_files']:,}"
    )
    print(f"Multi-file repositories: {population['multi_file_repositories']:,}")
    print(
        "Repositories with nearest-ancestor edges: "
        f"{hierarchy['repositories_with_nearest_ancestor_edges']:,}"
    )
    print(
        "Repositories with within-repository exact reuse: "
        f"{exact_reuse['repositories_with_exact_duplicate_groups']:,}"
    )
    print(f"Reported non-exact overlap pairs: {line_overlap['reported_pairs']:,}")
    top_repositories = summary["repository_multiplicity"]["top_repositories"]  # type: ignore[index]
    print("Top repositories by observed exact CLAUDE.md count:")
    for rank, item in enumerate(top_repositories[: args.top], 1):
        print(
            f"  {rank:>3}. {item['exact_files']:>4,}  "
            f"{item['repo_full_name']}"
        )
    print(f"Repositories CSV: {repository_path.resolve()}")
    print(f"Path components CSV: {component_path.resolve()}")
    print(f"Hierarchy edges CSV: {hierarchy_path.resolve()}")
    print(f"Line overlap CSV: {overlap_path.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
