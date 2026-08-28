#!/usr/bin/env python3
"""Analyze documentation-derived Claude ``@path`` import syntax.

Phase 2r is intentionally separate from the frozen Markdown-reference Phase 2
analysis.  It scans every exact case-insensitive ``CLAUDE.md`` file, preserves
active and excluded ``@path`` evidence, resolves relative targets lexically,
and optionally compares occurrences with the frozen Phase 2 high-confidence
reference artifact.

The source SQLite database is opened read-only with ``mode=ro&immutable=1`` and
``PRAGMA query_only=ON``.  Detected tokens are syntax candidates; target
existence, external-import approval, recursive expansion, and runtime loading
are not established by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miner.claude_imports import (  # noqa: E402
    DETECTOR_VERSION,
    DOCUMENTATION_ACCESSED,
    DOCUMENTATION_URL,
    MAX_DOCUMENTED_IMPORT_HOPS,
    ClaudeImportOccurrence,
    extract_claude_import_occurrences,
    resolve_import_target,
    target_extension_class,
    target_shape as classify_target_shape,
)


POPULATION_DEFINITION = (
    "files whose stored POSIX path has a case-insensitive basename equal to "
    "CLAUDE.md"
)
ACTIVE_DECISION = "import_candidate"


@dataclass(frozen=True)
class Phase2Record:
    reference_id: str
    repo_full_name: str
    source_path: str
    line_number: int
    intent_category: str
    syntax: str
    rule_id: str
    normalized_target: str
    target_basename: str


@dataclass
class Phase2Index:
    by_key: dict[tuple[str, str, int, str], deque[Phase2Record]]
    totals_by_intent: Counter[str]
    totals_by_file_intent: dict[tuple[str, str], Counter[str]]
    input_path: str
    input_sha256: str

    def pop_match(
        self,
        repo_full_name: str,
        source_path: str,
        line_number: int,
        normalized_target: str,
    ) -> Phase2Record | None:
        key = (repo_full_name, source_path, line_number, normalized_target)
        records = self.by_key.get(key)
        if not records:
            return None
        record = records.popleft()
        if not records:
            del self.by_key[key]
        return record

    def unmatched_records(self) -> list[Phase2Record]:
        return [record for records in self.by_key.values() for record in records]


@dataclass(frozen=True)
class ImportRow:
    repo_full_name: str
    source_path: str
    source_physical_location: str
    source_logical_scope: str
    source_directory_depth: int
    source_content_hash: str
    source_size_bytes: int | None
    repository_exact_files: int
    repository_stars: int | None
    repository_language: str
    line_number: int
    column_number: int
    end_column_number: int
    context_kind: str
    decision: str
    rule_id: str
    surface_form: str
    raw_token: str
    raw_target: str
    normalized_target: str
    resolved_target: str
    path_kind: str
    path_relation: str
    target_basename: str
    target_extension_class: str
    target_shape: str
    is_self_reference: int
    phase2_reference_id: str
    phase2_intent_category: str
    phase2_syntax: str
    phase2_rule_id: str
    evidence_context: str


@dataclass(frozen=True)
class FileRow:
    repo_full_name: str
    source_path: str
    source_physical_location: str
    source_logical_scope: str
    source_directory_depth: int
    content_hash: str
    size_bytes: int | None
    repository_stars: int | None
    repository_language: str
    repository_exact_files: int
    repository_multiplicity: str
    global_content_copies: int
    repository_content_copies: int
    global_exact_duplicate: int
    within_repository_exact_duplicate: int
    detected_at_path_occurrences: int
    import_candidate_occurrences: int
    distinct_import_candidate_targets: int
    markdown_candidate_occurrences: int
    other_extension_candidate_occurrences: int
    extensionless_candidate_occurrences: int
    bare_extensionless_candidate_occurrences: int
    path_extensionless_candidate_occurrences: int
    dotfile_extensionless_candidate_occurrences: int
    repository_internal_candidate_occurrences: int
    outside_repository_candidate_occurrences: int
    external_candidate_occurrences: int
    platform_dependent_candidate_occurrences: int
    self_reference_candidate_occurrences: int
    whole_line_candidate_occurrences: int
    list_item_candidate_occurrences: int
    standalone_formatted_candidate_occurrences: int
    prose_embedded_candidate_occurrences: int
    excluded_inline_code_occurrences: int
    excluded_fenced_code_occurrences: int
    excluded_html_comment_occurrences: int
    excluded_unsupported_uri_occurrences: int
    excluded_unsupported_pattern_occurrences: int
    excluded_unsupported_variable_occurrences: int
    excluded_non_file_target_occurrences: int
    phase2_high_confidence_occurrences: int
    phase2_direct_inclusion_occurrences: int
    phase2_instructional_delegation_occurrences: int
    phase2_high_confidence_matches: int
    candidate_target_extension_classes: str
    candidate_path_kinds: str


@dataclass(frozen=True)
class TargetRow:
    decision: str
    context_kind: str
    path_kind: str
    path_relation: str
    target_extension_class: str
    target_shape: str
    target_basename: str
    occurrences: int
    files: int
    repositories: int


@dataclass(frozen=True)
class StratumRow:
    dimension: str
    stratum: str
    files: int
    repositories: int
    files_with_import_candidate: int
    repositories_with_import_candidate: int
    import_candidate_file_share: float
    import_candidate_occurrences: int
    files_with_explicit_path_candidate: int
    repositories_with_explicit_path_candidate: int
    explicit_path_candidate_file_share: float
    explicit_path_candidate_occurrences: int
    ambiguous_bare_extensionless_occurrences: int
    markdown_candidate_occurrences: int
    other_extension_candidate_occurrences: int
    extensionless_candidate_occurrences: int


@dataclass(frozen=True)
class ComparisonRow:
    transition: str
    target_extension_class: str
    occurrences: int


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
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _integer_distribution(values: Iterable[int]) -> dict[str, int | float]:
    materialized = list(values)
    if not materialized:
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
        "count": len(materialized),
        "minimum": min(materialized),
        "p25": _nearest_rank(materialized, 0.25),
        "median": _nearest_rank(materialized, 0.50),
        "p75": _nearest_rank(materialized, 0.75),
        "p90": _nearest_rank(materialized, 0.90),
        "p95": _nearest_rank(materialized, 0.95),
        "p99": _nearest_rank(materialized, 0.99),
        "maximum": max(materialized),
        "mean": sum(materialized) / len(materialized),
    }


def _source_physical_location(path: str) -> str:
    return "literal_root" if "/" not in path else "physically_nested"


def _source_logical_scope(path: str) -> str:
    directory_parts = list(PurePosixPath(path).parts[:-1])
    if directory_parts and directory_parts[-1].casefold() == ".claude":
        directory_parts.pop()
    return "project_root" if not directory_parts else "non_root"


def _directory_depth(path: str) -> int:
    return max(0, len(PurePosixPath(path).parts) - 1)


def _size_band(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    if size_bytes <= 50:
        return "<=50"
    if size_bytes <= 200:
        return "51-200"
    if size_bytes <= 1_000:
        return "201-1000"
    if size_bytes <= 5_000:
        return "1001-5000"
    return ">5000"


def _star_band(stars: int | None) -> str:
    if stars is None:
        return "unknown"
    if stars == 0:
        return "0"
    if stars <= 9:
        return "1-9"
    if stars <= 99:
        return "10-99"
    if stars <= 999:
        return "100-999"
    return ">=1000"


def _load_population_counts(
    connection: sqlite3.Connection,
) -> tuple[dict[str, int], dict[str, int], dict[tuple[str, str], int]]:
    predicate = _exact_predicate(alias="f")
    repo_counts = {
        str(row["repo_full_name"]): int(row["copies"])
        for row in connection.execute(
            f"""SELECT f.repo_full_name, COUNT(*) AS copies
                FROM files AS f WHERE {predicate}
                GROUP BY f.repo_full_name"""
        )
    }
    hash_counts = {
        str(row["content_hash"]): int(row["copies"])
        for row in connection.execute(
            f"""SELECT f.content_hash, COUNT(*) AS copies
                FROM files AS f WHERE {predicate} AND f.content_hash IS NOT NULL
                GROUP BY f.content_hash"""
        )
    }
    repo_hash_counts = {
        (str(row["repo_full_name"]), str(row["content_hash"])): int(row["copies"])
        for row in connection.execute(
            f"""SELECT f.repo_full_name, f.content_hash, COUNT(*) AS copies
                FROM files AS f
                WHERE {predicate} AND f.content_hash IS NOT NULL
                GROUP BY f.repo_full_name, f.content_hash
                HAVING COUNT(*) > 1"""
        )
    }
    return repo_counts, hash_counts, repo_hash_counts


def load_phase2_index(path: str | Path | None) -> Phase2Index | None:
    if path is None:
        return None
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Phase 2 occurrence artifact not found: {input_path}")
    by_key: dict[tuple[str, str, int, str], deque[Phase2Record]] = defaultdict(deque)
    totals_by_intent: Counter[str] = Counter()
    totals_by_file_intent: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "reference_id",
            "repo_full_name",
            "source_path",
            "line_number",
            "intent_category",
            "syntax",
            "rule_id",
            "high_confidence_local_instructional",
            "normalized_target",
            "target_basename",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Phase 2 artifact is missing columns: {sorted(missing)}")
        for row in reader:
            if row["high_confidence_local_instructional"] != "1":
                continue
            record = Phase2Record(
                reference_id=row["reference_id"],
                repo_full_name=row["repo_full_name"],
                source_path=row["source_path"],
                line_number=int(row["line_number"]),
                intent_category=row["intent_category"],
                syntax=row["syntax"],
                rule_id=row["rule_id"],
                normalized_target=row["normalized_target"],
                target_basename=row["target_basename"],
            )
            key = (
                record.repo_full_name,
                record.source_path,
                record.line_number,
                record.normalized_target,
            )
            by_key[key].append(record)
            totals_by_intent[record.intent_category] += 1
            totals_by_file_intent[(record.repo_full_name, record.source_path)][
                record.intent_category
            ] += 1
    return Phase2Index(
        by_key=dict(by_key),
        totals_by_intent=totals_by_intent,
        totals_by_file_intent=dict(totals_by_file_intent),
        input_path=input_path.as_posix(),
        input_sha256=_sha256_file(input_path),
    )


def _candidate_target_key(row: ImportRow) -> tuple[str, str]:
    target = row.resolved_target or row.normalized_target
    return row.path_kind, target


def analyze_imports(
    db_path: str | Path,
    *,
    phase2_occurrences_path: str | Path | None = None,
) -> tuple[list[ImportRow], list[FileRow], int, Phase2Index | None]:
    """Analyze the exact population and return occurrence/file evidence."""
    phase2_index = load_phase2_index(phase2_occurrences_path)
    connection = _readonly_connection(db_path)
    repo_counts, hash_counts, repo_hash_counts = _load_population_counts(connection)
    occurrences: list[ImportRow] = []
    files: list[FileRow] = []
    predicate = _exact_predicate(alias="f")
    query = f"""
        SELECT f.id, f.repo_full_name, f.path, f.content_hash, f.size_bytes,
               f.content, r.stars, r.language
        FROM files AS f
        LEFT JOIN repos AS r ON r.full_name = f.repo_full_name
        WHERE {predicate}
        ORDER BY COALESCE(f.content_hash, printf('id:%020d', f.id)),
                 lower(f.repo_full_name), f.repo_full_name, lower(f.path), f.path
    """
    cached_key = ""
    cached_occurrences: tuple[ClaudeImportOccurrence, ...] = ()
    try:
        for row in connection.execute(query):
            repo = str(row["repo_full_name"])
            source_path = str(row["path"])
            content_hash = str(row["content_hash"] or "")
            cache_key = content_hash or f"id:{int(row['id'])}"
            if cache_key != cached_key:
                cached_occurrences = extract_claude_import_occurrences(
                    str(row["content"] or "")
                )
                cached_key = cache_key

            size_bytes = row["size_bytes"]
            stars = row["stars"]
            language = str(row["language"] or "Unknown")
            repo_exact_files = repo_counts[repo]
            physical_location = _source_physical_location(source_path)
            logical_scope = _source_logical_scope(source_path)
            depth = _directory_depth(source_path)
            local_rows: list[ImportRow] = []

            for detected in cached_occurrences:
                resolution = resolve_import_target(source_path, detected.raw_target)
                match = None
                if phase2_index is not None:
                    match = phase2_index.pop_match(
                        repo,
                        source_path,
                        detected.line_number,
                        resolution.normalized_target,
                    )
                local_rows.append(
                    ImportRow(
                        repo_full_name=repo,
                        source_path=source_path,
                        source_physical_location=physical_location,
                        source_logical_scope=logical_scope,
                        source_directory_depth=depth,
                        source_content_hash=content_hash,
                        source_size_bytes=size_bytes,
                        repository_exact_files=repo_exact_files,
                        repository_stars=stars,
                        repository_language=language,
                        line_number=detected.line_number,
                        column_number=detected.column_number,
                        end_column_number=detected.end_column_number,
                        context_kind=detected.context_kind,
                        decision=detected.decision,
                        rule_id=detected.rule_id,
                        surface_form=detected.surface_form,
                        raw_token=detected.raw_token,
                        raw_target=detected.raw_target,
                        normalized_target=resolution.normalized_target,
                        resolved_target=resolution.resolved_target,
                        path_kind=resolution.path_kind,
                        path_relation=resolution.path_relation,
                        target_basename=resolution.target_basename,
                        target_extension_class=resolution.target_extension_class,
                        target_shape=classify_target_shape(
                            detected.raw_target,
                            resolution.target_extension_class,
                        ),
                        is_self_reference=resolution.is_self_reference,
                        phase2_reference_id=match.reference_id if match else "",
                        phase2_intent_category=match.intent_category if match else "",
                        phase2_syntax=match.syntax if match else "",
                        phase2_rule_id=match.rule_id if match else "",
                        evidence_context=detected.evidence_context,
                    )
                )
            occurrences.extend(local_rows)

            candidates = [item for item in local_rows if item.decision == ACTIVE_DECISION]
            candidate_targets = {_candidate_target_key(item) for item in candidates}
            phase2_file_counts = (
                phase2_index.totals_by_file_intent.get((repo, source_path), Counter())
                if phase2_index is not None
                else Counter()
            )
            global_copies = hash_counts.get(content_hash, 1)
            repo_content_copies = repo_hash_counts.get((repo, content_hash), 1)
            files.append(
                FileRow(
                    repo_full_name=repo,
                    source_path=source_path,
                    source_physical_location=physical_location,
                    source_logical_scope=logical_scope,
                    source_directory_depth=depth,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    repository_stars=stars,
                    repository_language=language,
                    repository_exact_files=repo_exact_files,
                    repository_multiplicity=(
                        "multiple" if repo_exact_files > 1 else "single"
                    ),
                    global_content_copies=global_copies,
                    repository_content_copies=repo_content_copies,
                    global_exact_duplicate=int(global_copies > 1),
                    within_repository_exact_duplicate=int(repo_content_copies > 1),
                    detected_at_path_occurrences=len(local_rows),
                    import_candidate_occurrences=len(candidates),
                    distinct_import_candidate_targets=len(candidate_targets),
                    markdown_candidate_occurrences=sum(
                        item.target_extension_class == "markdown" for item in candidates
                    ),
                    other_extension_candidate_occurrences=sum(
                        item.target_extension_class == "other_extension"
                        for item in candidates
                    ),
                    extensionless_candidate_occurrences=sum(
                        item.target_extension_class == "extensionless"
                        for item in candidates
                    ),
                    bare_extensionless_candidate_occurrences=sum(
                        item.target_shape == "bare_extensionless" for item in candidates
                    ),
                    path_extensionless_candidate_occurrences=sum(
                        item.target_shape == "path_extensionless" for item in candidates
                    ),
                    dotfile_extensionless_candidate_occurrences=sum(
                        item.target_shape == "dotfile_extensionless" for item in candidates
                    ),
                    repository_internal_candidate_occurrences=sum(
                        item.path_relation
                        in {
                            "same_directory",
                            "ancestor_directory",
                            "descendant_directory",
                            "other_directory",
                        }
                        for item in candidates
                    ),
                    outside_repository_candidate_occurrences=sum(
                        item.path_relation == "outside_repository" for item in candidates
                    ),
                    external_candidate_occurrences=sum(
                        item.path_relation in {"external_home", "external_absolute"}
                        for item in candidates
                    ),
                    platform_dependent_candidate_occurrences=sum(
                        item.path_relation == "platform_dependent" for item in candidates
                    ),
                    self_reference_candidate_occurrences=sum(
                        item.is_self_reference for item in candidates
                    ),
                    whole_line_candidate_occurrences=sum(
                        item.surface_form == "whole_line" for item in candidates
                    ),
                    list_item_candidate_occurrences=sum(
                        item.surface_form == "list_item_only" for item in candidates
                    ),
                    standalone_formatted_candidate_occurrences=sum(
                        item.surface_form == "standalone_formatted" for item in candidates
                    ),
                    prose_embedded_candidate_occurrences=sum(
                        item.surface_form == "prose_embedded" for item in candidates
                    ),
                    excluded_inline_code_occurrences=sum(
                        item.decision == "excluded_inline_code" for item in local_rows
                    ),
                    excluded_fenced_code_occurrences=sum(
                        item.decision == "excluded_fenced_code" for item in local_rows
                    ),
                    excluded_html_comment_occurrences=sum(
                        item.decision == "excluded_html_comment" for item in local_rows
                    ),
                    excluded_unsupported_uri_occurrences=sum(
                        item.decision == "excluded_unsupported_uri" for item in local_rows
                    ),
                    excluded_unsupported_pattern_occurrences=sum(
                        item.decision == "excluded_unsupported_pattern"
                        for item in local_rows
                    ),
                    excluded_unsupported_variable_occurrences=sum(
                        item.decision == "excluded_unsupported_variable"
                        for item in local_rows
                    ),
                    excluded_non_file_target_occurrences=sum(
                        item.decision == "excluded_non_file_target" for item in local_rows
                    ),
                    phase2_high_confidence_occurrences=sum(phase2_file_counts.values()),
                    phase2_direct_inclusion_occurrences=phase2_file_counts.get(
                        "direct_inclusion", 0
                    ),
                    phase2_instructional_delegation_occurrences=phase2_file_counts.get(
                        "instructional_delegation", 0
                    ),
                    phase2_high_confidence_matches=sum(
                        bool(item.phase2_reference_id) for item in local_rows
                    ),
                    candidate_target_extension_classes=";".join(
                        sorted({item.target_extension_class for item in candidates})
                    ),
                    candidate_path_kinds=";".join(
                        sorted({item.path_kind for item in candidates})
                    ),
                )
            )
        database_changes = connection.total_changes
    finally:
        connection.close()

    occurrences.sort(
        key=lambda item: (
            item.repo_full_name.casefold(),
            item.repo_full_name,
            item.source_path.casefold(),
            item.source_path,
            item.line_number,
            item.column_number,
            item.decision,
            item.raw_target.casefold(),
            item.raw_target,
        )
    )
    files.sort(
        key=lambda item: (
            item.repo_full_name.casefold(),
            item.repo_full_name,
            item.source_path.casefold(),
            item.source_path,
        )
    )
    return occurrences, files, database_changes, phase2_index


def aggregate_targets(rows: Sequence[ImportRow]) -> list[TargetRow]:
    groups: dict[tuple[str, str, str, str, str, str, str], list[ImportRow]] = defaultdict(list)
    for row in rows:
        key = (
            row.decision,
            row.context_kind,
            row.path_kind,
            row.path_relation,
            row.target_extension_class,
            row.target_shape,
            row.target_basename.casefold(),
        )
        groups[key].append(row)
    targets = [
        TargetRow(
            decision=key[0],
            context_kind=key[1],
            path_kind=key[2],
            path_relation=key[3],
            target_extension_class=key[4],
            target_shape=key[5],
            target_basename=members[0].target_basename,
            occurrences=len(members),
            files=len({(item.repo_full_name, item.source_path) for item in members}),
            repositories=len({item.repo_full_name for item in members}),
        )
        for key, members in groups.items()
    ]
    targets.sort(
        key=lambda item: (
            0 if item.decision == ACTIVE_DECISION else 1,
            -item.occurrences,
            item.target_basename.casefold(),
            item.path_kind,
            item.path_relation,
        )
    )
    return targets


def _stratum_values(file: FileRow) -> dict[str, str]:
    return {
        "physical_location": file.source_physical_location,
        "logical_scope": file.source_logical_scope,
        "repository_multiplicity": file.repository_multiplicity,
        "global_exact_duplicate": "yes" if file.global_exact_duplicate else "no",
        "within_repository_exact_duplicate": (
            "yes" if file.within_repository_exact_duplicate else "no"
        ),
        "file_size_bytes": _size_band(file.size_bytes),
        "repository_stars": _star_band(file.repository_stars),
        "repository_primary_language": file.repository_language,
    }


def aggregate_strata(files: Sequence[FileRow]) -> list[StratumRow]:
    groups: dict[tuple[str, str], list[FileRow]] = defaultdict(list)
    for file in files:
        for dimension, stratum in _stratum_values(file).items():
            groups[(dimension, stratum)].append(file)
    rows: list[StratumRow] = []
    for (dimension, stratum), members in groups.items():
        candidate_members = [item for item in members if item.import_candidate_occurrences]
        explicit_members = [
            item
            for item in members
            if item.import_candidate_occurrences
            > item.bare_extensionless_candidate_occurrences
        ]
        repositories = {item.repo_full_name for item in members}
        candidate_repositories = {item.repo_full_name for item in candidate_members}
        explicit_repositories = {item.repo_full_name for item in explicit_members}
        rows.append(
            StratumRow(
                dimension=dimension,
                stratum=stratum,
                files=len(members),
                repositories=len(repositories),
                files_with_import_candidate=len(candidate_members),
                repositories_with_import_candidate=len(candidate_repositories),
                import_candidate_file_share=_share(len(candidate_members), len(members)),
                import_candidate_occurrences=sum(
                    item.import_candidate_occurrences for item in members
                ),
                files_with_explicit_path_candidate=len(explicit_members),
                repositories_with_explicit_path_candidate=len(explicit_repositories),
                explicit_path_candidate_file_share=_share(
                    len(explicit_members), len(members)
                ),
                explicit_path_candidate_occurrences=sum(
                    item.import_candidate_occurrences
                    - item.bare_extensionless_candidate_occurrences
                    for item in members
                ),
                ambiguous_bare_extensionless_occurrences=sum(
                    item.bare_extensionless_candidate_occurrences for item in members
                ),
                markdown_candidate_occurrences=sum(
                    item.markdown_candidate_occurrences for item in members
                ),
                other_extension_candidate_occurrences=sum(
                    item.other_extension_candidate_occurrences for item in members
                ),
                extensionless_candidate_occurrences=sum(
                    item.extensionless_candidate_occurrences for item in members
                ),
            )
        )
    rows.sort(key=lambda item: (item.dimension, item.stratum.casefold(), item.stratum))
    return rows


def aggregate_comparison(
    rows: Sequence[ImportRow],
    phase2_index: Phase2Index | None,
) -> list[ComparisonRow]:
    if phase2_index is None:
        return []
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.decision == ACTIVE_DECISION:
            if row.phase2_intent_category == "direct_inclusion":
                transition = "phase2_direct_retained_as_import_candidate"
            elif row.phase2_intent_category == "instructional_delegation":
                transition = "phase2_delegation_also_import_candidate"
            else:
                transition = "import_candidate_without_phase2_high_confidence_match"
        elif row.phase2_intent_category == "direct_inclusion":
            transition = f"phase2_direct_reclassified_{row.decision}"
        elif row.phase2_intent_category == "instructional_delegation":
            transition = f"phase2_delegation_reclassified_{row.decision}"
        else:
            transition = f"excluded_token_without_phase2_high_confidence_match_{row.decision}"
        counts[(transition, row.target_extension_class)] += 1

    for record in phase2_index.unmatched_records():
        extension_class = target_extension_class(record.target_basename)
        counts[(f"phase2_{record.intent_category}_without_phase2r_token", extension_class)] += 1

    result = [
        ComparisonRow(
            transition=transition,
            target_extension_class=extension_class,
            occurrences=count,
        )
        for (transition, extension_class), count in counts.items()
    ]
    result.sort(
        key=lambda item: (
            item.transition,
            item.target_extension_class,
        )
    )
    return result


def build_summary(
    rows: Sequence[ImportRow],
    files: Sequence[FileRow],
    targets: Sequence[TargetRow],
    strata: Sequence[StratumRow],
    comparison: Sequence[ComparisonRow],
    *,
    phase2_index: Phase2Index | None,
    database_path: str | Path,
    database_sha256_before: str,
    database_sha256_after: str,
    sqlite_total_changes: int,
) -> dict[str, object]:
    candidates = [item for item in rows if item.decision == ACTIVE_DECISION]
    candidate_files = {
        (item.repo_full_name, item.source_path) for item in candidates
    }
    candidate_repositories = {item.repo_full_name for item in candidates}
    candidate_edges = {
        (
            item.repo_full_name,
            item.source_path,
            item.path_kind,
            item.resolved_target or item.normalized_target,
        )
        for item in candidates
    }
    decision_counts = Counter(item.decision for item in rows)
    context_counts = Counter(item.context_kind for item in rows)
    extension_counts = Counter(item.target_extension_class for item in candidates)
    target_shape_counts = Counter(item.target_shape for item in candidates)
    path_kind_counts = Counter(item.path_kind for item in candidates)
    relation_counts = Counter(item.path_relation for item in candidates)
    surface_counts = Counter(item.surface_form for item in candidates)
    basename_counts = Counter(item.target_basename.casefold() for item in candidates)
    files_with_any = {(item.repo_full_name, item.source_path) for item in rows}
    repositories = {item.repo_full_name for item in files}
    comparison_totals: Counter[str] = Counter()
    for row in comparison:
        comparison_totals[row.transition] += row.occurrences

    internal_candidates = [
        item
        for item in candidates
        if item.path_relation
        in {
            "same_directory",
            "ancestor_directory",
            "descendant_directory",
            "other_directory",
        }
        and not item.is_self_reference
    ]
    external_candidates = [
        item
        for item in candidates
        if item.path_relation in {"outside_repository", "external_home", "external_absolute"}
    ]
    candidate_file_counts = [
        item.import_candidate_occurrences
        for item in files
        if item.import_candidate_occurrences
    ]
    bare_extensionless_candidates = [
        item for item in candidates if item.target_shape == "bare_extensionless"
    ]
    explicit_path_shape_candidates = [
        item for item in candidates if item.target_shape != "bare_extensionless"
    ]
    explicit_path_files = {
        (item.repo_full_name, item.source_path) for item in explicit_path_shape_candidates
    }
    explicit_path_repositories = {
        item.repo_full_name for item in explicit_path_shape_candidates
    }
    phase2_metadata: dict[str, object]
    if phase2_index is None:
        phase2_metadata = {"comparison_enabled": False}
    else:
        unmatched_phase2 = phase2_index.unmatched_records()
        unmatched_by_intent_syntax = Counter(
            f"{item.intent_category}|{item.syntax}" for item in unmatched_phase2
        )
        phase2_metadata = {
            "comparison_enabled": True,
            "input_path": phase2_index.input_path,
            "input_sha256": phase2_index.input_sha256,
            "high_confidence_input_counts_by_intent": dict(
                sorted(phase2_index.totals_by_intent.items())
            ),
            "transition_counts": dict(sorted(comparison_totals.items())),
            "unmatched_phase2_high_confidence_occurrences": len(
                unmatched_phase2
            ),
            "unmatched_counts_by_intent_and_syntax": dict(
                sorted(unmatched_by_intent_syntax.items())
            ),
        }

    summary: dict[str, object] = {
        "analysis": "Claude @path import-syntax candidates (Phase 2r)",
        "detector_version": DETECTOR_VERSION,
        "population_definition": POPULATION_DEFINITION,
        "documentation": {
            "url": DOCUMENTATION_URL,
            "accessed": DOCUMENTATION_ACCESSED,
            "documented_max_recursive_hops": MAX_DOCUMENTED_IMPORT_HOPS,
        },
        "database": {
            "path": Path(database_path).as_posix(),
            "sha256_before": database_sha256_before,
            "sha256_after": database_sha256_after,
            "sha256_unchanged": database_sha256_before == database_sha256_after,
            "sqlite_total_changes": sqlite_total_changes,
            "sqlite_library_version": sqlite3.sqlite_version,
        },
        "population": {
            "files": len(files),
            "repositories": len(repositories),
            "files_with_any_detected_at_path_token": len(files_with_any),
            "files_with_any_detected_at_path_token_share": _share(
                len(files_with_any), len(files)
            ),
        },
        "import_syntax_candidates": {
            "occurrences": len(candidates),
            "distinct_source_target_edges": len(candidate_edges),
            "files": len(candidate_files),
            "file_share": _share(len(candidate_files), len(files)),
            "repositories": len(candidate_repositories),
            "repository_share": _share(len(candidate_repositories), len(repositories)),
            "explicit_path_shape_occurrences": len(explicit_path_shape_candidates),
            "explicit_path_shape_files": len(explicit_path_files),
            "explicit_path_shape_file_share": _share(
                len(explicit_path_files), len(files)
            ),
            "explicit_path_shape_repositories": len(explicit_path_repositories),
            "explicit_path_shape_repository_share": _share(
                len(explicit_path_repositories), len(repositories)
            ),
            "ambiguous_bare_extensionless_occurrences": len(
                bare_extensionless_candidates
            ),
            "repository_internal_nonself_occurrences": len(internal_candidates),
            "external_or_repository_escape_occurrences": len(external_candidates),
            "self_reference_occurrences": sum(item.is_self_reference for item in candidates),
            "occurrences_per_candidate_file": _integer_distribution(candidate_file_counts),
            "target_extension_class_counts": dict(sorted(extension_counts.items())),
            "target_shape_counts": dict(sorted(target_shape_counts.items())),
            "path_kind_counts": dict(sorted(path_kind_counts.items())),
            "path_relation_counts": dict(sorted(relation_counts.items())),
            "surface_form_counts": dict(sorted(surface_counts.items())),
            "leading_target_basenames": [
                {"target_basename": basename, "occurrences": count}
                for basename, count in basename_counts.most_common(25)
            ],
        },
        "all_detected_tokens": {
            "occurrences": len(rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "context_counts": dict(sorted(context_counts.items())),
        },
        "phase2_comparison": phase2_metadata,
        "artifact_rows": {
            "occurrences": len(rows),
            "files": len(files),
            "targets": len(targets),
            "strata": len(strata),
            "comparison": len(comparison),
        },
        "operational_rules": {
            "plain_text_at_path_tokens_are_candidates": True,
            "markdown_extension_required": False,
            "whole_line_required": False,
            "fenced_code_excluded": True,
            "inline_code_spans_excluded": True,
            "html_comment_spans_excluded": True,
            "inline_code_uses_equal_backtick_runs_and_may_span_nonblank_lines": True,
            "blank_lines_fences_and_block_comments_break_inline_code_pairing": True,
            "inline_code_precedes_inline_html_comment_classification": True,
            "uri_schemes_excluded": True,
            "undocumented_globs_and_template_patterns_excluded": True,
            "variable_interpolation_forms_excluded": True,
            "email_and_identifier_internal_at_signs_excluded_by_left_boundary": True,
            "relative_paths_resolve_against_containing_claude_file": True,
            "leading_posix_slash_is_filesystem_absolute_not_repository_root": True,
            "backslash_relative_paths_are_platform_dependent": True,
            "spaces_terminate_targets": True,
            "target_existence_checked": False,
            "recursive_expansion_performed": False,
            "runtime_loading_observed": False,
        },
        "limitations": [
            "The official documentation does not publish a complete character-level token grammar; this is a versioned operational detector.",
            "A syntactic candidate can be a social @mention, scoped package name, or nonexistent path until a revision-aligned tree is checked.",
            "The database does not contain complete repository trees or imported target content, so existence and recursive expansion are unverified.",
            "Absolute and home-relative imports may require external-import approval; approval state is not observable here.",
            "HTML comment spans are excluded operationally; runtime parser ordering should be verified with a versioned fixture before stronger claims.",
            "Backslash-relative paths are retained as platform-dependent rather than resolved against POSIX GitHub paths.",
        ],
    }
    return summary


def _write_dataclass_csv(
    rows: Sequence[object],
    output_path: str | Path,
    *,
    id_field: str | None = None,
    id_prefix: str = "",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot infer CSV columns for empty output: {output}")
    fieldnames = [field.name for field in fields(rows[0])]
    if id_field is not None:
        fieldnames = [id_field, *fieldnames]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            record = asdict(row)
            if id_field is not None:
                record = {id_field: f"{id_prefix}{index:08d}", **record}
            writer.writerow(record)
    return output


def write_occurrences_csv(rows: Sequence[ImportRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(
        rows,
        output_path,
        id_field="import_id",
        id_prefix="IMP-",
    )


def write_files_csv(rows: Sequence[FileRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_targets_csv(rows: Sequence[TargetRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_strata_csv(rows: Sequence[StratumRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_comparison_csv(
    rows: Sequence[ComparisonRow], output_path: str | Path
) -> Path:
    return _write_dataclass_csv(rows, output_path)


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
        "--phase2-occurrences",
        default="article/claude_markdown_reference_occurrences.csv",
        help="Frozen Phase 2 occurrence CSV, or an empty string to disable comparison.",
    )
    parser.add_argument(
        "--occurrences-output",
        default="article/claude_import_syntax_occurrences.csv",
    )
    parser.add_argument(
        "--files-output",
        default="article/claude_import_syntax_files.csv",
    )
    parser.add_argument(
        "--targets-output",
        default="article/claude_import_syntax_targets.csv",
    )
    parser.add_argument(
        "--strata-output",
        default="article/claude_import_syntax_strata.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="article/claude_import_phase2_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="article/claude_import_syntax_summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    phase2_path = arguments.phase2_occurrences or None
    database_hash_before = _sha256_file(arguments.db)
    rows, files, changes, phase2_index = analyze_imports(
        arguments.db,
        phase2_occurrences_path=phase2_path,
    )
    targets = aggregate_targets(rows)
    strata = aggregate_strata(files)
    comparison = aggregate_comparison(rows, phase2_index)
    database_hash_after = _sha256_file(arguments.db)
    summary = build_summary(
        rows,
        files,
        targets,
        strata,
        comparison,
        phase2_index=phase2_index,
        database_path=arguments.db,
        database_sha256_before=database_hash_before,
        database_sha256_after=database_hash_after,
        sqlite_total_changes=changes,
    )

    occurrence_path = write_occurrences_csv(rows, arguments.occurrences_output)
    file_path = write_files_csv(files, arguments.files_output)
    target_path = write_targets_csv(targets, arguments.targets_output)
    stratum_path = write_strata_csv(strata, arguments.strata_output)
    comparison_path = write_comparison_csv(comparison, arguments.comparison_output)
    summary_path = write_summary_json(summary, arguments.summary_output)
    print(
        f"Wrote {len(rows):,} detected tokens to {occurrence_path}, "
        f"{len(files):,} file rows to {file_path}, {len(targets):,} target rows "
        f"to {target_path}, {len(strata):,} strata to {stratum_path}, "
        f"{len(comparison):,} comparison rows to {comparison_path}, and "
        f"{summary_path}."
    )
    print(
        f"SQLite changes: {changes}; database SHA-256 unchanged: "
        f"{database_hash_before == database_hash_after}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
