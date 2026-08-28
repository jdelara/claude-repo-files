"""Extract candidate Markdown references from every exact ``CLAUDE.md``.

The script reads SQLite with ``mode=ro&immutable=1`` and ``query_only=ON``.
It preserves reference-level evidence, resolves local paths lexically relative
to the source document, assigns operational intent/confidence labels, and
reports file-, repository-, target-, and stratum-level summaries.  It does not
claim that a target exists; repository-tree verification belongs to Phase 4.

Example:

    python scripts/claude_markdown_references.py \
        --db mined.db \
        --occurrences-output article/claude_markdown_reference_occurrences.csv \
        --files-output article/claude_markdown_reference_files.csv \
        --targets-output article/claude_markdown_reference_targets.csv \
        --strata-output article/claude_markdown_reference_strata.csv \
        --summary-output article/claude_markdown_reference_summary.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import posixpath
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miner.markdown_references import (  # noqa: E402
    MarkdownReference,
    extract_external_markdown_references,
    extract_local_markdown_reference_occurrences,
    iter_unfenced_lines,
)


ANALYZER_ID = "claude_markdown_references_v1"
EXACT_CLAUDE_SCOPE = (
    "case-insensitive basename equal to CLAUDE.md; stored GitHub paths use "
    "POSIX '/' separators"
)
WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.UNICODE)
HEADING_RE = re.compile(r"^[ ]{0,3}#{1,6}[ \t]+.*$")
FENCE_LINE_RE = re.compile(r"^[ ]{0,3}(?:`{3,}|~{3,})", re.MULTILINE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
TARGET_MARKER = "__markdown_target__"
FOLLOW_TARGET_RE = re.compile(
    r"\b(?:follow|obey|adhere\s+to|comply\s+with|apply)\b"
    r"[^.\n]{0,100}__markdown_target__"
    r"|__markdown_target__[^.\n]{0,100}"
    r"\b(?:must\s+be\s+followed|governs?|takes?\s+precedence)\b",
    re.IGNORECASE,
)
REQUIRED_TARGET_ACTION_RE = re.compile(
    r"\b(?:must|always|required|mandatory)\b(?:\s+\w+){0,3}\s+"
    r"(?:read|review|consult|use|refer\s+to)\b[^.\n]{0,80}"
    r"__markdown_target__"
    r"|\bbefore\b[^.\n]{0,100}\b(?:read|review|consult|use|refer\s+to)\b"
    r"[^.\n]{0,80}__markdown_target__",
    re.IGNORECASE,
)
INSTRUCTION_LOCATION_TARGET_RE = re.compile(
    r"\b(?:instructions?|rules?|guidelines?|requirements?|conventions?|"
    r"polic(?:y|ies)|source\s+of\s+truth)\b[^.\n]{0,80}"
    r"\b(?:live|lives|are|is|located|defined|documented|reside|"
    r"see|read|follow|use|from)\b[^.\n]{0,50}__markdown_target__"
    r"|\b(?:instructions?|rules?|guidelines?|requirements?|conventions?|"
    r"polic(?:y|ies))\b[^.\n]{0,30}:\s*[^.\n]{0,30}__markdown_target__"
    r"|\b(?:read|review|consult|use|refer\s+to|see)\b[^.\n]{0,80}"
    r"\b(?:instructions?|rules?|guidelines?|requirements?|conventions?|"
    r"polic(?:y|ies)|source\s+of\s+truth)\b[^.\n]{0,80}__markdown_target__"
    r"|__markdown_target__[^.\n]{0,80}"
    r"\b(?:contains?|defines?|documents?)\b[^.\n]{0,60}"
    r"\b(?:instructions?|rules?|guidelines?|requirements?|conventions?|polic(?:y|ies))\b",
    re.IGNORECASE,
)
INSTRUCTION_TARGET_ACTION_RE = re.compile(
    r"\b(?:read|review|consult|use|refer\s+to|see)\b"
    r"[^.\n]{0,100}__markdown_target__",
    re.IGNORECASE,
)
FRONT_MATTER_BOUNDARY = {"---", "..."}
POINTER_MAX_RESIDUAL_WORDS = 50


@dataclass(frozen=True)
class ParsedDocument:
    local_references: tuple[MarkdownReference, ...]
    external_references: tuple[MarkdownReference, ...]
    contexts: dict[int, str]
    residual_word_count: int
    has_fenced_block: bool


@dataclass(frozen=True)
class ReferenceRow:
    repo_full_name: str
    source_path: str
    source_location: str
    source_directory_depth: int
    source_content_hash: str
    source_size_bytes: int | None
    repository_exact_files: int
    repository_stars: int | None
    repository_language: str
    line_number: int
    syntax: str
    scope: str
    intent_category: str
    confidence: str
    rule_id: str
    high_confidence_local_instructional: int
    raw_target: str
    normalized_target: str
    resolved_target: str
    target_basename: str
    target_category: str
    path_relation: str
    is_self_reference: int
    source_line: str
    evidence_context: str


@dataclass(frozen=True)
class FileRow:
    repo_full_name: str
    source_path: str
    source_location: str
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
    local_reference_occurrences: int
    local_nonself_reference_occurrences: int
    external_reference_occurrences: int
    distinct_local_nonself_targets: int
    high_confidence_local_instructional_occurrences: int
    instructional_fan_out: int
    direct_inclusion_occurrences: int
    instructional_delegation_occurrences: int
    contextual_local_occurrences: int
    self_reference_occurrences: int
    document_form: str
    pointer_residual_words: int
    target_categories: str


@dataclass(frozen=True)
class TargetRow:
    scope: str
    target_category: str
    target_basename: str
    occurrences: int
    files: int
    repositories: int
    high_confidence_local_instructional_occurrences: int
    high_confidence_local_instructional_files: int
    high_confidence_local_instructional_repositories: int
    direct_inclusion_occurrences: int
    instructional_delegation_occurrences: int
    contextual_occurrences: int


@dataclass(frozen=True)
class StratumRow:
    dimension: str
    value: str
    files: int
    repositories: int
    files_with_any_markdown_reference: int
    files_with_local_nonself_reference: int
    files_with_high_confidence_local_instructional_candidate: int
    candidate_share: float
    pointer_only_files: int
    instructions_plus_delegation_files: int
    high_confidence_local_instructional_occurrences: int
    total_instructional_fan_out: int
    mean_instructional_fan_out_all_files: float
    median_instructional_fan_out_all_files: int


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


def _share(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    rank = max(1, math.ceil(probability * len(values)))
    return int(values[rank - 1])


def _integer_distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": 0,
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
        "median": _nearest_rank(ordered, 0.50),
        "p75": _nearest_rank(ordered, 0.75),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _front_matter_lines(content: str) -> set[int]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return set()
    for index, line in enumerate(lines[1:], 2):
        if line.strip() in FRONT_MATTER_BOUNDARY:
            return set(range(1, index + 1))
    return set()


def _evidence_context(lines: Sequence[str], line_number: int) -> str:
    current = lines[line_number - 1].strip() if line_number <= len(lines) else ""
    if len(WORD_RE.findall(current)) > 18:
        return current[:700]
    for prior_number in range(line_number - 1, max(0, line_number - 3), -1):
        prior = lines[prior_number - 1].strip()
        if not prior:
            continue
        if prior.endswith(":") or HEADING_RE.match(prior):
            return (prior + "\n" + current)[:700]
        break
    return current[:700]


def _pointer_residual_words(
    content: str,
    references: Sequence[MarkdownReference],
) -> int:
    references_by_line: dict[int, list[MarkdownReference]] = defaultdict(list)
    explicit_lines: set[int] = set()
    for reference in references:
        references_by_line[reference.line_number].append(reference)
        if reference.syntax in {"direct_include", "comment_include"}:
            explicit_lines.add(reference.line_number)

    retained: list[str] = []
    front_matter = _front_matter_lines(content)
    for line_number, line in iter_unfenced_lines(content):
        if (
            line_number in explicit_lines
            or line_number in front_matter
            or not line.strip()
            or HEADING_RE.match(line)
        ):
            continue
        cleaned = line
        for reference in references_by_line.get(line_number, []):
            cleaned = cleaned.replace(reference.raw_target, "")
        retained.append(cleaned)
    residual = HTML_COMMENT_RE.sub("", "\n".join(retained))
    return len(WORD_RE.findall(residual))


def _parse_document(content: str) -> ParsedDocument:
    local = extract_local_markdown_reference_occurrences(content)
    external = extract_external_markdown_references(content)
    all_references = local + external
    lines = content.splitlines()
    contexts = {
        line_number: _evidence_context(lines, line_number)
        for line_number in {reference.line_number for reference in all_references}
    }
    return ParsedDocument(
        local_references=local,
        external_references=external,
        contexts=contexts,
        residual_word_count=_pointer_residual_words(content, all_references),
        has_fenced_block=FENCE_LINE_RE.search(content) is not None,
    )


def resolve_local_target(source_path: str, normalized_target: str) -> str:
    """Resolve a normalized local target lexically against a source path."""
    if normalized_target.startswith("/"):
        return posixpath.normpath(normalized_target).lstrip("/")
    source_directory = posixpath.dirname(source_path)
    return posixpath.normpath(posixpath.join(source_directory, normalized_target))


def local_path_relation(source_path: str, resolved_target: str) -> str:
    if resolved_target == ".." or resolved_target.startswith("../"):
        return "outside_repository"
    source_directory = PurePosixPath(posixpath.dirname(source_path)).parts
    target_directory = PurePosixPath(posixpath.dirname(resolved_target)).parts
    if source_directory == target_directory:
        return "same_directory"
    if (
        len(target_directory) < len(source_directory)
        and source_directory[: len(target_directory)] == target_directory
    ):
        return "ancestor_directory"
    if (
        len(source_directory) < len(target_directory)
        and target_directory[: len(source_directory)] == source_directory
    ):
        return "descendant_directory"
    return "other_directory"


def target_category(target_basename: str, resolved_target: str) -> str:
    basename = target_basename.casefold()
    parts = {part.casefold() for part in PurePosixPath(resolved_target).parts[:-1]}
    stem = basename.removesuffix(".md")
    if basename == "agents.md":
        return "agents_md"
    if basename == "claude.md":
        return "claude_md"
    if basename.startswith("readme"):
        return "readme_md"
    if basename.startswith("contributing"):
        return "contributing_md"
    if "rule" in stem or "rules" in parts:
        return "rules_md"
    if any(token in stem for token in ("instruction", "guideline", "convention", "policy")):
        return "instruction_guidance_md"
    if "skill" in stem or parts & {"skill", "skills"}:
        return "skills_md"
    if parts & {"doc", "docs", "documentation"}:
        return "project_docs_md"
    return "other_md"


def _at_include_line(reference: MarkdownReference) -> bool:
    if not reference.raw_target.lstrip().startswith("@"):
        return False
    residual = reference.source_line.replace(reference.raw_target, "")
    residual = re.sub(r"[\s`*_~>#.+()\[\]{}:;,-]", "", residual)
    return not residual


def _target_cue_window(reference: MarkdownReference, context: str) -> str:
    """Return bounded evidence with the current target replaced by a marker."""
    source_words = len(WORD_RE.findall(reference.source_line))
    evidence = context if source_words <= 18 else reference.source_line
    position = evidence.find(reference.raw_target)
    if position < 0:
        evidence = reference.source_line
        position = evidence.find(reference.raw_target)
    if position < 0:
        return ""
    start = max(0, position - 140)
    end = min(len(evidence), position + len(reference.raw_target) + 140)
    window = evidence[start:end]
    return window.replace(reference.raw_target, TARGET_MARKER, 1)


def classify_local_intent(
    reference: MarkdownReference,
    category: str,
    context: str,
) -> tuple[str, str, str, int]:
    """Return intent, confidence, rule identifier, and high-confidence flag."""
    if reference.syntax in {"direct_include", "comment_include"}:
        return "direct_inclusion", "high", "explicit_include_syntax_v1", 1
    if _at_include_line(reference):
        return "direct_inclusion", "high", "standalone_at_include_v1", 1
    cue_window = _target_cue_window(reference, context)
    if FOLLOW_TARGET_RE.search(cue_window):
        return "instructional_delegation", "high", "follow_normative_cue_v1", 1
    if REQUIRED_TARGET_ACTION_RE.search(cue_window):
        return "instructional_delegation", "high", "required_action_cue_v1", 1
    if INSTRUCTION_LOCATION_TARGET_RE.search(cue_window):
        return "instructional_delegation", "high", "instruction_location_cue_v1", 1
    if (
        category in {"agents_md", "rules_md", "instruction_guidance_md"}
        and INSTRUCTION_TARGET_ACTION_RE.search(cue_window)
    ):
        return "instructional_delegation", "high", "instruction_target_action_v1", 1
    return "contextual_documentation", "low", "no_explicit_instructional_cue_v1", 0


def _source_location(path: str) -> str:
    return "root" if "/" not in path else "nested"


def _directory_depth(path: str) -> int:
    return max(0, len(PurePosixPath(path).parts) - 1)


def _size_band(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    if size_bytes <= 50:
        return "<=50"
    if size_bytes <= 200:
        return "51-200"
    if size_bytes <= 1000:
        return "201-1000"
    if size_bytes <= 5000:
        return "1001-5000"
    return ">5000"


def _star_band(stars: int | None) -> str:
    if stars is None:
        return "unknown"
    if stars == 0:
        return "0"
    if stars < 10:
        return "1-9"
    if stars < 100:
        return "10-99"
    if stars < 1000:
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


def analyze_references(
    db_path: str | Path,
) -> tuple[list[ReferenceRow], list[FileRow], int]:
    """Analyze exact ``CLAUDE.md`` content and return evidence/file rows."""
    connection = _readonly_connection(db_path)
    repo_counts, hash_counts, repo_hash_counts = _load_population_counts(connection)
    parsed_cache: dict[str, ParsedDocument] = {}
    references: list[ReferenceRow] = []
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
    try:
        for row in connection.execute(query):
            repo = str(row["repo_full_name"])
            source_path = str(row["path"])
            content_hash = str(row["content_hash"] or "")
            cache_key = content_hash or f"id:{int(row['id'])}"
            parsed = parsed_cache.get(cache_key)
            if parsed is None:
                parsed = _parse_document(str(row["content"] or ""))
                parsed_cache[cache_key] = parsed

            size_bytes = row["size_bytes"]
            stars = row["stars"]
            language = str(row["language"] or "Unknown")
            repo_exact_files = repo_counts[repo]
            location = _source_location(source_path)
            depth = _directory_depth(source_path)
            local_rows: list[ReferenceRow] = []

            for reference in parsed.local_references:
                resolved = resolve_local_target(source_path, reference.normalized_target)
                relation = local_path_relation(source_path, resolved)
                self_reference = int(resolved.casefold() == source_path.casefold())
                category = target_category(reference.target_basename, resolved)
                context = parsed.contexts.get(reference.line_number, reference.source_line)
                intent, confidence, rule_id, high_confidence = classify_local_intent(
                    reference,
                    category,
                    context,
                )
                if self_reference or relation == "outside_repository":
                    high_confidence = 0
                local_rows.append(
                    ReferenceRow(
                        repo_full_name=repo,
                        source_path=source_path,
                        source_location=location,
                        source_directory_depth=depth,
                        source_content_hash=content_hash,
                        source_size_bytes=size_bytes,
                        repository_exact_files=repo_exact_files,
                        repository_stars=stars,
                        repository_language=language,
                        line_number=reference.line_number,
                        syntax=reference.syntax,
                        scope="local",
                        intent_category=intent,
                        confidence=confidence,
                        rule_id=rule_id,
                        high_confidence_local_instructional=high_confidence,
                        raw_target=reference.raw_target,
                        normalized_target=reference.normalized_target,
                        resolved_target=resolved,
                        target_basename=reference.target_basename,
                        target_category=category,
                        path_relation=relation,
                        is_self_reference=self_reference,
                        source_line=reference.source_line[:700],
                        evidence_context=context,
                    )
                )

            external_rows = [
                ReferenceRow(
                    repo_full_name=repo,
                    source_path=source_path,
                    source_location=location,
                    source_directory_depth=depth,
                    source_content_hash=content_hash,
                    source_size_bytes=size_bytes,
                    repository_exact_files=repo_exact_files,
                    repository_stars=stars,
                    repository_language=language,
                    line_number=reference.line_number,
                    syntax=reference.syntax,
                    scope="external",
                    intent_category="external_reference",
                    confidence="high",
                    rule_id="external_markdown_url_v1",
                    high_confidence_local_instructional=0,
                    raw_target=reference.raw_target,
                    normalized_target=reference.normalized_target,
                    resolved_target="",
                    target_basename=reference.target_basename,
                    target_category=target_category(reference.target_basename, reference.raw_target),
                    path_relation="external",
                    is_self_reference=0,
                    source_line=reference.source_line[:700],
                    evidence_context=parsed.contexts.get(
                        reference.line_number,
                        reference.source_line,
                    ),
                )
                for reference in parsed.external_references
            ]
            references.extend(local_rows)
            references.extend(external_rows)

            nonself_local = [item for item in local_rows if not item.is_self_reference]
            high = [item for item in nonself_local if item.high_confidence_local_instructional]
            high_targets = {item.resolved_target for item in high}
            local_targets = {
                item.resolved_target
                for item in nonself_local
                if item.path_relation != "outside_repository"
            }
            if not local_rows and not external_rows:
                document_form = "no_detected_markdown_reference"
            elif high and not parsed.has_fenced_block and parsed.residual_word_count <= POINTER_MAX_RESIDUAL_WORDS:
                document_form = "pointer_only"
            elif high:
                document_form = "instructions_plus_delegation"
            elif nonself_local:
                document_form = "contextual_local_references_only"
            elif external_rows:
                document_form = "external_references_only"
            else:
                document_form = "self_references_only"

            global_copies = hash_counts.get(content_hash, 1)
            repo_content_copies = repo_hash_counts.get((repo, content_hash), 1)
            files.append(
                FileRow(
                    repo_full_name=repo,
                    source_path=source_path,
                    source_location=location,
                    source_directory_depth=depth,
                    content_hash=content_hash,
                    size_bytes=size_bytes,
                    repository_stars=stars,
                    repository_language=language,
                    repository_exact_files=repo_exact_files,
                    repository_multiplicity="multiple" if repo_exact_files > 1 else "single",
                    global_content_copies=global_copies,
                    repository_content_copies=repo_content_copies,
                    global_exact_duplicate=int(global_copies > 1),
                    within_repository_exact_duplicate=int(repo_content_copies > 1),
                    local_reference_occurrences=len(local_rows),
                    local_nonself_reference_occurrences=len(nonself_local),
                    external_reference_occurrences=len(external_rows),
                    distinct_local_nonself_targets=len(local_targets),
                    high_confidence_local_instructional_occurrences=len(high),
                    instructional_fan_out=len(high_targets),
                    direct_inclusion_occurrences=sum(
                        item.intent_category == "direct_inclusion" for item in nonself_local
                    ),
                    instructional_delegation_occurrences=sum(
                        item.intent_category == "instructional_delegation" for item in nonself_local
                    ),
                    contextual_local_occurrences=sum(
                        item.intent_category == "contextual_documentation" for item in nonself_local
                    ),
                    self_reference_occurrences=sum(item.is_self_reference for item in local_rows),
                    document_form=document_form,
                    pointer_residual_words=parsed.residual_word_count,
                    target_categories=";".join(
                        sorted({item.target_category for item in nonself_local})
                    ),
                )
            )
        database_changes = connection.total_changes
    finally:
        connection.close()

    references.sort(
        key=lambda item: (
            item.repo_full_name.casefold(),
            item.repo_full_name,
            item.source_path.casefold(),
            item.source_path,
            item.line_number,
            item.scope,
            item.syntax,
            item.normalized_target.casefold(),
            item.normalized_target,
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
    return references, files, database_changes


def aggregate_targets(references: Sequence[ReferenceRow]) -> list[TargetRow]:
    groups: dict[tuple[str, str, str], list[ReferenceRow]] = defaultdict(list)
    for reference in references:
        groups[(reference.scope, reference.target_category, reference.target_basename)].append(reference)
    rows: list[TargetRow] = []
    for (scope, category, basename), members in groups.items():
        high = [item for item in members if item.high_confidence_local_instructional]
        rows.append(
            TargetRow(
                scope=scope,
                target_category=category,
                target_basename=basename,
                occurrences=len(members),
                files=len({(item.repo_full_name, item.source_path) for item in members}),
                repositories=len({item.repo_full_name for item in members}),
                high_confidence_local_instructional_occurrences=len(high),
                high_confidence_local_instructional_files=len(
                    {(item.repo_full_name, item.source_path) for item in high}
                ),
                high_confidence_local_instructional_repositories=len(
                    {item.repo_full_name for item in high}
                ),
                direct_inclusion_occurrences=sum(
                    item.intent_category == "direct_inclusion" for item in members
                ),
                instructional_delegation_occurrences=sum(
                    item.intent_category == "instructional_delegation" for item in members
                ),
                contextual_occurrences=sum(
                    item.intent_category == "contextual_documentation" for item in members
                ),
            )
        )
    rows.sort(
        key=lambda item: (
            item.scope,
            -item.high_confidence_local_instructional_files,
            -item.files,
            -item.occurrences,
            item.target_basename,
        )
    )
    return rows


def _stratum_values(file: FileRow) -> dict[str, str]:
    return {
        "source_location": file.source_location,
        "repository_multiplicity": file.repository_multiplicity,
        "within_repository_exact_duplicate": (
            "yes" if file.within_repository_exact_duplicate else "no"
        ),
        "global_exact_duplicate": "yes" if file.global_exact_duplicate else "no",
        "file_size_band": _size_band(file.size_bytes),
        "repository_star_band": _star_band(file.repository_stars),
        "repository_primary_language": file.repository_language,
    }


def aggregate_strata(files: Sequence[FileRow]) -> list[StratumRow]:
    groups: dict[tuple[str, str], list[FileRow]] = defaultdict(list)
    for file in files:
        for dimension, value in _stratum_values(file).items():
            groups[(dimension, value)].append(file)
    rows: list[StratumRow] = []
    for (dimension, value), members in groups.items():
        candidate_files = sum(item.instructional_fan_out > 0 for item in members)
        fan_out = sorted(item.instructional_fan_out for item in members)
        rows.append(
            StratumRow(
                dimension=dimension,
                value=value,
                files=len(members),
                repositories=len({item.repo_full_name for item in members}),
                files_with_any_markdown_reference=sum(
                    item.local_reference_occurrences + item.external_reference_occurrences > 0
                    for item in members
                ),
                files_with_local_nonself_reference=sum(
                    item.local_nonself_reference_occurrences > 0 for item in members
                ),
                files_with_high_confidence_local_instructional_candidate=candidate_files,
                candidate_share=_share(candidate_files, len(members)),
                pointer_only_files=sum(item.document_form == "pointer_only" for item in members),
                instructions_plus_delegation_files=sum(
                    item.document_form == "instructions_plus_delegation" for item in members
                ),
                high_confidence_local_instructional_occurrences=sum(
                    item.high_confidence_local_instructional_occurrences for item in members
                ),
                total_instructional_fan_out=sum(item.instructional_fan_out for item in members),
                mean_instructional_fan_out_all_files=(
                    sum(item.instructional_fan_out for item in members) / len(members)
                ),
                median_instructional_fan_out_all_files=_nearest_rank(fan_out, 0.50),
            )
        )
    rows.sort(key=lambda item: (item.dimension, -item.files, item.value.casefold(), item.value))
    return rows


def build_summary(
    references: Sequence[ReferenceRow],
    files: Sequence[FileRow],
    targets: Sequence[TargetRow],
    strata: Sequence[StratumRow],
    *,
    database_sha256_before: str,
    database_sha256_after: str,
    database_changes: int,
) -> dict[str, object]:
    total_files = len(files)
    repositories = {item.repo_full_name for item in files}
    local = [item for item in references if item.scope == "local"]
    local_nonself = [item for item in local if not item.is_self_reference]
    high = [item for item in local_nonself if item.high_confidence_local_instructional]
    external = [item for item in references if item.scope == "external"]
    files_with_any = {
        (item.repo_full_name, item.source_path)
        for item in files
        if item.local_reference_occurrences + item.external_reference_occurrences > 0
    }
    files_with_local = {
        (item.repo_full_name, item.source_path)
        for item in files
        if item.local_nonself_reference_occurrences > 0
    }
    candidate_files = {
        (item.repo_full_name, item.source_path)
        for item in files
        if item.instructional_fan_out > 0
    }
    candidate_repositories = {
        item.repo_full_name for item in files if item.instructional_fan_out > 0
    }
    files_with_local_including_self = {
        (item.repo_full_name, item.source_path)
        for item in files
        if item.local_reference_occurrences > 0
    }
    files_with_external = {
        (item.repo_full_name, item.source_path)
        for item in files
        if item.external_reference_occurrences > 0
    }
    form_counts = Counter(item.document_form for item in files)
    syntax_counts = Counter(item.syntax for item in references)
    intent_counts = Counter(item.intent_category for item in references)
    relation_counts = Counter(item.path_relation for item in local_nonself)
    high_relation_counts = Counter(item.path_relation for item in high)
    category_counts = Counter(item.target_category for item in local_nonself)
    high_category_counts = Counter(item.target_category for item in high)

    key_strata = [
        asdict(row)
        for row in strata
        if row.dimension in {"source_location", "repository_multiplicity"}
    ]
    top_local_targets = [
        asdict(row)
        for row in targets
        if row.scope == "local"
    ][:20]

    return {
        "analyzer_id": ANALYZER_ID,
        "scope": EXACT_CLAUDE_SCOPE,
        "intent_status": (
            "operational candidate labels; independent manual validation is Phase 3"
        ),
        "target_resolution_status": (
            "lexical normalization only; existence and content verification are Phase 4"
        ),
        "database_sha256_before": database_sha256_before,
        "database_sha256_after": database_sha256_after,
        "database_unchanged": database_sha256_before == database_sha256_after,
        "sqlite_total_changes": database_changes,
        "population": {
            "files": total_files,
            "repositories": len(repositories),
        },
        "reference_counts": {
            "all_occurrences": len(references),
            "local_occurrences_including_self": len(local),
            "local_nonself_occurrences": len(local_nonself),
            "self_reference_occurrences": sum(item.is_self_reference for item in local),
            "external_markdown_occurrences": len(external),
            "files_with_any_markdown_reference": len(files_with_any),
            "repositories_with_any_markdown_reference": len(
                {repo for repo, _path in files_with_any}
            ),
            "files_with_local_reference_including_self": len(
                files_with_local_including_self
            ),
            "files_with_local_nonself_reference": len(files_with_local),
            "files_with_external_markdown_reference": len(files_with_external),
            "repositories_with_external_markdown_reference": len(
                {repo for repo, _path in files_with_external}
            ),
        },
        "high_confidence_local_instructional_candidates": {
            "occurrences": len(high),
            "distinct_source_target_edges": len(
                {
                    (item.repo_full_name, item.source_path, item.resolved_target)
                    for item in high
                }
            ),
            "files": len(candidate_files),
            "file_share": _share(len(candidate_files), total_files),
            "repositories": len(candidate_repositories),
            "repository_share": _share(len(candidate_repositories), len(repositories)),
            "direct_inclusion_occurrences": sum(
                item.intent_category == "direct_inclusion" for item in high
            ),
            "instructional_delegation_occurrences": sum(
                item.intent_category == "instructional_delegation" for item in high
            ),
        },
        "document_forms": [
            {
                "document_form": form,
                "files": count,
                "share": _share(count, total_files),
            }
            for form, count in sorted(form_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "instructional_fan_out_all_files": _integer_distribution(
            item.instructional_fan_out for item in files
        ),
        "instructional_fan_out_candidate_files": _integer_distribution(
            item.instructional_fan_out for item in files if item.instructional_fan_out > 0
        ),
        "syntax_counts": dict(sorted(syntax_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "local_path_relation_counts": dict(sorted(relation_counts.items())),
        "high_confidence_path_relation_counts": dict(sorted(high_relation_counts.items())),
        "local_target_category_counts": dict(sorted(category_counts.items())),
        "high_confidence_target_category_counts": dict(sorted(high_category_counts.items())),
        "key_strata": key_strata,
        "top_local_target_basenames": top_local_targets,
        "configuration": {
            "pointer_max_residual_words": POINTER_MAX_RESIDUAL_WORDS,
            "fenced_blocks_excluded_from_detection": True,
            "self_references_excluded_from_candidate_numerator": True,
            "outside_repository_paths_excluded_from_candidate_numerator": True,
            "local_target_case_sensitive_for_fan_out": True,
            "reference_unit": "literal detected occurrence; repeated mentions retained",
        },
    }


def _write_dataclass_csv(
    rows: Sequence[object],
    output_path: str | Path,
    *,
    extra_first_column: str | None = None,
    extra_prefix: str = "",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot infer CSV fields for empty output: {output}")
    fieldnames = list(asdict(rows[0]))
    if extra_first_column:
        fieldnames.insert(0, extra_first_column)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            values = asdict(row)
            if extra_first_column:
                values[extra_first_column] = f"{extra_prefix}{index:08d}"
            writer.writerow(values)
    return output


def write_occurrences_csv(rows: Sequence[ReferenceRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(
        rows,
        output_path,
        extra_first_column="reference_id",
        extra_prefix="REF-",
    )


def write_files_csv(rows: Sequence[FileRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_targets_csv(rows: Sequence[TargetRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_strata_csv(rows: Sequence[StratumRow], output_path: str | Path) -> Path:
    return _write_dataclass_csv(rows, output_path)


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mined.db")
    parser.add_argument(
        "--occurrences-output",
        default="article/claude_markdown_reference_occurrences.csv",
    )
    parser.add_argument(
        "--files-output",
        default="article/claude_markdown_reference_files.csv",
    )
    parser.add_argument(
        "--targets-output",
        default="article/claude_markdown_reference_targets.csv",
    )
    parser.add_argument(
        "--strata-output",
        default="article/claude_markdown_reference_strata.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="article/claude_markdown_reference_summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    database_hash_before = _sha256_file(arguments.db)
    references, files, database_changes = analyze_references(arguments.db)
    database_hash_after = _sha256_file(arguments.db)
    targets = aggregate_targets(references)
    strata = aggregate_strata(files)
    summary = build_summary(
        references,
        files,
        targets,
        strata,
        database_sha256_before=database_hash_before,
        database_sha256_after=database_hash_after,
        database_changes=database_changes,
    )
    occurrence_path = write_occurrences_csv(references, arguments.occurrences_output)
    file_path = write_files_csv(files, arguments.files_output)
    target_path = write_targets_csv(targets, arguments.targets_output)
    stratum_path = write_strata_csv(strata, arguments.strata_output)
    summary_path = write_summary_json(summary, arguments.summary_output)
    print(
        f"Analyzed {len(files):,} exact CLAUDE.md files; wrote "
        f"{len(references):,} references to {occurrence_path}, file summaries to "
        f"{file_path}, {len(targets):,} target rows to {target_path}, "
        f"{len(strata):,} strata to {stratum_path}, and {summary_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
