"""Inspect notation/diagram candidates in ``mined.db`` without modifying it.

The stored notation summary contains exact per-file totals, but its detailed
``detections`` array is capped at 30 entries per file. This script reports the
stored totals and replays the unchanged fenced-block rules for Mermaid, ASCII,
JSON, and YAML so occurrence-level provenance and Mermaid subtypes are not
truncated. It also exports a deterministic snippet sample for inspection.

Example:

    python scripts/notation_candidate_stats.py \
        --db mined.db \
        --scope all \
        --summary-output article/notation_candidate_summary.json \
        --mermaid-output article/mermaid_subtypes.csv \
        --sample-output article/notation_candidate_inspection_sample.csv

The source database is opened with ``mode=ro&immutable=1`` and
``PRAGMA query_only=ON``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

# Make ``python scripts/notation_candidate_stats.py`` work from any directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from miner.analyzers.notation import FENCE_RE  # noqa: E402
from miner.config import NotationDef, load_notations  # noqa: E402


SCOPES = ("all", "exact-claude")
REPLAY_IDS = {"ascii_diagram", "json_schema", "yaml_schema", "mermaid"}
EXPLICIT_DIAGRAM_IDS = {"mermaid", "plantuml", "graphviz", "d2"}
ASCII_RULE_NAMES = {
    1: "box_drawing_characters",
    2: "plus_dash_box",
    3: "ascii_arrow",
}


def _scope_predicate(scope: str, *, alias: str = "") -> str:
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope!r}")
    if scope == "all":
        return "1 = 1"
    prefix = f"{alias}." if alias else ""
    return (
        f"(lower({prefix}path) = 'claude.md' "
        f"OR lower({prefix}path) LIKE '%/claude.md')"
    )


def _scope_definition(scope: str) -> str:
    return {
        "all": "all query-matched stored file records",
        "exact-claude": "case-insensitive basename equal to CLAUDE.md",
    }[scope]


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


def _heuristics_match(notation: NotationDef, body: str) -> bool:
    return bool(_matching_heuristic_rules(notation, body))


def _matching_heuristic_rules(notation: NotationDef, body: str) -> tuple[int, ...]:
    matches: list[int] = []
    for index, pattern in enumerate(notation.untagged_heuristics, 1):
        try:
            matched = re.search(pattern, body) is not None
        except re.error:
            matched = pattern in body
        if matched:
            matches.append(index)
    return tuple(matches)


def _detect_subtype(notation: NotationDef, body: str) -> str:
    first_lines = "\n".join(body.strip().splitlines()[:3]).lower()
    for subtype, markers in notation.subtype_markers.items():
        if any(marker.lower() in first_lines for marker in markers):
            return subtype
    return "(unclassified)"


def _unclassified_mermaid_directive(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9_-]*)", stripped)
        return match.group(1).casefold() if match else "(other)"
    return "(empty/comment-only)"


class DeterministicSampler:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
        self._sequence = 0

    def consider(self, category: str, identity: str, record: dict[str, Any]) -> None:
        if self.capacity <= 0:
            return
        score = int.from_bytes(
            hashlib.sha256(
                f"notation-inspection-v1\0{identity}".encode("utf-8")
            ).digest(),
            "big",
        )
        self._sequence += 1
        item = (-score, self._sequence, record)
        heap = self._heaps.setdefault(category, [])
        if len(heap) < self.capacity:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for category in sorted(self._heaps):
            items = sorted(self._heaps[category], key=lambda item: -item[0])
            for _negative_score, _sequence, record in items:
                rows.append({"sample_category": category, **record})
        for index, row in enumerate(rows, 1):
            row["sample_id"] = f"N{index:03d}"
        return rows


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def analyze_notation_candidates(
    db_path: str | Path,
    *,
    scope: str = "all",
    sample_per_category: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    notations = load_notations()
    by_id = {notation.id: notation for notation in notations}
    by_fence_language = {
        language.lower(): notation
        for notation in notations
        for language in notation.fence_languages
    }
    predicate = _scope_predicate(scope, alias="f")
    connection = _readonly_connection(db_path)

    stored_occurrences: Counter[str] = Counter()
    stored_files: Counter[str] = Counter()
    stored_subtype_files: Counter[str] = Counter()
    stored_notation_confidence_files: Counter[str] = Counter()
    captured_confidence: Counter[str] = Counter()
    captured_via: Counter[str] = Counter()
    captured_by_notation: Counter[str] = Counter()
    omitted_by_notation: Counter[str] = Counter()
    capped_files = 0
    omitted_detection_records = 0

    replay_occurrences: Counter[str] = Counter()
    replay_files: Counter[str] = Counter()
    replay_confidence: Counter[str] = Counter()
    replay_via: Counter[str] = Counter()
    fence_languages: dict[str, Counter[str]] = {
        notation_id: Counter() for notation_id in REPLAY_IDS
    }
    subtype_occurrences: Counter[str] = Counter()
    subtype_files: Counter[str] = Counter()
    ascii_rule_combinations: Counter[str] = Counter()
    ascii_arrow_diagnostics: Counter[str] = Counter()
    json_evidence: Counter[str] = Counter()
    yaml_evidence: Counter[str] = Counter()
    mermaid_unclassified_directives: Counter[str] = Counter()
    sampler = DeterministicSampler(sample_per_category)
    analyzed_files = 0
    files_with_any = 0
    files_with_high_explicit_diagram = 0
    fenced_blocks = 0

    try:
        scoped_files = int(
            connection.execute(
                f"SELECT count(*) FROM files AS f WHERE {predicate}"
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            SELECT f.id, f.repo_full_name, f.path, f.content, a.result_json
            FROM files AS f
            INNER JOIN analysis AS a ON a.file_id = f.id
            WHERE a.analyzer_id = 'notation' AND {predicate}
            ORDER BY f.id
            """
        )
        for row in rows:
            analyzed_files += 1
            stored = json.loads(row["result_json"])
            if stored.get("any_notation_detected"):
                files_with_any += 1
            file_has_high_explicit_diagram = False

            detections = stored.get("detections", [])
            total = int(stored.get("notation_count_total", 0))
            captured_in_file: Counter[str] = Counter(
                str(detection.get("notation_id", "?"))
                for detection in detections
            )
            if total > len(detections):
                capped_files += 1
                omitted_detection_records += total - len(detections)

            for notation_id, notation_summary in stored.get(
                "notations_summary", {}
            ).items():
                count = int(notation_summary.get("count", 0))
                stored_occurrences[notation_id] += count
                stored_files[notation_id] += 1
                missing = count - captured_in_file[notation_id]
                if missing > 0:
                    omitted_by_notation[notation_id] += missing
                for subtype in notation_summary.get("subtypes", []):
                    stored_subtype_files[f"{notation_id}:{subtype}"] += 1
                for confidence in notation_summary.get("confidences", []):
                    stored_notation_confidence_files[
                        f"{notation_id}:{confidence}"
                    ] += 1
                    if (
                        notation_id in EXPLICIT_DIAGRAM_IDS
                        and confidence == "high"
                    ):
                        file_has_high_explicit_diagram = True

            if file_has_high_explicit_diagram:
                files_with_high_explicit_diagram += 1

            for detection in detections:
                notation_id = str(detection.get("notation_id", "?"))
                captured_by_notation[notation_id] += 1
                captured_confidence[str(detection.get("confidence", "?"))] += 1
                captured_via[str(detection.get("detected_via", "?"))] += 1

            seen_notations: set[str] = set()
            seen_subtypes: set[str] = set()
            blocks = FENCE_RE.findall(row["content"] or "")
            fenced_blocks += len(blocks)
            for block_index, (raw_language, body) in enumerate(blocks):
                language = raw_language.strip().lower()
                display_language = language or "(untagged)"
                direct = by_fence_language.get(language)
                if direct is not None:
                    candidate_notations = ((direct, "high", "fence_language"),)
                else:
                    candidate_notations = tuple(
                        (notation, "medium", "fenced_block_heuristic")
                        for notation in notations
                        if _heuristics_match(notation, body)
                    )

                for notation, confidence, via in candidate_notations:
                    if notation.id not in REPLAY_IDS:
                        continue
                    notation_id = notation.id
                    replay_occurrences[notation_id] += 1
                    replay_confidence[confidence] += 1
                    replay_via[via] += 1
                    fence_languages[notation_id][display_language] += 1
                    seen_notations.add(notation_id)
                    subtype = _detect_subtype(notation, body)
                    category_detail = subtype

                    if notation_id == "mermaid":
                        subtype_key = f"mermaid:{subtype}"
                        subtype_occurrences[subtype_key] += 1
                        seen_subtypes.add(subtype_key)
                        if subtype == "(unclassified)":
                            mermaid_unclassified_directives[
                                _unclassified_mermaid_directive(body)
                            ] += 1
                    elif notation_id == "ascii_diagram":
                        rules = _matching_heuristic_rules(notation, body)
                        if confidence == "high":
                            category_detail = "explicit_ascii_fence"
                        else:
                            names = tuple(
                                ASCII_RULE_NAMES.get(index, f"rule_{index}")
                                for index in rules
                            )
                            category_detail = "+".join(names)
                            ascii_rule_combinations[category_detail] += 1
                            if 3 in rules:
                                diagnostic_prefix = (
                                    "arrow_only"
                                    if rules == (3,)
                                    else "multiple_rules"
                                )
                                without_html_comments = re.sub(
                                    r"<!--.*?-->", "", body, flags=re.DOTALL
                                )
                                arrow_pattern = notation.untagged_heuristics[2]
                                if re.search(arrow_pattern, without_html_comments):
                                    ascii_arrow_diagnostics[
                                        f"{diagnostic_prefix}:"
                                        "arrow_remains_after_html_comment_removal"
                                    ] += 1
                                else:
                                    ascii_arrow_diagnostics[
                                        f"{diagnostic_prefix}:"
                                        "html_comment_delimiters_only"
                                    ] += 1
                    elif notation_id in {"json_schema", "yaml_schema"}:
                        evidence = _heuristics_match(notation, body)
                        evidence_label = (
                            "schema_marker_present"
                            if evidence
                            else "no_configured_schema_marker"
                        )
                        category_detail = evidence_label
                        target = (
                            json_evidence
                            if notation_id == "json_schema"
                            else yaml_evidence
                        )
                        target[evidence_label] += 1

                    snippet = " ".join(body.strip().split())[:600]
                    sampler.consider(
                        f"{notation_id}:{confidence}:{category_detail}",
                        f"{row['id']}:{block_index}:{notation_id}",
                        {
                            "repo_full_name": row["repo_full_name"],
                            "path": row["path"],
                            "block_index": block_index,
                            "fence_language": display_language,
                            "notation_id": notation_id,
                            "confidence": confidence,
                            "subtype": (
                                subtype if notation_id == "mermaid" else ""
                            ),
                            "snippet": snippet,
                        },
                    )

            for notation_id in seen_notations:
                replay_files[notation_id] += 1
            for subtype_key in seen_subtypes:
                subtype_files[subtype_key] += 1
    finally:
        connection.close()

    if analyzed_files != scoped_files:
        raise RuntimeError(
            "notation analysis rows do not match scoped files: "
            f"{analyzed_files} != {scoped_files}"
        )
    for notation_id in REPLAY_IDS:
        if replay_occurrences[notation_id] != stored_occurrences[notation_id]:
            raise RuntimeError(
                f"replayed {notation_id} total does not match stored summary: "
                f"{replay_occurrences[notation_id]} != "
                f"{stored_occurrences[notation_id]}"
            )

    mermaid_total = replay_occurrences["mermaid"]
    mermaid_files = replay_files["mermaid"]
    mermaid_rows: list[dict[str, Any]] = []
    subtype_names = {
        key.split(":", 1)[1]
        for key in set(subtype_occurrences) | set(subtype_files)
        if key.startswith("mermaid:")
    }
    for subtype in sorted(
        subtype_names,
        key=lambda value: (
            -subtype_occurrences[f"mermaid:{value}"],
            value,
        ),
    ):
        blocks = subtype_occurrences[f"mermaid:{subtype}"]
        files = subtype_files[f"mermaid:{subtype}"]
        mermaid_rows.append(
            {
                "subtype": subtype,
                "blocks": blocks,
                "block_share": blocks / mermaid_total if mermaid_total else 0.0,
                "files": files,
                "file_share": files / mermaid_files if mermaid_files else 0.0,
            }
        )

    summary = {
        "scope": scope,
        "scope_definition": _scope_definition(scope),
        "scoped_files": scoped_files,
        "analyzed_files": analyzed_files,
        "files_with_any_candidate": files_with_any,
        "files_with_any_candidate_share": (
            files_with_any / scoped_files if scoped_files else 0.0
        ),
        "stored_results": {
            "occurrences_by_notation": _counter_dict(stored_occurrences),
            "files_by_notation": _counter_dict(stored_files),
            "report_style_subtype_file_counts": _counter_dict(
                stored_subtype_files
            ),
            "file_notation_confidence_counts": _counter_dict(
                stored_notation_confidence_files
            ),
            "files_with_high_explicit_diagram_dsl": (
                files_with_high_explicit_diagram
            ),
            "files_with_high_explicit_diagram_dsl_share": (
                files_with_high_explicit_diagram / scoped_files
                if scoped_files
                else 0.0
            ),
            "detailed_detection_cap": 30,
            "files_exceeding_cap": capped_files,
            "omitted_detailed_detection_records": omitted_detection_records,
            "omitted_detailed_records_by_notation": _counter_dict(
                omitted_by_notation
            ),
            "captured_detections_by_notation": _counter_dict(
                captured_by_notation
            ),
            "captured_confidence_counts": _counter_dict(captured_confidence),
            "captured_detection_methods": _counter_dict(captured_via),
        },
        "fenced_block_replay": {
            "fenced_blocks": fenced_blocks,
            "occurrences_by_notation": _counter_dict(replay_occurrences),
            "files_by_notation": _counter_dict(replay_files),
            "confidence_counts": _counter_dict(replay_confidence),
            "detection_methods": _counter_dict(replay_via),
            "fence_languages_by_notation": {
                notation_id: _counter_dict(counter)
                for notation_id, counter in sorted(fence_languages.items())
            },
            "ascii_rule_combinations": _counter_dict(
                ascii_rule_combinations
            ),
            "ascii_arrow_diagnostics": _counter_dict(
                ascii_arrow_diagnostics
            ),
            "json_configured_marker_evidence": _counter_dict(json_evidence),
            "yaml_configured_marker_evidence": _counter_dict(yaml_evidence),
            "mermaid_subtype_occurrences": _counter_dict(
                Counter(
                    {
                        key.split(":", 1)[1]: value
                        for key, value in subtype_occurrences.items()
                        if key.startswith("mermaid:")
                    }
                )
            ),
            "mermaid_subtype_files": _counter_dict(
                Counter(
                    {
                        key.split(":", 1)[1]: value
                        for key, value in subtype_files.items()
                        if key.startswith("mermaid:")
                    }
                )
            ),
            "mermaid_unclassified_first_directives": _counter_dict(
                mermaid_unclassified_directives
            ),
        },
        "inspection_sample_rows": len(sampler.rows()),
        "database_writes": 0,
    }
    return summary, mermaid_rows, sampler.rows()


def write_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_mermaid_csv(
    rows: Sequence[dict[str, Any]], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("subtype", "blocks", "block_share", "files", "file_share"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_sample_csv(
    rows: Sequence[dict[str, Any]], output_path: str | Path
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id",
        "sample_category",
        "repo_full_name",
        "path",
        "block_index",
        "fence_language",
        "notation_id",
        "confidence",
        "subtype",
        "snippet",
    )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect stored notation candidates without modifying mined.db."
    )
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="all",
        help="population to inspect (default: all)",
    )
    parser.add_argument(
        "--summary-output",
        default="article/notation_candidate_summary.json",
        help="machine-readable JSON summary",
    )
    parser.add_argument(
        "--mermaid-output",
        default="article/mermaid_subtypes.csv",
        help="Mermaid subtype CSV",
    )
    parser.add_argument(
        "--sample-output",
        default="article/notation_candidate_inspection_sample.csv",
        help="deterministic inspection-sample CSV",
    )
    parser.add_argument(
        "--sample-per-category",
        type=nonnegative_int,
        default=10,
        help="maximum snippets per rule/category (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, mermaid_rows, sample_rows = analyze_notation_candidates(
            args.db,
            scope=args.scope,
            sample_per_category=args.sample_per_category,
        )
        summary_path = write_summary(summary, args.summary_output)
        mermaid_path = write_mermaid_csv(mermaid_rows, args.mermaid_output)
        sample_path = write_sample_csv(sample_rows, args.sample_output)
    except (FileNotFoundError, json.JSONDecodeError, OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    stored = summary["stored_results"]
    replay = summary["fenced_block_replay"]
    print(f"Scope: {summary['scope']} ({summary['scope_definition']})")
    print(f"Analyzed files: {summary['analyzed_files']:,}")
    print(
        f"Files with candidates: {summary['files_with_any_candidate']:,} "
        f"({summary['files_with_any_candidate_share']:.2%})"
    )
    print("Stored candidate occurrences:")
    for notation_id, count in stored["occurrences_by_notation"].items():
        print(f"  {notation_id:20s} {count:>8,}")
    print("Mermaid subtypes (replayed block occurrences):")
    for row in mermaid_rows:
        print(
            f"  {row['subtype']:20s} {row['blocks']:>5,} blocks "
            f"in {row['files']:>4,} files"
        )
    print(
        "Detailed detection cap: "
        f"{stored['files_exceeding_cap']} files omit "
        f"{stored['omitted_detailed_detection_records']} detailed records"
    )
    print(f"Replayed fenced blocks: {replay['fenced_blocks']:,}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print(f"Mermaid CSV: {mermaid_path.resolve()}")
    print(f"Inspection sample: {sample_path.resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
