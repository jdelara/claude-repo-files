"""Export fenced-code language statistics from ``mined.db`` read-only.

The default population is files whose case-insensitive basename is exactly
``CLAUDE.md``. Raw fence labels are reported as stored by the structure
analyzer, alongside a conservative normalization of common aliases such as
``bash``/``sh`` and ``typescript``/``ts``/``tsx``.

Example:

    python scripts/fenced_language_stats.py \
        --db mined.db \
        --scope exact-claude \
        --raw-output article/fenced_languages_raw.csv \
        --normalized-output article/fenced_languages_normalized.csv \
        --summary-output article/fenced_languages_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SCOPES = ("exact-claude", "markdown", "all")

# This deliberately normalizes only common spelling aliases and language
# variants. Unrecognized labels remain unchanged so unusual or malformed tags
# are visible rather than silently reclassified.
NORMALIZED_ALIASES = {
    "(untagged)": "(untagged)",
    "bash": "Shell",
    "sh": "Shell",
    "shell": "Shell",
    "shellscript": "Shell",
    "zsh": "Shell",
    "fish": "Shell",
    "ksh": "Shell",
    "powershell": "PowerShell",
    "ps1": "PowerShell",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "tsx": "TypeScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "jsx": "JavaScript",
    "node": "JavaScript",
    "nodejs": "JavaScript",
    "python": "Python",
    "py": "Python",
    "python3": "Python",
    "json": "JSON",
    "jsonc": "JSON",
    "json5": "JSON",
    "yaml": "YAML",
    "yml": "YAML",
    "text": "Plain text",
    "txt": "Plain text",
    "plaintext": "Plain text",
    "plain": "Plain text",
    "markdown": "Markdown",
    "md": "Markdown",
    "csharp": "C#",
    "cs": "C#",
    "c#": "C#",
    "cpp": "C++",
    "c++": "C++",
    "cxx": "C++",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "rs": "Rust",
    "sql": "SQL",
    "postgresql": "SQL",
    "postgres": "SQL",
    "mysql": "SQL",
    "sqlite": "SQL",
    "html": "HTML",
    "htm": "HTML",
    "css": "CSS",
    "java": "Java",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "php": "PHP",
    "ruby": "Ruby",
    "lua": "Lua",
    "dart": "Dart",
    "toml": "TOML",
    "xml": "XML",
    "dockerfile": "Dockerfile",
    "mermaid": "Mermaid",
}


@dataclass(frozen=True)
class FenceLanguageStat:
    label: str
    blocks: int
    files: int


@dataclass(frozen=True)
class FenceLanguageSummary:
    scope: str
    scope_definition: str
    scoped_files: int
    analyzed_files: int
    files_with_code_blocks: int
    files_with_code_block_share: float
    total_code_blocks: int
    tagged_code_blocks: int
    tagged_code_block_share: float
    untagged_code_blocks: int
    untagged_code_block_share: float
    raw_labels: int
    normalized_groups: int
    block_count_mismatches: int


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


def normalize_fence_label(label: str) -> str:
    cleaned = label.strip().lower()
    return NORMALIZED_ALIASES.get(cleaned, cleaned)


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _sorted_stats(
    block_counts: Counter[str],
    file_counts: Counter[str],
) -> list[FenceLanguageStat]:
    stats = [
        FenceLanguageStat(label, blocks, file_counts[label])
        for label, blocks in block_counts.items()
    ]
    stats.sort(key=lambda item: (-item.blocks, -item.files, item.label.casefold()))
    return stats


def analyze_fenced_languages(
    db_path: str | Path,
    *,
    scope: str = "exact-claude",
) -> tuple[
    FenceLanguageSummary,
    list[FenceLanguageStat],
    list[FenceLanguageStat],
]:
    """Return summary, raw-label stats, and normalized-label stats."""
    predicate = _scope_predicate(scope, alias="f")
    connection = _readonly_connection(db_path)
    try:
        scoped_files = int(
            connection.execute(
                f"SELECT count(*) FROM files AS f WHERE {predicate}"
            ).fetchone()[0]
        )
        rows = connection.execute(
            f"""
            SELECT a.result_json
            FROM analysis AS a
            INNER JOIN files AS f ON f.id = a.file_id
            WHERE a.analyzer_id = 'structure'
              AND {predicate}
            ORDER BY f.id
            """
        )

        raw_blocks: Counter[str] = Counter()
        raw_files: Counter[str] = Counter()
        normalized_blocks: Counter[str] = Counter()
        normalized_files: Counter[str] = Counter()
        analyzed_files = 0
        files_with_code = 0
        total_blocks = 0
        mismatches = 0

        for row in rows:
            result = json.loads(row["result_json"])
            analyzed_files += 1
            reported_total = int(result.get("code_block_count", 0) or 0)
            labels = result.get("code_block_languages", {}) or {}
            file_raw: Counter[str] = Counter()
            for raw_label, raw_count in labels.items():
                count = int(raw_count)
                if count <= 0:
                    continue
                label = str(raw_label).strip().lower()
                file_raw[label] += count

            counted_total = sum(file_raw.values())
            if counted_total != reported_total:
                mismatches += 1
            total_blocks += reported_total
            if reported_total > 0:
                files_with_code += 1

            file_normalized: Counter[str] = Counter()
            for label, count in file_raw.items():
                raw_blocks[label] += count
                raw_files[label] += 1
                file_normalized[normalize_fence_label(label)] += count
            for label, count in file_normalized.items():
                normalized_blocks[label] += count
                normalized_files[label] += 1
    finally:
        connection.close()

    raw_stats = _sorted_stats(raw_blocks, raw_files)
    normalized_stats = _sorted_stats(normalized_blocks, normalized_files)
    untagged = raw_blocks.get("(untagged)", 0)
    tagged = total_blocks - untagged
    summary = FenceLanguageSummary(
        scope=scope,
        scope_definition=_scope_definition(scope),
        scoped_files=scoped_files,
        analyzed_files=analyzed_files,
        files_with_code_blocks=files_with_code,
        files_with_code_block_share=(
            files_with_code / scoped_files if scoped_files else 0.0
        ),
        total_code_blocks=total_blocks,
        tagged_code_blocks=tagged,
        tagged_code_block_share=tagged / total_blocks if total_blocks else 0.0,
        untagged_code_blocks=untagged,
        untagged_code_block_share=(
            untagged / total_blocks if total_blocks else 0.0
        ),
        raw_labels=len(raw_stats),
        normalized_groups=len(normalized_stats),
        block_count_mismatches=mismatches,
    )
    return summary, raw_stats, normalized_stats


def write_stats_csv(
    stats: Sequence[FenceLanguageStat],
    output_path: str | Path,
    *,
    total_blocks: int,
    files_with_code_blocks: int,
    scoped_files: int,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "label",
                "blocks",
                "files",
                "share_of_all_blocks",
                "share_of_code_bearing_files",
                "share_of_scoped_files",
            ),
        )
        writer.writeheader()
        for rank, item in enumerate(stats, 1):
            writer.writerow(
                {
                    "rank": rank,
                    "label": item.label,
                    "blocks": item.blocks,
                    "files": item.files,
                    "share_of_all_blocks": (
                        item.blocks / total_blocks if total_blocks else 0.0
                    ),
                    "share_of_code_bearing_files": (
                        item.files / files_with_code_blocks
                        if files_with_code_blocks
                        else 0.0
                    ),
                    "share_of_scoped_files": (
                        item.files / scoped_files if scoped_files else 0.0
                    ),
                }
            )
    return output


def write_summary_json(
    summary: FenceLanguageSummary,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read mined.db without modifying it and export fenced-code "
            "language statistics."
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
        "--raw-output",
        default="fenced_languages_raw.csv",
        help="raw fence-label CSV output",
    )
    parser.add_argument(
        "--normalized-output",
        default="fenced_languages_normalized.csv",
        help="normalized language-family CSV output",
    )
    parser.add_argument(
        "--summary-output",
        default="fenced_languages_summary.json",
        help="machine-readable JSON summary output",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=20,
        help="number of raw and normalized rows to print (default: 20)",
    )
    return parser


def _print_stats(title: str, stats: Sequence[FenceLanguageStat], top: int) -> None:
    print(title)
    for rank, item in enumerate(stats[:top], 1):
        print(
            f"  {rank:>3}. {item.label:20s} "
            f"{item.blocks:>8,} blocks in {item.files:>7,} files"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, raw_stats, normalized_stats = analyze_fenced_languages(
            args.db,
            scope=args.scope,
        )
        raw_path = write_stats_csv(
            raw_stats,
            args.raw_output,
            total_blocks=summary.total_code_blocks,
            files_with_code_blocks=summary.files_with_code_blocks,
            scoped_files=summary.scoped_files,
        )
        normalized_path = write_stats_csv(
            normalized_stats,
            args.normalized_output,
            total_blocks=summary.total_code_blocks,
            files_with_code_blocks=summary.files_with_code_blocks,
            scoped_files=summary.scoped_files,
        )
        summary_path = write_summary_json(summary, args.summary_output)
    except (FileNotFoundError, json.JSONDecodeError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Scope: {summary.scope} ({summary.scope_definition})")
    print(f"Scoped files: {summary.scoped_files:,}")
    print(
        f"Files with fenced blocks: {summary.files_with_code_blocks:,} "
        f"({summary.files_with_code_block_share:.2%})"
    )
    print(f"Total fenced blocks: {summary.total_code_blocks:,}")
    print(
        f"Untagged blocks: {summary.untagged_code_blocks:,} "
        f"({summary.untagged_code_block_share:.2%})"
    )
    _print_stats("Raw labels:", raw_stats, args.top)
    _print_stats("Normalized language families:", normalized_stats, args.top)
    print(f"Raw CSV: {raw_path.resolve()}")
    print(f"Normalized CSV: {normalized_path.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
