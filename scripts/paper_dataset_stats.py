"""Reproduce the paper's dataset, repository, and exact-file size tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


EXACT_PREDICATE = "(lower(f.path) = 'claude.md' OR lower(f.path) LIKE '%/claude.md')"
STRICT_PREDICATE = "(f.path = 'CLAUDE.md' OR substr(f.path, -10) = '/CLAUDE.md')"


def describe(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot describe an empty sequence")
    ordered = sorted(values)
    count = len(ordered)

    def nearest_rank(probability: float) -> int:
        return ordered[max(0, min(count - 1, math.ceil(probability * count) - 1))]

    return {
        "count": count,
        "total": sum(ordered),
        "min": ordered[0],
        "p25": nearest_rank(0.25),
        "median": nearest_rank(0.50),
        "mean": sum(ordered) / count,
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
        "zero": sum(value == 0 for value in ordered),
    }


def open_read_only(path: Path) -> sqlite3.Connection:
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


def analyze_database(
    db_path: str | Path, *, chars_per_token: float = 3.5
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")

    path = Path(db_path)
    connection = open_read_only(path)
    try:
        repositories = list(
            connection.execute(
                "SELECT full_name, stars, language FROM repos "
                "ORDER BY lower(full_name), full_name"
            )
        )
        population = connection.execute(
            f"""
            SELECT
              count(*) AS files,
              count(DISTINCT repo_full_name) AS represented_repositories,
              sum(CASE WHEN {EXACT_PREDICATE} THEN 1 ELSE 0 END) AS exact_files,
              count(DISTINCT CASE WHEN {EXACT_PREDICATE}
                                  THEN repo_full_name END) AS exact_repositories,
              sum(CASE WHEN {STRICT_PREDICATE} THEN 1 ELSE 0 END) AS strict_files,
              count(DISTINCT CASE WHEN {STRICT_PREDICATE}
                                  THEN repo_full_name END) AS strict_repositories,
              sum(CASE WHEN NOT {EXACT_PREDICATE} THEN 1 ELSE 0 END) AS nonexact_files,
              count(DISTINCT CASE WHEN NOT {EXACT_PREDICATE}
                                  THEN repo_full_name END) AS nonexact_repositories
            FROM files AS f
            """
        ).fetchone()
        assert population is not None

        bytes_values: list[int] = []
        line_values: list[int] = []
        word_values: list[int] = []
        character_values: list[int] = []
        token_values: list[int] = []
        for row in connection.execute(
            f"""
            SELECT f.id, f.size_bytes, a.result_json
            FROM files AS f
            INNER JOIN analysis AS a ON a.file_id = f.id
            WHERE a.analyzer_id = 'structure' AND {EXACT_PREDICATE}
            ORDER BY f.id
            """
        ):
            result = json.loads(row["result_json"])
            try:
                size_bytes = int(row["size_bytes"])
                lines = int(result["line_count"])
                words = int(result["word_count"])
                characters = int(result["char_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"file {row['id']} has an incomplete structure result"
                ) from exc
            if min(size_bytes, lines, words, characters) < 0:
                raise ValueError(f"file {row['id']} has a negative size metric")
            bytes_values.append(size_bytes)
            line_values.append(lines)
            word_values.append(words)
            character_values.append(characters)
            token_values.append(math.ceil(characters / chars_per_token))
    finally:
        connection.close()

    exact_files = int(population["exact_files"])
    if len(bytes_values) != exact_files:
        raise ValueError(
            f"found {len(bytes_values)} structure rows for {exact_files} exact files"
        )

    language_counts = Counter(
        str(row["language"]) if row["language"] else "(unknown)"
        for row in repositories
    )
    repo_count = len(repositories)
    language_rows: list[dict[str, object]] = []
    for language, count in sorted(
        language_counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0])
    ):
        language_rows.append(
            {
                "rank": len(language_rows) + 1,
                "language": language,
                "repositories": count,
                "repository_share": count / repo_count,
            }
        )

    top_repositories = [
        {
            "rank": rank,
            "repository": str(row["full_name"]),
            "stars": int(row["stars"] or 0),
            "language": str(row["language"]) if row["language"] else "(unknown)",
        }
        for rank, row in enumerate(
            sorted(
                repositories,
                key=lambda row: (
                    -int(row["stars"] or 0),
                    str(row["full_name"]).casefold(),
                    str(row["full_name"]),
                ),
            )[:15],
            start=1,
        )
    ]
    star_values = [int(row["stars"] or 0) for row in repositories]

    summary: dict[str, object] = {
        "analysis": "paper dataset, repository, and exact-file size statistics",
        "database": {
            "path": str(path),
            "access": "SQLite mode=ro&immutable=1 with PRAGMA query_only=ON",
        },
        "definitions": {
            "paper_file_population": (
                "stored POSIX path has a case-insensitive basename equal to CLAUDE.md"
            ),
            "strict_case_population": "stored basename is exactly CLAUDE.md",
            "repository_language": "GitHub primary language stored in the repos table",
            "repository_stars": "stored GitHub stargazer count; missing values treated as zero",
            "quantiles": "empirical nearest-rank",
            "estimated_tokens": f"ceil(character_count / {chars_per_token}) per file",
        },
        "file_populations": {
            "all_query_matches": {
                "files": int(population["files"]),
                "repositories": int(population["represented_repositories"]),
            },
            "case_insensitive_exact_claude": {
                "files": exact_files,
                "repositories": int(population["exact_repositories"]),
            },
            "strict_case_exact_claude": {
                "files": int(population["strict_files"]),
                "repositories": int(population["strict_repositories"]),
            },
            "nonexact_query_matches": {
                "files": int(population["nonexact_files"]),
                "repositories": int(population["nonexact_repositories"]),
            },
        },
        "repositories": {
            "count": repo_count,
            "stars": describe(star_values),
            "languages_known": repo_count - language_counts.get("(unknown)", 0),
            "languages_unknown": language_counts.get("(unknown)", 0),
            "top_15_by_stars": top_repositories,
        },
        "exact_file_size": {
            "bytes": describe(bytes_values),
            "lines": describe(line_values),
            "words": describe(word_values),
            "characters": describe(character_values),
            "estimated_tokens": describe(token_values),
        },
    }
    return summary, language_rows, top_repositories


def write_json(value: object, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(rows: Sequence[dict[str, object]], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's dataset, repository, and size tables."
    )
    parser.add_argument("--db", default="data/mined.db", help="SQLite database")
    parser.add_argument(
        "--summary-output",
        default="results/paper_dataset_summary.json",
        help="machine-readable summary JSON",
    )
    parser.add_argument(
        "--languages-output",
        default="results/repository_languages.csv",
        help="one row per stored repository primary language",
    )
    parser.add_argument(
        "--top-repositories-output",
        default="results/top_repositories.csv",
        help="the fifteen repositories with the largest stored star count",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=3.5,
        help="character/token estimate used by the paper (default: 3.5)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, languages, top_repositories = analyze_database(
            args.db, chars_per_token=args.chars_per_token
        )
        write_json(summary, args.summary_output)
        write_csv(languages, args.languages_output)
        write_csv(top_repositories, args.top_repositories_output)
    except (FileNotFoundError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    populations = summary["file_populations"]
    exact = populations["case_insensitive_exact_claude"]
    repositories = summary["repositories"]
    size = summary["exact_file_size"]
    print(
        f"Exact CLAUDE.md: {exact['files']:,} files in "
        f"{exact['repositories']:,} repositories"
    )
    print(
        f"Stored repositories: {repositories['count']:,}; "
        f"median stars: {repositories['stars']['median']:,}"
    )
    print(
        f"Exact-file medians: {size['bytes']['median']:,} bytes, "
        f"{size['lines']['median']:,} lines, {size['words']['median']:,} words"
    )
    print(f"Summary JSON: {Path(args.summary_output).resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

