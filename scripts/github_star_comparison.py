"""Replay the paper's frozen GitHub star comparison without network access.

The committed input contains the two GitHub responses collected for the
study. This program joins that frozen snapshot against ``mined.db`` and
regenerates the derived CSV and JSON artifacts. It has no live-query mode.

SQLite is opened with ``mode=ro&immutable=1`` and ``PRAGMA query_only=ON``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ANALYSIS_VERSION = "github-star-comparison-v1"

GLOBAL_FIELDS = (
    "global_rank",
    "github_database_id",
    "github_node_id",
    "full_name",
    "stars",
    "primary_language",
    "fork",
    "archived",
    "disabled",
    "visibility",
    "default_branch",
    "html_url",
    "created_at",
    "updated_at",
    "pushed_at",
    "in_dataset_snapshot",
    "matched_dataset_full_name",
    "dataset_stored_stars",
    "dataset_stars_observed_at",
    "dataset_stored_files",
    "dataset_exact_claude_files",
    "dataset_root_exact_claude_files",
    "dataset_nested_exact_claude_files",
)

COMPARISON_FIELDS = (
    "dataset_stored_rank",
    "full_name",
    "dataset_stored_stars",
    "github_current_stars",
    "star_change",
    "global_rank_or_lower_bound",
    "within_global_return",
    "github_current_language",
    "dataset_stars_observed_at",
    "dataset_stored_files",
    "dataset_exact_claude_files",
    "dataset_root_exact_claude_files",
    "dataset_nested_exact_claude_files",
    "github_is_archived",
)

STORED_REFRESH_FIELDS = (
    "dataset_stored_rank",
    "dataset_full_name",
    "dataset_stored_stars",
    "dataset_stars_observed_at",
    "dataset_stored_language",
    "dataset_stored_files",
    "dataset_exact_claude_files",
    "dataset_root_exact_claude_files",
    "dataset_nested_exact_claude_files",
    "github_available",
    "github_database_id",
    "github_node_id",
    "github_current_full_name",
    "github_current_stars",
    "star_change",
    "github_current_language",
    "github_is_fork",
    "github_is_archived",
    "github_url",
    "global_rank",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def global_limit(value: str) -> int:
    parsed = positive_int(value)
    if parsed > 100:
        raise argparse.ArgumentTypeError("must be at most 100")
    return parsed


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _casefold(value: str) -> str:
    return value.casefold()


def load_dataset_repositories(
    db_path: str | Path,
    *,
    dataset_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load repository metadata and file counts without modifying SQLite."""
    connection = _readonly_connection(db_path)
    try:
        rows = connection.execute(
            """
            SELECT
                r.full_name,
                COALESCE(r.stars, 0) AS stored_stars,
                r.language AS stored_language,
                r.fetched_at AS stars_observed_at,
                COUNT(f.id) AS stored_files,
                COALESCE(SUM(
                    CASE WHEN lower(f.path) = 'claude.md'
                              OR lower(f.path) LIKE '%/claude.md'
                         THEN 1 ELSE 0 END
                ), 0) AS exact_claude_files,
                COALESCE(SUM(
                    CASE WHEN lower(f.path) = 'claude.md'
                         THEN 1 ELSE 0 END
                ), 0) AS root_exact_claude_files,
                COALESCE(SUM(
                    CASE WHEN lower(f.path) LIKE '%/claude.md'
                         THEN 1 ELSE 0 END
                ), 0) AS nested_exact_claude_files
            FROM repos AS r
            LEFT JOIN files AS f ON f.repo_full_name = r.full_name
            GROUP BY r.full_name, r.stars, r.language, r.fetched_at
            ORDER BY COALESCE(r.stars, 0) DESC,
                     lower(r.full_name), r.full_name
            """
        ).fetchall()
    finally:
        connection.close()

    repositories: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        repository = {
            "full_name": str(row["full_name"]),
            "stored_stars": int(row["stored_stars"]),
            "stored_language": row["stored_language"],
            "stars_observed_at": row["stars_observed_at"],
            "stored_files": int(row["stored_files"]),
            "exact_claude_files": int(row["exact_claude_files"]),
            "root_exact_claude_files": int(row["root_exact_claude_files"]),
            "nested_exact_claude_files": int(row["nested_exact_claude_files"]),
        }
        key = _casefold(repository["full_name"])
        if key in lookup:
            raise ValueError(
                "dataset contains repository names that differ only by case: "
                f"{lookup[key]['full_name']!r} and {repository['full_name']!r}"
            )
        lookup[key] = repository
        repositories.append(repository)

    selected = [dict(repository) for repository in repositories[:dataset_limit]]
    for rank, repository in enumerate(selected, start=1):
        repository["stored_rank"] = rank
    return selected, lookup


def _graphql_refresh_rows(
    raw_snapshot: Mapping[str, Any],
    selected_repositories: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    refresh = raw_snapshot["dataset_stored_top_refresh"]
    aliases = dict(refresh["request"]["aliases"])
    body = refresh["response"]
    data = body.get("data") or {}
    repository_by_name = {
        _casefold(str(repository["full_name"])): repository
        for repository in selected_repositories
    }
    current_name_aliases: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for alias, dataset_full_name in aliases.items():
        stored = repository_by_name[_casefold(dataset_full_name)]
        node = data.get(alias)
        current_name = node.get("nameWithOwner") if node else None
        if current_name:
            current_name_aliases[_casefold(str(current_name))] = dataset_full_name
        current_stars = (
            int(node.get("stargazerCount") or 0) if node is not None else None
        )
        language_node = node.get("primaryLanguage") if node else None
        rows.append(
            {
                "dataset_stored_rank": stored["stored_rank"],
                "dataset_full_name": dataset_full_name,
                "dataset_stored_stars": stored["stored_stars"],
                "dataset_stars_observed_at": stored["stars_observed_at"],
                "dataset_stored_language": stored["stored_language"],
                "dataset_stored_files": stored["stored_files"],
                "dataset_exact_claude_files": stored["exact_claude_files"],
                "dataset_root_exact_claude_files": stored[
                    "root_exact_claude_files"
                ],
                "dataset_nested_exact_claude_files": stored[
                    "nested_exact_claude_files"
                ],
                "github_available": node is not None,
                "github_database_id": node.get("databaseId") if node else None,
                "github_node_id": node.get("id") if node else None,
                "github_current_full_name": current_name,
                "github_current_stars": current_stars,
                "star_change": (
                    current_stars - int(stored["stored_stars"])
                    if current_stars is not None
                    else None
                ),
                "github_current_language": (
                    language_node.get("name") if language_node else None
                ),
                "github_is_fork": node.get("isFork") if node else None,
                "github_is_archived": node.get("isArchived") if node else None,
                "github_url": node.get("url") if node else None,
                "global_rank": None,
            }
        )
    rows.sort(key=lambda row: int(row["dataset_stored_rank"]))
    return rows, current_name_aliases


def analyze_snapshot(
    raw_snapshot: Mapping[str, Any],
    *,
    selected_repositories: Sequence[Mapping[str, Any]],
    dataset_lookup: Mapping[str, Mapping[str, Any]],
    dataset_limit: int,
    global_limit: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Derive summary, global rows, current top rows, and refresh rows."""
    search_body = raw_snapshot["global_search"]["response"]
    items = list(search_body.get("items") or [])[:global_limit]
    if not items:
        raise ValueError("GitHub repository search returned no items")
    star_counts = [int(item.get("stargazers_count") or 0) for item in items]
    if any(left < right for left, right in zip(star_counts, star_counts[1:])):
        raise ValueError("global search response is not ordered by decreasing stars")
    if any(bool(item.get("fork")) for item in items):
        raise ValueError("global search response contains a fork despite fork:false")
    if any(item.get("visibility") != "public" for item in items):
        raise ValueError("global search response contains a non-public repository")

    stored_refresh, renamed_aliases = _graphql_refresh_rows(
        raw_snapshot,
        selected_repositories,
    )
    global_rows: list[dict[str, Any]] = []
    global_rank_by_name: dict[str, int] = {}
    for rank, item in enumerate(items, start=1):
        full_name = str(item["full_name"])
        key = _casefold(full_name)
        global_rank_by_name[key] = rank
        matched_name = None
        dataset_repository = dataset_lookup.get(key)
        if dataset_repository is None and key in renamed_aliases:
            matched_name = renamed_aliases[key]
            dataset_repository = dataset_lookup.get(_casefold(matched_name))
        elif dataset_repository is not None:
            matched_name = str(dataset_repository["full_name"])

        global_rows.append(
            {
                "global_rank": rank,
                "github_database_id": item.get("id"),
                "github_node_id": item.get("node_id"),
                "full_name": full_name,
                "stars": int(item.get("stargazers_count") or 0),
                "primary_language": item.get("language"),
                "fork": bool(item.get("fork")),
                "archived": bool(item.get("archived")),
                "disabled": bool(item.get("disabled")),
                "visibility": item.get("visibility"),
                "default_branch": item.get("default_branch"),
                "html_url": item.get("html_url"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "pushed_at": item.get("pushed_at"),
                "in_dataset_snapshot": dataset_repository is not None,
                "matched_dataset_full_name": matched_name,
                "dataset_stored_stars": (
                    dataset_repository["stored_stars"]
                    if dataset_repository is not None
                    else None
                ),
                "dataset_stars_observed_at": (
                    dataset_repository["stars_observed_at"]
                    if dataset_repository is not None
                    else None
                ),
                "dataset_stored_files": (
                    dataset_repository["stored_files"]
                    if dataset_repository is not None
                    else None
                ),
                "dataset_exact_claude_files": (
                    dataset_repository["exact_claude_files"]
                    if dataset_repository is not None
                    else None
                ),
                "dataset_root_exact_claude_files": (
                    dataset_repository["root_exact_claude_files"]
                    if dataset_repository is not None
                    else None
                ),
                "dataset_nested_exact_claude_files": (
                    dataset_repository["nested_exact_claude_files"]
                    if dataset_repository is not None
                    else None
                ),
            }
        )

    for row in stored_refresh:
        current_name = row["github_current_full_name"] or row["dataset_full_name"]
        row["global_rank"] = global_rank_by_name.get(_casefold(str(current_name)))

    represented_global_rows = [
        row for row in global_rows if row["in_dataset_snapshot"]
    ]
    possible_current_rows: list[dict[str, Any]] = []
    for current_rank, global_row in enumerate(
        represented_global_rows[:dataset_limit],
        start=1,
    ):
        stored_stars = int(global_row["dataset_stored_stars"])
        possible_current_rows.append(
            {
                "dataset_current_rank": current_rank,
                "global_rank": global_row["global_rank"],
                "full_name": global_row["full_name"],
                "stars": global_row["stars"],
                "primary_language": global_row["primary_language"],
                "dataset_stored_stars": stored_stars,
                "star_change_since_dataset_observation": (
                    int(global_row["stars"]) - stored_stars
                ),
                "dataset_stars_observed_at": global_row[
                    "dataset_stars_observed_at"
                ],
                "dataset_stored_files": global_row["dataset_stored_files"],
                "dataset_exact_claude_files": global_row[
                    "dataset_exact_claude_files"
                ],
                "dataset_root_exact_claude_files": global_row[
                    "dataset_root_exact_claude_files"
                ],
                "dataset_nested_exact_claude_files": global_row[
                    "dataset_nested_exact_claude_files"
                ],
            }
        )

    current_top_established = len(represented_global_rows) >= dataset_limit
    current_rows = possible_current_rows if current_top_established else []
    comparison_rows: list[dict[str, Any]] = []
    for row in stored_refresh:
        global_rank = row["global_rank"]
        comparison_rows.append(
            {
                "dataset_stored_rank": row["dataset_stored_rank"],
                "full_name": row["github_current_full_name"]
                or row["dataset_full_name"],
                "dataset_stored_stars": row["dataset_stored_stars"],
                "github_current_stars": row["github_current_stars"],
                "star_change": row["star_change"],
                "global_rank_or_lower_bound": (
                    global_rank if global_rank is not None else f">{global_limit}"
                ),
                "within_global_return": global_rank is not None,
                "github_current_language": row["github_current_language"],
                "dataset_stars_observed_at": row[
                    "dataset_stars_observed_at"
                ],
                "dataset_stored_files": row["dataset_stored_files"],
                "dataset_exact_claude_files": row[
                    "dataset_exact_claude_files"
                ],
                "dataset_root_exact_claude_files": row[
                    "dataset_root_exact_claude_files"
                ],
                "dataset_nested_exact_claude_files": row[
                    "dataset_nested_exact_claude_files"
                ],
                "github_is_archived": row["github_is_archived"],
            }
        )

    stored_names = {
        _casefold(str(row["github_current_full_name"] or row["dataset_full_name"]))
        for row in stored_refresh
        if row["github_available"]
    }
    current_names = {_casefold(str(row["full_name"])) for row in current_rows}
    archived_count = sum(bool(row["archived"]) for row in global_rows)
    band_rows: list[dict[str, Any]] = []
    for boundary in (10, 25, 50, 100):
        if boundary > len(global_rows):
            continue
        represented = sum(
            bool(row["in_dataset_snapshot"])
            for row in global_rows
            if int(row["global_rank"]) <= boundary
        )
        band_rows.append(
            {
                "global_top_n": boundary,
                "represented_in_dataset": represented,
                "representation_share": represented / boundary,
            }
        )

    graphql_body = raw_snapshot["dataset_stored_top_refresh"]["response"]
    rate_limit = (graphql_body.get("data") or {}).get("rateLimit")
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "collection": raw_snapshot["collection"],
        "definitions": {
            "global_population": (
                "public GitHub repositories returned for stars:>0 fork:false, "
                "sorted by stars descending; archived repositories retained"
            ),
            "dataset_population": "all repository records in the immutable mined.db snapshot",
            "join": (
                "case-insensitive full_name, augmented by current names returned "
                "for the stored top repositories"
            ),
            "stars": (
                "GitHub stargazer count observed when the frozen responses "
                "were collected"
            ),
        },
        "global_search": {
            "query": raw_snapshot["global_search"]["request"]["parameters"]["q"],
            "requested_repositories": global_limit,
            "returned_repositories": len(global_rows),
            "reported_total_count": search_body.get("total_count"),
            "incomplete_results": bool(search_body.get("incomplete_results")),
            "highest_stars": int(global_rows[0]["stars"]),
            "lowest_returned_stars": int(global_rows[-1]["stars"]),
            "archived_repositories": archived_count,
        },
        "dataset": {
            "repositories": len(dataset_lookup),
            "stored_top_limit": dataset_limit,
            "represented_within_global_return": len(represented_global_rows),
            "stored_top_within_global_return": sum(
                row["global_rank"] is not None for row in stored_refresh
            ),
            "current_top_established_within_global_return": current_top_established,
            "current_top_rows_returned": len(current_rows),
            "reason_current_top_not_established": (
                None
                if current_top_established
                else (
                    f"Only {len(represented_global_rows)} represented repositories "
                    f"occur within the global top {len(global_rows)}; at least "
                    f"{dataset_limit} are required."
                )
            ),
            "stored_top_still_in_current_top": (
                len(stored_names & current_names)
                if current_top_established
                else None
            ),
            "entered_current_top": (
                sorted(current_names - stored_names)
                if current_top_established
                else None
            ),
            "left_current_top": (
                sorted(stored_names - current_names)
                if current_top_established
                else None
            ),
        },
        "representation_by_global_rank": band_rows,
        "graphql": {
            "errors": graphql_body.get("errors", []),
            "rate_limit": rate_limit,
        },
        "represented_repositories_within_global_return": possible_current_rows,
        "current_represented_top_if_established": (
            current_rows if current_top_established else None
        ),
        "stored_top_global_comparison": comparison_rows,
        "stored_top_refresh": stored_refresh,
        "limitations": [
            "The comparison is a timestamped snapshot; star counts and ranks change.",
            "Private repositories are outside the observable global population.",
            "Repository full-name joins can miss repositories renamed outside the stored-top refresh.",
            "Absence from the mined dataset does not establish absence of CLAUDE.md because acquisition coverage is incomplete.",
            "Stars measure expressed GitHub interest, not instruction-file quality or usage.",
        ],
    }
    return summary, global_rows, comparison_rows, stored_refresh


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(value: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/mined.db")
    parser.add_argument("--dataset-limit", type=positive_int, default=15)
    parser.add_argument("--global-limit", type=global_limit, default=100)
    parser.add_argument(
        "--raw-input",
        default="inputs/github_star_comparison_raw.json",
        help="frozen GitHub responses supplied with this package",
    )
    parser.add_argument(
        "--global-output",
        default="results/github_global_top100_snapshot.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="results/github_dataset_top15_global_comparison.csv",
    )
    parser.add_argument(
        "--stored-refresh-output",
        default="results/github_dataset_stored_top15_refresh.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/github_star_comparison_summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.dataset_limit > arguments.global_limit:
        print(
            "Error: --dataset-limit cannot exceed --global-limit.",
            file=sys.stderr,
        )
        return 2

    database_sha256_before = _sha256_file(arguments.db)
    selected, lookup = load_dataset_repositories(
        arguments.db,
        dataset_limit=arguments.dataset_limit,
    )

    with Path(arguments.raw_input).open(encoding="utf-8") as handle:
        raw_snapshot = json.load(handle)
    if raw_snapshot.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError(
            "raw snapshot analysis version does not match this script: "
            f"{raw_snapshot.get('analysis_version')!r}"
        )

    summary, global_rows, comparison_rows, stored_refresh = analyze_snapshot(
        raw_snapshot,
        selected_repositories=selected,
        dataset_lookup=lookup,
        dataset_limit=arguments.dataset_limit,
        global_limit=arguments.global_limit,
    )
    database_sha256_after = _sha256_file(arguments.db)
    summary["database"] = {
        "path": str(Path(arguments.db)),
        "sha256_before": database_sha256_before,
        "sha256_after": database_sha256_after,
        "unchanged": database_sha256_before == database_sha256_after,
        "access": "mode=ro&immutable=1; PRAGMA query_only=ON",
    }
    if not summary["database"]["unchanged"]:
        raise RuntimeError("database hash changed during analysis")

    write_csv(global_rows, GLOBAL_FIELDS, arguments.global_output)
    write_csv(comparison_rows, COMPARISON_FIELDS, arguments.comparison_output)
    write_csv(stored_refresh, STORED_REFRESH_FIELDS, arguments.stored_refresh_output)
    write_json(summary, arguments.summary_output)

    print("Mode: frozen-input replay; GitHub requests made: 0")
    print(
        "Global repositories returned: "
        f"{summary['global_search']['returned_repositories']}"
    )
    print(
        "Represented in dataset: "
        f"{summary['dataset']['represented_within_global_return']}"
    )
    print(
        "Current represented top established: "
        f"{summary['dataset']['current_top_established_within_global_return']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
