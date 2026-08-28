from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.github_star_comparison import (
    analyze_snapshot,
    build_parser,
    load_dataset_repositories,
)


def repository_item(name: str, stars: int, database_id: int) -> dict:
    return {
        "id": database_id,
        "node_id": f"node-{database_id}",
        "full_name": name,
        "stargazers_count": stars,
        "language": "Python",
        "fork": False,
        "archived": False,
        "disabled": False,
        "visibility": "public",
        "default_branch": "main",
        "html_url": f"https://github.com/{name}",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2026-07-27T00:00:00Z",
        "pushed_at": "2026-07-27T00:00:00Z",
    }


class GitHubStarComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """CREATE TABLE repos (
                   full_name TEXT PRIMARY KEY,
                   stars INTEGER,
                   language TEXT,
                   fetched_at TEXT
               )"""
        )
        connection.execute(
            """CREATE TABLE files (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   repo_full_name TEXT,
                   path TEXT
               )"""
        )
        connection.executemany(
            "INSERT INTO repos VALUES (?, ?, ?, ?)",
            [
                ("org/alpha", 100, "Python", "2026-07-20T00:00:00Z"),
                ("org/beta", 90, "Python", "2026-07-20T00:00:00Z"),
                ("org/gamma", 80, "Python", "2026-07-20T00:00:00Z"),
                ("org/delta", 70, "Python", "2026-07-20T00:00:00Z"),
            ],
        )
        connection.executemany(
            "INSERT INTO files(repo_full_name, path) VALUES (?, ?)",
            [
                ("org/alpha", "CLAUDE.md"),
                ("org/alpha", "docs/CLAUDE.md"),
                ("org/beta", "CLAUDE.md"),
                ("org/gamma", "CLAUDE.md.template"),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _fixture(self):
        selected, lookup = load_dataset_repositories(
            self.db_path,
            dataset_limit=2,
        )
        graphql = {
            "data": {
                "r0": {
                    "id": "alpha-node",
                    "databaseId": 10,
                    "nameWithOwner": "org/alpha",
                    "url": "https://github.com/org/alpha",
                    "stargazerCount": 105,
                    "isFork": False,
                    "isArchived": False,
                    "primaryLanguage": {"name": "Python"},
                },
                "r1": {
                    "id": "beta-node",
                    "databaseId": 11,
                    "nameWithOwner": "org/beta",
                    "url": "https://github.com/org/beta",
                    "stargazerCount": 120,
                    "isFork": False,
                    "isArchived": False,
                    "primaryLanguage": {"name": "Python"},
                },
                "rateLimit": {
                    "cost": 1,
                    "remaining": 4998,
                    "resetAt": "2026-07-27T18:00:00Z",
                },
            }
        }
        search = {
            "total_count": 1_000_000,
            "incomplete_results": False,
            "items": [
                repository_item("org/beta", 120, 11),
                repository_item("org/gamma", 110, 12),
                repository_item("org/alpha", 105, 10),
            ],
        }
        aliases = {"r0": "org/alpha", "r1": "org/beta"}
        raw = {
            "analysis_version": "github-star-comparison-v1",
            "collection": {
                "mode": "live",
                "started_at_utc": "2026-07-27T17:00:00+00:00",
                "ended_at_utc": "2026-07-27T17:00:01+00:00",
                "github_api_version": "2026-03-10",
                "request_count": 2,
                "definition": "fixture",
            },
            "global_search": {
                "request": {
                    "parameters": {"q": "stars:>0 fork:false"},
                },
                "response": search,
            },
            "dataset_stored_top_refresh": {
                "request": {"aliases": aliases},
                "response": graphql,
            },
        }
        return selected, lookup, search, graphql, raw

    def test_derives_current_top_from_global_join(self) -> None:
        selected, lookup, _search, _graphql, raw = self._fixture()
        summary, global_rows, comparison_rows, refresh_rows = analyze_snapshot(
            raw,
            selected_repositories=selected,
            dataset_lookup=lookup,
            dataset_limit=2,
            global_limit=3,
        )

        self.assertEqual(
            [
                row["full_name"]
                for row in summary["current_represented_top_if_established"]
            ],
            ["org/beta", "org/gamma"],
        )
        self.assertTrue(
            summary["dataset"]["current_top_established_within_global_return"]
        )
        self.assertEqual(summary["dataset"]["stored_top_still_in_current_top"], 1)
        self.assertEqual(summary["dataset"]["entered_current_top"], ["org/gamma"])
        self.assertEqual(summary["dataset"]["left_current_top"], ["org/alpha"])
        self.assertEqual(global_rows[0]["dataset_exact_claude_files"], 1)
        self.assertEqual(global_rows[2]["dataset_exact_claude_files"], 2)
        self.assertEqual(
            [row["full_name"] for row in comparison_rows],
            ["org/alpha", "org/beta"],
        )
        self.assertEqual(comparison_rows[0]["global_rank_or_lower_bound"], 3)
        self.assertEqual(refresh_rows[0]["star_change"], 5)
        self.assertEqual(refresh_rows[1]["global_rank"], 1)

    def test_command_line_is_frozen_input_only(self) -> None:
        arguments = build_parser().parse_args([])
        self.assertEqual(arguments.db, "data/mined.db")
        self.assertEqual(
            arguments.raw_input,
            "inputs/github_star_comparison_raw.json",
        )
        self.assertFalse(hasattr(arguments, "token"))
        self.assertFalse(hasattr(arguments, "query"))


if __name__ == "__main__":
    unittest.main()
