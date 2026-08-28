from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_context_window_stats import (
    FileMetrics,
    build_parser,
    build_summary,
    estimate_tokens,
    load_file_metrics,
    nearest_rank,
    write_summary,
)


class ContextWindowDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                repo_full_name TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER
            );
            CREATE TABLE analysis (
                file_id INTEGER NOT NULL,
                analyzer_id TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            """
        )
        rows = [
            (1, "org/one", "CLAUDE.md", 7, 7, 2),
            (2, "org/two", "docs/claude.MD", 700, 700, 200),
            (3, "org/three", "README.md", 3_500, 3_500, 201),
            (4, "org/four", "CLAUDE.md.template", 70, 70, 10),
        ]
        for file_id, repo, path, size, chars, lines in rows:
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?)",
                (file_id, repo, path, size),
            )
            connection.execute(
                "INSERT INTO analysis VALUES (?, 'structure', ?)",
                (
                    file_id,
                    json.dumps({"char_count": chars, "line_count": lines}),
                ),
            )
        connection.execute(
            "INSERT INTO analysis VALUES (1, 'notation', ?) ",
            (json.dumps({"char_count": 999, "line_count": 999}),),
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_exact_and_all_scopes_from_structure_results(self) -> None:
        exact = load_file_metrics(self.db_path, scope="exact-claude")
        all_rows = load_file_metrics(self.db_path, scope="all-query-matched")

        self.assertEqual([item.file_id for item in exact], [1, 2])
        self.assertEqual([item.file_id for item in all_rows], [1, 2, 3, 4])
        self.assertEqual(exact[1].char_count, 700)
        self.assertEqual(exact[1].line_count, 200)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.total_changes, 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                4,
            )
        finally:
            connection.close()


class ContextWindowCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = [
            FileMetrics(1, "org/a", "CLAUDE.md", 7, 7, 2),
            FileMetrics(2, "org/b", "CLAUDE.md", 700, 700, 200),
            FileMetrics(3, "org/c", "CLAUDE.md", 3_500, 3_500, 201),
        ]

    def test_estimates_tokens_with_ceil_and_multiplier(self) -> None:
        self.assertEqual(estimate_tokens(0), 0)
        self.assertEqual(estimate_tokens(1), 1)
        self.assertEqual(estimate_tokens(7), 2)
        self.assertEqual(estimate_tokens(7, multiplier=1.3), 3)

    def test_uses_nearest_rank_quantiles(self) -> None:
        self.assertEqual(nearest_rank([100, 1, 10, 20], 0.50), 10)
        self.assertEqual(nearest_rank([100, 1, 10, 20], 0.90), 100)

    def test_builds_line_and_context_statistics(self) -> None:
        summary = build_summary(
            self.metrics,
            context_windows=(1_000,),
            line_guideline=200,
        )

        distributions = summary["distributions"]
        self.assertEqual(distributions["baseline_estimated_tokens"]["maximum"], 1000)
        self.assertEqual(
            distributions["uplift_sensitivity_estimated_tokens"]["maximum"],
            1300,
        )
        occupancy = summary["context_occupancy"]["1000"]
        self.assertEqual(occupancy["baseline_files_at_or_above_nominal_window"], 1)
        self.assertEqual(occupancy["uplift_files_at_or_above_nominal_window"], 1)
        self.assertEqual(occupancy["baseline_percent"]["maximum"], 100.0)
        self.assertEqual(
            occupancy["uplift_sensitivity_percent"]["maximum"],
            130.0,
        )

        guideline = summary["line_guideline"]
        self.assertEqual(guideline["files_below"], 1)
        self.assertEqual(guideline["files_exactly_at"], 1)
        self.assertEqual(guideline["files_above"], 1)
        self.assertEqual(guideline["files_at_or_above"], 2)
        self.assertEqual(guideline["files_at_or_above_share"], 0.666667)

    def test_rejects_invalid_estimation_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "chars_per_token"):
            build_summary(self.metrics, chars_per_token=0)
        with self.assertRaisesRegex(ValueError, "token_uplift"):
            build_summary(self.metrics, token_uplift=-0.1)
        with self.assertRaisesRegex(ValueError, "token_uplift"):
            build_summary(self.metrics, token_uplift=float("nan"))
        with self.assertRaisesRegex(ValueError, "context windows"):
            build_summary(self.metrics, context_windows=(0,))


class ContextWindowOutputTests(unittest.TestCase):
    def test_defaults_target_article_summary(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(
            arguments.output,
            "article/claude_context_window_summary.json",
        )
        self.assertEqual(arguments.chars_per_token, 3.5)
        self.assertEqual(arguments.token_uplift, 0.30)
        self.assertEqual(arguments.context_windows, [200_000, 500_000, 1_000_000])
        self.assertEqual(arguments.line_guideline, 200)

    def test_writes_deterministic_json(self) -> None:
        metrics = [FileMetrics(1, "org/a", "CLAUDE.md", 7, 7, 2)]
        summary = build_summary(metrics)
        with tempfile.TemporaryDirectory() as temporary_directory:
            left = Path(temporary_directory) / "left.json"
            right = Path(temporary_directory) / "right.json"

            write_summary(summary, left)
            write_summary(summary, right)

            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(json.loads(left.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
