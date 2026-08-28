from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_reading_time_distribution import (
    build_parser,
    load_word_counts,
    nearest_rank,
    reading_time_survival,
    save_reading_time_distribution,
)


class ReadingTimeDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL
            );
            CREATE TABLE analysis (
                file_id INTEGER NOT NULL,
                analyzer_id TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            """
        )
        rows = [
            (1, "CLAUDE.md", 0),
            (2, "docs/claude.MD", 10),
            (3, "README.md", 20),
            (4, "CLAUDE.md.template", 30),
        ]
        for file_id, path, words in rows:
            connection.execute("INSERT INTO files VALUES (?, ?)", (file_id, path))
            connection.execute(
                "INSERT INTO analysis VALUES (?, 'structure', ?)",
                (file_id, json.dumps({"word_count": words})),
            )
        connection.execute(
            "INSERT INTO analysis VALUES (1, 'notation', '{\"word_count\": 999}')"
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_exact_and_all_populations_from_stored_structure_results(self) -> None:
        self.assertEqual(
            load_word_counts(self.db_path, scope="exact-claude"),
            [0, 10],
        )
        self.assertEqual(
            load_word_counts(self.db_path, scope="all-query-matched"),
            [0, 10, 20, 30],
        )
        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.total_changes, 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                4,
            )
        finally:
            connection.close()


class ReadingTimeCalculationTests(unittest.TestCase):
    def test_uses_nearest_rank_quantiles(self) -> None:
        values = [100, 1, 10, 20]

        self.assertEqual(nearest_rank(values, 0.50), 10)
        self.assertEqual(nearest_rank(values, 0.90), 100)

    def test_survival_curve_retains_zero_files_in_denominator(self) -> None:
        minutes, percentages = reading_time_survival(
            [0, 10, 20, 20, 30],
            words_per_minute=10,
        )

        self.assertEqual(minutes, [1.0, 2.0, 3.0])
        self.assertEqual(percentages, [80.0, 60.0, 20.0])


class ReadingTimeImageTests(unittest.TestCase):
    def test_defaults_target_exact_article_figure(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(arguments.output, "article/claude_reading_time_distribution.png")
        self.assertEqual(arguments.slow_wpm, 175)
        self.assertEqual(arguments.baseline_wpm, 238)
        self.assertEqual(arguments.fast_wpm, 300)

    def test_writes_png(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "reading-time.png"
            returned = save_reading_time_distribution(
                [0, 10, 20, 30, 100, 1000],
                output,
                dpi=72,
            )

            self.assertEqual(returned, output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
