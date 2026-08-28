from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_size_percentile_figure import (
    SizeObservation,
    build_parser,
    empirical_cdf,
    estimate_tokens,
    load_observations,
    nearest_rank,
    percentile_profile,
    save_percentile_figure,
)


class SizePercentileDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL
            );
            CREATE TABLE analysis (
                file_id INTEGER NOT NULL,
                analyzer_id TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            """
        )
        rows = [
            (1, "CLAUDE.md", 100, 10, 20, 80),
            (2, "docs/claude.MD", 200, 20, 40, 160),
            (3, "README.md", 300, 30, 60, 240),
            (4, "CLAUDE.md.template", 400, 40, 80, 320),
        ]
        for file_id, path, size, lines, words, chars in rows:
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?)",
                (file_id, path, size),
            )
            result = {
                "line_count": lines,
                "word_count": words,
                "char_count": chars,
            }
            connection.execute(
                "INSERT INTO analysis VALUES (?, 'structure', ?)",
                (file_id, json.dumps(result)),
            )
        connection.execute(
            "INSERT INTO analysis VALUES (1, 'notation', ?)",
            (json.dumps({"line_count": 999, "word_count": 999, "char_count": 999}),),
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_exact_and_all_populations_without_writing(self) -> None:
        exact = load_observations(self.db_path, scope="exact-claude")
        all_rows = load_observations(self.db_path, scope="all-query-matched")

        self.assertEqual([item.file_id for item in exact], [1, 2])
        self.assertEqual([item.file_id for item in all_rows], [1, 2, 3, 4])
        self.assertEqual(exact[1].word_count, 40)
        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.total_changes, 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                4,
            )
        finally:
            connection.close()


class SizePercentileCalculationTests(unittest.TestCase):
    def test_nearest_rank_and_profile(self) -> None:
        values = [100, 1, 10, 20]

        self.assertEqual(nearest_rank(values, 0.50), 10)
        self.assertEqual(nearest_rank(values, 0.90), 100)
        self.assertEqual(
            percentile_profile(values),
            {"median": 10, "p90": 100, "p99": 100},
        )

    def test_token_estimate_rounds_up(self) -> None:
        self.assertEqual(estimate_tokens(10, chars_per_token=3.5), 3)
        self.assertEqual(
            estimate_tokens(10, chars_per_token=3.5, multiplier=1.3),
            4,
        )

    def test_empirical_cdf_collapses_ties_and_reaches_100_percent(self) -> None:
        x_values, percentages = empirical_cdf([4, 1, 2, 2])

        self.assertEqual(x_values, [1, 2, 4])
        self.assertEqual(percentages, [25.0, 75.0, 100.0])


class SizePercentileFigureTests(unittest.TestCase):
    @staticmethod
    def observations() -> list[SizeObservation]:
        return [
            SizeObservation(
                file_id=index,
                path=f"d{index}/CLAUDE.md",
                size_bytes=value * 10,
                line_count=value,
                word_count=value * 5,
                char_count=value * 20,
            )
            for index, value in enumerate(
                [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560],
                start=1,
            )
        ]

    def test_defaults_target_vector_pdf(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(
            arguments.output,
            "output/pdf/claude_size_percentile_figure.pdf",
        )
        self.assertIsNone(arguments.preview_output)

    def test_writes_deterministic_vector_pdf_and_png_preview(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_pdf = directory / "first.pdf"
            second_pdf = directory / "second.pdf"
            preview = directory / "preview.png"

            save_percentile_figure(
                self.observations(),
                first_pdf,
                preview_output=preview,
                preview_dpi=72,
            )
            save_percentile_figure(self.observations(), second_pdf)

            first_bytes = first_pdf.read_bytes()
            self.assertTrue(first_bytes.startswith(b"%PDF-"))
            self.assertNotIn(b"/Subtype /Image", first_bytes)
            self.assertEqual(first_bytes, second_pdf.read_bytes())
            self.assertTrue(preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
