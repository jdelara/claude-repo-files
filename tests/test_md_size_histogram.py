from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.md_size_histogram import (
    _x_tick_positions,
    build_parser,
    load_markdown_sizes,
    save_histogram,
)


class MarkdownSizeQueryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       path TEXT NOT NULL,
                       tool_id TEXT NOT NULL,
                       size_bytes INTEGER
                   )"""
            )
            connection.executemany(
                "INSERT INTO files (path, tool_id, size_bytes) VALUES (?, ?, ?)",
                [
                    ("CLAUDE.md", "claude", 100),
                    ("docs/guide.MD", "claude", 200),
                    ("README.md", "other", 300),
                    ("source.py", "claude", 400),
                    ("missing.md", "claude", None),
                    ("invalid.md", "claude", -1),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_only_markdown_files_with_valid_sizes(self):
        self.assertCountEqual(load_markdown_sizes(self.db_path), [100, 200, 300])

    def test_can_filter_by_tool(self):
        self.assertCountEqual(
            load_markdown_sizes(self.db_path, tool="claude"),
            [100, 200],
        )


class HistogramImageTests(unittest.TestCase):
    def test_x_ticks_include_the_exact_maximum(self):
        ticks = _x_tick_positions(343_528, intervals=16)

        self.assertEqual(len(ticks), 17)
        self.assertEqual(ticks[0], 0)
        self.assertEqual(ticks[-1], 343_528)

    def test_defaults_use_fine_bins_and_dense_x_intervals(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.bins, 200)
        self.assertEqual(arguments.x_intervals, 16)

    def test_writes_a_png_image(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "histogram.png"

            returned_path = save_histogram(
                [10, 20, 20, 30],
                output,
                bins=3,
                x_intervals=3,
            )

            self.assertEqual(returned_path, output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
