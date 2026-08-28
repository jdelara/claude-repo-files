from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.mined_claude_size_histogram import (
    build_parser,
    load_claude_sizes,
    save_histogram,
)


class ClaudeSizeQueryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       path TEXT NOT NULL,
                       size_bytes INTEGER
                   )"""
            )
            connection.executemany(
                "INSERT INTO files (path, size_bytes) VALUES (?, ?)",
                [
                    ("CLAUDE.md", 100),
                    ("docs/CLAUDE.md", 200),
                    ("nested/claude.MD", 300),
                    ("README.md", 400),
                    ("CLAUDE.md.template", 500),
                    ("CLAUDE.md", None),
                    ("CLAUDE.md", -1),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_only_claude_md_basenames_with_valid_sizes(self):
        self.assertCountEqual(load_claude_sizes(self.db_path), [100, 200, 300])


class HistogramImageTests(unittest.TestCase):
    def test_defaults_target_mined_database_and_png(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.output, "claude_size_histogram.png")
        self.assertEqual(arguments.bins, 100)

    def test_writes_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "histogram.png"

            returned_path = save_histogram([10, 20, 20, 30], output, bins=3)

            self.assertEqual(returned_path, output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
