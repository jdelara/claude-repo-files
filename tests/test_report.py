from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.report import describe_counts, report


class CountDescriptionTests(unittest.TestCase):
    def test_uses_nearest_rank_quantiles_independent_of_input_order(self):
        result = describe_counts([100, 1, 10, 20])

        self.assertEqual(result["min"], 1)
        self.assertEqual(result["p25"], 1)
        self.assertEqual(result["median"], 10)
        self.assertEqual(result["p75"], 20)
        self.assertEqual(result["p90"], 100)
        self.assertEqual(result["max"], 100)
        self.assertEqual(result["mean"], 32.75)


class ReportStructureTests(unittest.TestCase):
    def test_reports_sorted_line_and_word_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            db_path = Path(temporary_directory) / "mined.db"
            connection = sqlite3.connect(db_path)
            try:
                connection.executescript(
                    """
                    CREATE TABLE repos (
                        full_name TEXT PRIMARY KEY,
                        stars INTEGER,
                        language TEXT
                    );
                    CREATE TABLE files (
                        id INTEGER PRIMARY KEY,
                        repo_full_name TEXT,
                        path TEXT,
                        tool_id TEXT,
                        size_bytes INTEGER,
                        html_url TEXT
                    );
                    CREATE TABLE analysis (
                        file_id INTEGER,
                        analyzer_id TEXT,
                        result_json TEXT
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO repos VALUES ('owner/repo', 10, 'Python')"
                )
                structures = [(100, 300), (1, 3), (10, 30)]
                for file_id, (lines, words) in enumerate(structures, 1):
                    connection.execute(
                        "INSERT INTO files VALUES (?, 'owner/repo', ?, 'claude', 10, '')",
                        (file_id, f"{file_id}.md"),
                    )
                    connection.execute(
                        "INSERT INTO analysis VALUES (?, 'structure', ?)",
                        (
                            file_id,
                            json.dumps(
                                {
                                    "line_count": lines,
                                    "word_count": words,
                                    "header_count": 0,
                                    "code_block_count": 0,
                                    "code_block_languages": {},
                                }
                            ),
                        ),
                    )
                connection.commit()
            finally:
                connection.close()

            output = io.StringIO()
            with redirect_stdout(output):
                report(str(db_path))

            text = output.getvalue()
            self.assertIn("median=10, mean=37", text)
            self.assertIn("median=30, mean=111", text)


if __name__ == "__main__":
    unittest.main()
