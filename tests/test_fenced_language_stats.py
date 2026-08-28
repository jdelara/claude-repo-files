from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.fenced_language_stats import (
    analyze_fenced_languages,
    build_parser,
    normalize_fence_label,
    write_stats_csv,
    write_summary_json,
)


class FencedLanguageStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        try:
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
            files = [
                (1, "CLAUDE.md"),
                (2, "docs/claude.MD"),
                (3, "CLAUDE.md.template"),
                (4, "README.md"),
            ]
            connection.executemany("INSERT INTO files VALUES (?, ?)", files)
            structures = {
                1: {"code_block_count": 4, "code_block_languages": {"bash": 2, "sh": 1, "(untagged)": 1}},
                2: {"code_block_count": 3, "code_block_languages": {"ts": 1, "tsx": 2}},
                3: {"code_block_count": 5, "code_block_languages": {"python": 5}},
                4: {"code_block_count": 1, "code_block_languages": {"json": 1}},
            }
            for file_id, result in structures.items():
                connection.execute(
                    "INSERT INTO analysis VALUES (?, 'structure', ?)",
                    (file_id, json.dumps(result)),
                )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_exact_claude_scope_and_alias_normalization(self):
        summary, raw_stats, normalized_stats = analyze_fenced_languages(self.db_path)

        self.assertEqual(summary.scoped_files, 2)
        self.assertEqual(summary.analyzed_files, 2)
        self.assertEqual(summary.files_with_code_blocks, 2)
        self.assertEqual(summary.total_code_blocks, 7)
        self.assertEqual(summary.untagged_code_blocks, 1)
        self.assertEqual(summary.block_count_mismatches, 0)
        self.assertEqual(
            {item.label: item.blocks for item in raw_stats},
            {"bash": 2, "sh": 1, "(untagged)": 1, "ts": 1, "tsx": 2},
        )
        normalized = {item.label: item for item in normalized_stats}
        self.assertEqual(normalized["Shell"].blocks, 3)
        self.assertEqual(normalized["Shell"].files, 1)
        self.assertEqual(normalized["TypeScript"].blocks, 3)
        self.assertEqual(normalized["TypeScript"].files, 1)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM files").fetchone()[0], 4)
        finally:
            connection.close()

    def test_common_aliases_are_normalized_conservatively(self):
        self.assertEqual(normalize_fence_label("BASH"), "Shell")
        self.assertEqual(normalize_fence_label("tsx"), "TypeScript")
        self.assertEqual(normalize_fence_label("jsonc"), "JSON")
        self.assertEqual(normalize_fence_label("custom-label"), "custom-label")

    def test_writes_csv_and_json_outputs(self):
        summary, raw_stats, _normalized_stats = analyze_fenced_languages(self.db_path)
        output_directory = Path(self.temporary_directory.name) / "reports"
        csv_path = write_stats_csv(
            raw_stats,
            output_directory / "raw.csv",
            total_blocks=summary.total_code_blocks,
            files_with_code_blocks=summary.files_with_code_blocks,
            scoped_files=summary.scoped_files,
        )
        json_path = write_summary_json(summary, output_directory / "summary.json")

        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        data = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["label"], "bash")
        self.assertEqual(rows[0]["blocks"], "2")
        self.assertEqual(data["total_code_blocks"], 7)

    def test_parser_defaults_to_exact_claude(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(arguments.top, 20)


if __name__ == "__main__":
    unittest.main()
