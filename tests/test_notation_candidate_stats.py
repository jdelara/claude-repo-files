from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from miner.analyzers.notation import NotationAnalyzer
from scripts.notation_candidate_stats import (
    analyze_notation_candidates,
    build_parser,
    write_mermaid_csv,
    write_sample_csv,
    write_summary,
)


class NotationCandidateStatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        contents = [
            (
                1,
                "a/repo",
                "CLAUDE.md",
                """# Guide
```mermaid
flowchart LR
A --> B
```
```mermaid
sequenceDiagram
A->>B: hello
```
```json
{"status": "ok"}
```
```html
<!-- explanatory comment -->
<div>content</div>
```
""",
            ),
            (
                2,
                "b/repo",
                "docs/CLAUDE.md",
                """```json
{"$schema": "example", "type": "object"}
```
```yaml
name: ordinary-configuration
```
```
root/
├── src/
└── tests/
```
""",
            ),
            (
                3,
                "c/repo",
                "notes.md",
                """```mermaid
pie title Languages
"Python" : 10
```
```mermaid
timeline
title Releases
```
""",
            ),
        ]
        analyzer = NotationAnalyzer()
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       id INTEGER PRIMARY KEY,
                       repo_full_name TEXT NOT NULL,
                       path TEXT NOT NULL,
                       content TEXT NOT NULL
                   )"""
            )
            connection.execute(
                """CREATE TABLE analysis (
                       file_id INTEGER NOT NULL,
                       analyzer_id TEXT NOT NULL,
                       result_json TEXT NOT NULL
                   )"""
            )
            connection.executemany("INSERT INTO files VALUES (?, ?, ?, ?)", contents)
            for file_id, repo, path, content in contents:
                result = analyzer.analyze(content, {"repo": repo, "path": path})
                connection.execute(
                    "INSERT INTO analysis VALUES (?, 'notation', ?)",
                    (file_id, json.dumps(result)),
                )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_replays_subtypes_and_distinguishes_schema_markers(self):
        summary, mermaid_rows, sample_rows = analyze_notation_candidates(
            self.db_path,
            scope="exact-claude",
            sample_per_category=2,
        )

        replay = summary["fenced_block_replay"]
        self.assertEqual(summary["scoped_files"], 2)
        self.assertEqual(
            summary["stored_results"]["files_with_high_explicit_diagram_dsl"],
            1,
        )
        self.assertEqual(replay["occurrences_by_notation"]["mermaid"], 2)
        self.assertEqual(replay["occurrences_by_notation"]["json_schema"], 2)
        self.assertEqual(replay["occurrences_by_notation"]["yaml_schema"], 1)
        self.assertEqual(replay["occurrences_by_notation"]["ascii_diagram"], 2)
        self.assertEqual(
            replay["json_configured_marker_evidence"],
            {"no_configured_schema_marker": 1, "schema_marker_present": 1},
        )
        self.assertEqual(
            replay["yaml_configured_marker_evidence"],
            {"no_configured_schema_marker": 1},
        )
        by_subtype = {row["subtype"]: row for row in mermaid_rows}
        self.assertEqual(by_subtype["flowchart"]["blocks"], 1)
        self.assertEqual(by_subtype["sequence"]["blocks"], 1)
        self.assertEqual(
            replay["ascii_arrow_diagnostics"][
                "arrow_only:html_comment_delimiters_only"
            ],
            1,
        )
        self.assertTrue(sample_rows)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM files").fetchone()[0], 3)
        finally:
            connection.close()

    def test_writes_machine_readable_outputs(self):
        summary, mermaid_rows, sample_rows = analyze_notation_candidates(
            self.db_path,
            sample_per_category=1,
        )
        summary_path = write_summary(
            summary, Path(self.temporary_directory.name) / "summary.json"
        )
        mermaid_path = write_mermaid_csv(
            mermaid_rows, Path(self.temporary_directory.name) / "mermaid.csv"
        )
        sample_path = write_sample_csv(
            sample_rows, Path(self.temporary_directory.name) / "sample.csv"
        )

        self.assertTrue(summary_path.is_file())
        self.assertTrue(mermaid_path.is_file())
        self.assertTrue(sample_path.is_file())
        self.assertEqual(
            summary["fenced_block_replay"][
                "mermaid_unclassified_first_directives"
            ]["timeline"],
            1,
        )

    def test_parser_defaults_to_all_query_matched_files(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.scope, "all")
        self.assertEqual(arguments.sample_per_category, 10)


if __name__ == "__main__":
    unittest.main()
