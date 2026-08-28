from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_import_syntax import (
    aggregate_comparison,
    aggregate_strata,
    aggregate_targets,
    analyze_imports,
    build_summary,
    write_comparison_csv,
    write_files_csv,
    write_occurrences_csv,
    write_strata_csv,
    write_summary_json,
    write_targets_csv,
)


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClaudeImportCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.db_path = root / "sample.db"
        self.phase2_path = root / "phase2.csv"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE repos (
                full_name TEXT PRIMARY KEY,
                stars INTEGER,
                language TEXT
            );
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                repo_full_name TEXT NOT NULL,
                path TEXT NOT NULL,
                content TEXT,
                content_hash TEXT,
                size_bytes INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO repos(full_name, stars, language) VALUES (?, ?, ?)",
            [
                ("owner/alpha", 100, "Python"),
                ("owner/beta", 0, "TypeScript"),
            ],
        )
        alpha = """@AGENTS.md
`@literal.md`
<!-- import: @comment.md -->
Read @docs/rules.md before editing.
Follow docs/guide.md.
See @README and @package.json.
"""
        records = [
            (1, "owner/alpha", "CLAUDE.md", alpha),
            (2, "owner/alpha", ".claude/CLAUDE.md", "No imports here.\n"),
            (3, "owner/beta", "CLAUDE.md.template", "@ignored.md\n"),
        ]
        connection.executemany(
            """INSERT INTO files(
                   id, repo_full_name, path, content, content_hash, size_bytes
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (identifier, repo, path, content, _hash_text(content), len(content.encode("utf-8")))
                for identifier, repo, path, content in records
            ],
        )
        connection.commit()
        connection.close()

        fieldnames = [
            "reference_id",
            "repo_full_name",
            "source_path",
            "line_number",
            "intent_category",
            "syntax",
            "rule_id",
            "high_confidence_local_instructional",
            "normalized_target",
            "target_basename",
        ]
        rows = [
            ("REF-1", 1, "direct_inclusion", "direct_include", "explicit", "AGENTS.md", "agents.md"),
            ("REF-2", 2, "direct_inclusion", "path_mention", "standalone", "literal.md", "literal.md"),
            ("REF-3", 3, "direct_inclusion", "comment_include", "explicit", "comment.md", "comment.md"),
            ("REF-4", 4, "instructional_delegation", "path_mention", "required", "docs/rules.md", "rules.md"),
            ("REF-5", 5, "instructional_delegation", "path_mention", "follow", "docs/guide.md", "guide.md"),
        ]
        with self.phase2_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for reference_id, line, intent, syntax, rule, target, basename in rows:
                writer.writerow(
                    {
                        "reference_id": reference_id,
                        "repo_full_name": "owner/alpha",
                        "source_path": "CLAUDE.md",
                        "line_number": line,
                        "intent_category": intent,
                        "syntax": syntax,
                        "rule_id": rule,
                        "high_confidence_local_instructional": 1,
                        "normalized_target": target,
                        "target_basename": basename,
                    }
                )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_filters_population_and_compares_with_frozen_phase2(self):
        before = _hash_file(self.db_path)
        occurrences, files, changes, phase2_index = analyze_imports(
            self.db_path,
            phase2_occurrences_path=self.phase2_path,
        )
        after = _hash_file(self.db_path)

        self.assertEqual(changes, 0)
        self.assertEqual(before, after)
        self.assertEqual(len(files), 2)
        self.assertEqual({file.source_path for file in files}, {"CLAUDE.md", ".claude/CLAUDE.md"})
        hidden = next(file for file in files if file.source_path == ".claude/CLAUDE.md")
        self.assertEqual(hidden.source_logical_scope, "project_root")

        by_target = {row.raw_target: row for row in occurrences}
        self.assertEqual(by_target["AGENTS.md"].decision, "import_candidate")
        self.assertEqual(by_target["literal.md"].decision, "excluded_inline_code")
        self.assertEqual(by_target["comment.md"].decision, "excluded_html_comment")
        self.assertEqual(by_target["README"].target_extension_class, "extensionless")
        self.assertEqual(by_target["package.json"].target_extension_class, "other_extension")
        self.assertEqual(by_target["docs/rules.md"].phase2_intent_category, "instructional_delegation")

        comparison = aggregate_comparison(occurrences, phase2_index)
        totals: dict[str, int] = {}
        for row in comparison:
            totals[row.transition] = totals.get(row.transition, 0) + row.occurrences
        self.assertEqual(totals["phase2_direct_retained_as_import_candidate"], 1)
        self.assertEqual(totals["phase2_direct_reclassified_excluded_inline_code"], 1)
        self.assertEqual(totals["phase2_direct_reclassified_excluded_html_comment"], 1)
        self.assertEqual(totals["phase2_delegation_also_import_candidate"], 1)
        self.assertEqual(totals["phase2_instructional_delegation_without_phase2r_token"], 1)
        self.assertEqual(totals["import_candidate_without_phase2_high_confidence_match"], 2)

    def test_summary_and_artifacts_are_byte_deterministic(self):
        outputs: list[dict[str, Path]] = []
        for run in ("first", "second"):
            occurrences, files, changes, phase2_index = analyze_imports(
                self.db_path,
                phase2_occurrences_path=self.phase2_path,
            )
            targets = aggregate_targets(occurrences)
            strata = aggregate_strata(files)
            comparison = aggregate_comparison(occurrences, phase2_index)
            database_hash = _hash_file(self.db_path)
            summary = build_summary(
                occurrences,
                files,
                targets,
                strata,
                comparison,
                phase2_index=phase2_index,
                database_path=self.db_path,
                database_sha256_before=database_hash,
                database_sha256_after=database_hash,
                sqlite_total_changes=changes,
            )
            directory = Path(self.temporary_directory.name) / run
            paths = {
                "occurrences": write_occurrences_csv(occurrences, directory / "occurrences.csv"),
                "files": write_files_csv(files, directory / "files.csv"),
                "targets": write_targets_csv(targets, directory / "targets.csv"),
                "strata": write_strata_csv(strata, directory / "strata.csv"),
                "comparison": write_comparison_csv(comparison, directory / "comparison.csv"),
                "summary": write_summary_json(summary, directory / "summary.json"),
            }
            outputs.append(paths)

        for name in outputs[0]:
            self.assertEqual(_hash_file(outputs[0][name]), _hash_file(outputs[1][name]))

        summary = json.loads(outputs[0]["summary"].read_text(encoding="utf-8"))
        self.assertEqual(summary["population"]["files"], 2)
        self.assertEqual(summary["import_syntax_candidates"]["occurrences"], 4)
        self.assertTrue(summary["database"]["sha256_unchanged"])


if __name__ == "__main__":
    unittest.main()
