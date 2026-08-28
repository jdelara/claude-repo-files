from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.exact_duplicate_families import (
    analyze_exact_duplicates,
    build_parser,
    write_family_csv,
    write_member_csv,
    write_summary_json,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ExactDuplicateFamilyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       repo_full_name TEXT NOT NULL,
                       path TEXT NOT NULL,
                       content_hash TEXT,
                       size_bytes INTEGER,
                       html_url TEXT,
                       fetched_at TEXT
                   )"""
            )
            hash_a = _hash("shared across repositories")
            hash_b = _hash("shared within one repository")
            hash_c = _hash("unique")
            connection.executemany(
                """INSERT INTO files
                   (repo_full_name, path, content_hash, size_bytes,
                    html_url, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    ("z/repo", "CLAUDE.md", hash_a, 26, "https://z", "2026-01-01"),
                    ("a/repo", "docs/guide.MD", hash_a, 26, "https://a", "2026-01-02"),
                    ("b/repo", "CLAUDE.md", hash_a, 26, "https://b", "2026-01-03"),
                    ("a/repo", "notes.txt", hash_a, 26, "https://txt", "2026-01-04"),
                    ("one/repo", "x.md", hash_b, 28, "https://x", "2026-01-05"),
                    ("one/repo", "y.md", hash_b, 28, "https://y", "2026-01-06"),
                    ("u/repo", "unique.md", hash_c, 6, "https://u", "2026-01-07"),
                    ("u/repo", "unhashed.md", None, 7, "https://n", "2026-01-08"),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_markdown_scope_groups_and_sorts_duplicate_families(self):
        summary, families = analyze_exact_duplicates(self.db_path)

        self.assertEqual(summary.scope, "markdown")
        self.assertEqual(summary.scoped_files, 7)
        self.assertEqual(summary.hashed_files, 6)
        self.assertEqual(summary.unhashed_files, 1)
        self.assertEqual(summary.distinct_content_hashes, 3)
        self.assertEqual(summary.duplicate_families, 2)
        self.assertEqual(summary.files_in_duplicate_families, 5)
        self.assertEqual(summary.repeated_instances, 3)
        self.assertEqual(summary.cross_repository_families, 1)
        self.assertEqual(summary.single_repository_families, 1)
        self.assertEqual(summary.largest_family_size, 3)
        self.assertEqual(summary.largest_family_repositories, 3)
        self.assertEqual([family.copies for family in families], [3, 2])
        self.assertEqual(families[0].representative.repo_full_name, "a/repo")
        self.assertEqual(families[0].representative.path, "docs/guide.MD")

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM files").fetchone()[0], 8)
        finally:
            connection.close()

    def test_exact_claude_scope_excludes_other_markdown_paths(self):
        summary, families = analyze_exact_duplicates(
            self.db_path,
            scope="exact-claude",
        )

        self.assertEqual(summary.scoped_files, 2)
        self.assertEqual(summary.duplicate_families, 1)
        self.assertEqual(summary.files_in_duplicate_families, 2)
        self.assertEqual(families[0].repositories, 2)

    def test_writes_deterministic_machine_readable_reports(self):
        summary, families = analyze_exact_duplicates(self.db_path)
        output_directory = Path(self.temporary_directory.name) / "reports"

        family_path = write_family_csv(families, output_directory / "families.csv")
        member_path = write_member_csv(families, output_directory / "members.csv")
        summary_path = write_summary_json(summary, output_directory / "summary.json")

        with family_path.open(encoding="utf-8", newline="") as handle:
            family_rows = list(csv.DictReader(handle))
        with member_path.open(encoding="utf-8", newline="") as handle:
            member_rows = list(csv.DictReader(handle))
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual([row["copies"] for row in family_rows], ["3", "2"])
        self.assertEqual(len(member_rows), 5)
        self.assertEqual(summary_data["duplicate_families"], 2)
        self.assertEqual(summary_data["largest_family_size"], 3)

    def test_parser_defaults_to_markdown_scope(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.scope, "markdown")
        self.assertEqual(arguments.top, 20)


if __name__ == "__main__":
    unittest.main()
