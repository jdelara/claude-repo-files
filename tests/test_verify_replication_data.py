from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.verify_replication_data import database_counts, file_sha256


class VerifyReplicationDataTests(unittest.TestCase):
    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"replication data")
            self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_database_counts_case_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE repos (full_name TEXT PRIMARY KEY);
                CREATE TABLE files (
                    id INTEGER PRIMARY KEY,
                    repo_full_name TEXT,
                    path TEXT
                );
                CREATE TABLE analysis (id INTEGER PRIMARY KEY);
                INSERT INTO repos VALUES ('one/a'), ('two/b'), ('three/c');
                INSERT INTO files VALUES
                    (1, 'one/a', 'CLAUDE.md'),
                    (2, 'one/a', 'docs/claude.md'),
                    (3, 'two/b', 'CLAUDE.md.template'),
                    (4, 'three/c', 'nested/CLAUDE.md');
                INSERT INTO analysis VALUES (1), (2);
                """
            )
            connection.commit()
            connection.close()

            counts = database_counts(path)
            self.assertEqual(counts["repositories"], 3)
            self.assertEqual(counts["files"], 4)
            self.assertEqual(counts["analysis rows"], 2)
            self.assertEqual(counts["case-insensitive exact CLAUDE.md files"], 3)
            self.assertEqual(
                counts["repositories with a case-insensitive exact CLAUDE.md"], 2
            )
            self.assertEqual(counts["non-exact query-matched files"], 1)
            self.assertEqual(
                counts["repositories represented by non-exact query matches"], 1
            )
            self.assertEqual(counts["strict-case exact CLAUDE.md files"], 2)
            self.assertEqual(counts["repositories with a strict-case exact CLAUDE.md"], 2)


if __name__ == "__main__":
    unittest.main()
