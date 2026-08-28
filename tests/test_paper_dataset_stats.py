from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.paper_dataset_stats import analyze_database, describe


class PaperDatasetStatsTests(unittest.TestCase):
    def test_nearest_rank_distribution(self) -> None:
        result = describe([10, 0, 3, 4])
        self.assertEqual(result["p25"], 0)
        self.assertEqual(result["median"], 3)
        self.assertEqual(result["p90"], 10)
        self.assertEqual(result["total"], 17)

    def test_uses_repositories_once_and_exact_paths_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.db"
            connection = sqlite3.connect(path)
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
                    size_bytes INTEGER
                );
                CREATE TABLE analysis (
                    file_id INTEGER,
                    analyzer_id TEXT,
                    result_json TEXT
                );
                INSERT INTO repos VALUES
                    ('one/a', 5, 'Python'),
                    ('two/b', 0, 'Python'),
                    ('three/c', 20, 'Rust');
                INSERT INTO files VALUES
                    (1, 'one/a', 'CLAUDE.md', 10),
                    (2, 'one/a', 'docs/claude.md', 20),
                    (3, 'two/b', 'CLAUDE.md.template', 30),
                    (4, 'three/c', 'other.md', 40);
                """
            )
            for file_id, lines, words, characters in [(1, 2, 3, 7), (2, 4, 5, 8)]:
                connection.execute(
                    "INSERT INTO analysis VALUES (?, 'structure', ?)",
                    (
                        file_id,
                        json.dumps(
                            {
                                "line_count": lines,
                                "word_count": words,
                                "char_count": characters,
                            }
                        ),
                    ),
                )
            connection.commit()
            connection.close()

            summary, languages, top = analyze_database(path)
            populations = summary["file_populations"]
            self.assertEqual(populations["all_query_matches"]["files"], 4)
            self.assertEqual(populations["case_insensitive_exact_claude"]["files"], 2)
            self.assertEqual(populations["case_insensitive_exact_claude"]["repositories"], 1)
            self.assertEqual(populations["strict_case_exact_claude"]["files"], 1)
            self.assertEqual(summary["repositories"]["count"], 3)
            self.assertEqual(languages[0]["language"], "Python")
            self.assertEqual(languages[0]["repositories"], 2)
            self.assertEqual(top[0]["repository"], "three/c")
            self.assertEqual(summary["exact_file_size"]["words"]["total"], 8)
            self.assertEqual(summary["exact_file_size"]["estimated_tokens"]["total"], 5)


if __name__ == "__main__":
    unittest.main()

