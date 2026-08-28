from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_hierarchy_allocation import (
    analyze_hierarchy_allocation,
    build_parser,
    write_edge_csv,
    write_heading_csv,
    write_repository_csv,
    write_summary_json,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ClaudeHierarchyAllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.execute(
            """CREATE TABLE files (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   repo_full_name TEXT NOT NULL,
                   path TEXT NOT NULL,
                   content_hash TEXT,
                   size_bytes INTEGER,
                   content TEXT
               )"""
        )
        root = "# Overview\n\n@packages/CLAUDE.md\n\nRoot guidance.\n"
        child = "# Testing\n\n@../CLAUDE.md\n\n- Run tests.\n"
        grandchild = (
            "# API\n\nAPI guidance.\n\n"
            "```sh\npytest\n```\n"
        )
        sibling_a = "# Alpha\n\nAlpha guidance.\n"
        sibling_b = "# Beta\n\nBeta guidance with more words.\n"
        pointer = "# Instructions\n\n@AGENTS.md\n"
        single = "# Single\n\nExcluded.\n"
        records = [
            ("owner/hierarchy", "CLAUDE.md", root),
            ("owner/hierarchy", "packages/CLAUDE.md", child),
            ("owner/hierarchy", "packages/api/CLAUDE.md", grandchild),
            ("owner/siblings", "apps/a/CLAUDE.md", sibling_a),
            ("owner/siblings", "apps/b/CLAUDE.md", sibling_b),
            ("owner/duplicate", "CLAUDE.md", pointer),
            ("owner/duplicate", "copy/CLAUDE.md", pointer),
            ("owner/single", "CLAUDE.md", single),
            ("owner/excluded", "CLAUDE.md.template", root),
        ]
        connection.executemany(
            """INSERT INTO files(
                   repo_full_name, path, content_hash, size_bytes, content
               ) VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    repo,
                    path,
                    _hash(content),
                    len(content.encode("utf-8")),
                    content,
                )
                for repo, path, content in records
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_analyzes_allocation_edges_overlap_and_delegation(self) -> None:
        summary, repositories, edges, headings = analyze_hierarchy_allocation(
            self.db_path,
            heading_limit=50,
        )

        population = summary["population"]
        self.assertEqual(population["repositories"], 3)
        self.assertEqual(population["files"], 7)
        self.assertEqual(population["root_files"], 2)
        self.assertEqual(population["nested_files"], 5)
        self.assertEqual(population["nearest_ancestor_edges"], 3)
        self.assertEqual(
            population["location_configurations"],
            {"root_only": 0, "nested_only": 1, "root_and_nested": 2},
        )

        by_repository = {row.repo_full_name: row for row in repositories}
        hierarchy = by_repository["owner/hierarchy"]
        self.assertEqual(hierarchy.exact_files, 3)
        self.assertEqual(hierarchy.hierarchy_edges, 2)
        self.assertEqual(hierarchy.parent_to_child_literal_reference_edges, 1)
        self.assertEqual(hierarchy.child_to_parent_literal_reference_edges, 1)
        self.assertEqual(hierarchy.either_direction_literal_reference_edges, 1)
        self.assertEqual(hierarchy.bidirectional_literal_reference_edges, 1)
        self.assertEqual(hierarchy.parent_to_child_high_confidence_edges, 1)
        self.assertEqual(hierarchy.child_to_parent_high_confidence_edges, 1)

        duplicate = by_repository["owner/duplicate"]
        self.assertEqual(duplicate.within_repository_exact_duplicate_groups, 1)
        self.assertEqual(
            duplicate.files_in_within_repository_exact_duplicate_groups,
            2,
        )
        self.assertEqual(duplicate.pointer_only_files, 2)
        self.assertAlmostEqual(
            duplicate.largest_file_byte_share,
            0.5,
        )

        by_edge = {(row.repo_full_name, row.parent_path, row.child_path): row for row in edges}
        root_child = by_edge[
            ("owner/hierarchy", "CLAUDE.md", "packages/CLAUDE.md")
        ]
        self.assertEqual(root_child.parent_references_child_literal, 1)
        self.assertEqual(root_child.child_references_parent_literal, 1)
        self.assertEqual(root_child.parent_references_child_high_confidence, 1)
        self.assertEqual(root_child.child_references_parent_high_confidence, 1)
        duplicate_edge = by_edge[
            ("owner/duplicate", "CLAUDE.md", "copy/CLAUDE.md")
        ]
        self.assertEqual(duplicate_edge.same_content_hash, 1)
        self.assertEqual(duplicate_edge.same_normalized_line_set, 1)
        self.assertEqual(duplicate_edge.jaccard_similarity, 1.0)
        self.assertEqual(duplicate_edge.child_unique_normalized_lines, 0)

        heading_by_name = {row.normalized_heading: row for row in headings}
        self.assertEqual(heading_by_name["overview"].root_documents, 1)
        self.assertEqual(heading_by_name["testing"].nested_documents, 1)
        self.assertEqual(heading_by_name["api"].child_role_documents, 1)
        allocation = summary["content_allocation"]
        by_file_count = {
            row["value"]: row
            for row in allocation["by_observed_file_count"]
        }
        self.assertEqual(by_file_count["2"]["repositories"], 2)
        self.assertEqual(by_file_count["3"]["repositories"], 1)
        comparison = summary["parent_child_comparison"]
        self.assertEqual(comparison["different_hash_edges"], 2)
        self.assertEqual(
            comparison["different_hash_same_normalized_line_set_edges"],
            0,
        )
        self.assertTrue(summary["database"]["unchanged"])

    def test_writes_byte_identical_artifacts(self) -> None:
        summary, repositories, edges, headings = analyze_hierarchy_allocation(
            self.db_path,
            heading_limit=50,
        )
        first = Path(self.temporary_directory.name) / "first"
        second = Path(self.temporary_directory.name) / "second"
        first_paths = (
            write_repository_csv(repositories, first / "repositories.csv"),
            write_edge_csv(edges, first / "edges.csv"),
            write_heading_csv(headings, first / "headings.csv"),
            write_summary_json(summary, first / "summary.json"),
        )
        second_paths = (
            write_repository_csv(repositories, second / "repositories.csv"),
            write_edge_csv(edges, second / "edges.csv"),
            write_heading_csv(headings, second / "headings.csv"),
            write_summary_json(summary, second / "summary.json"),
        )

        with first_paths[0].open(encoding="utf-8", newline="") as handle:
            repository_rows = list(csv.DictReader(handle))
        with first_paths[1].open(encoding="utf-8", newline="") as handle:
            edge_rows = list(csv.DictReader(handle))
        with first_paths[2].open(encoding="utf-8", newline="") as handle:
            heading_rows = list(csv.DictReader(handle))
        loaded_summary = json.loads(first_paths[3].read_text(encoding="utf-8"))

        self.assertEqual(len(repository_rows), 3)
        self.assertEqual(len(edge_rows), 3)
        self.assertGreater(len(heading_rows), 0)
        self.assertEqual(loaded_summary["population"]["files"], 7)
        for left, right in zip(first_paths, second_paths):
            self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_parser_defaults_to_article_artifacts(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.heading_limit, 1000)
        self.assertEqual(
            arguments.repositories_output,
            "article/claude_hierarchy_allocation_repositories.csv",
        )
        self.assertEqual(
            arguments.summary_output,
            "article/claude_hierarchy_allocation_summary.json",
        )


if __name__ == "__main__":
    unittest.main()
