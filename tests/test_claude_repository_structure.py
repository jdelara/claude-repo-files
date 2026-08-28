from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_repository_structure import (
    FileRecord,
    _build_hierarchy,
    _documented_scope_mapping,
    _normalize_lines,
    analyze_repository_structure,
    build_parser,
    write_component_csv,
    write_hierarchy_csv,
    write_overlap_csv,
    write_repository_csv,
    write_summary_json,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class RepositoryStructureTests(unittest.TestCase):
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
                       content TEXT
                   )"""
            )
            base = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n"
            similar = base + "eta\n"
            contained = base + "one\ntwo\nthree\nfour\n"
            normalized_a = "One  space\nTWO\n"
            normalized_b = " one space \n two \n"
            rows = [
                ("root/only", "CLAUDE.md", "root only\n"),
                ("nested/only", "docs/CLAUDE.md", "nested only\n"),
                ("mixed/repo", "CLAUDE.md", base),
                ("mixed/repo", "packages/CLAUDE.md", base),
                ("mixed/repo", "packages/api/claude.MD", similar),
                ("mixed/repo", "apps/CLAUDE.md", contained),
                ("siblings/repo", "apps/a/CLAUDE.md", normalized_a),
                ("siblings/repo", "apps/b/CLAUDE.md", normalized_b),
                ("excluded/repo", "CLAUDE.md.template", base),
            ]
            connection.executemany(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?)",
                [
                    (repo, path, _hash(content), len(content.encode()), content)
                    for repo, path, content in rows
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_analyzes_location_multiplicity_hierarchy_and_reuse(self):
        summary, repositories, components, edges, overlaps = (
            analyze_repository_structure(self.db_path)
        )

        population = summary["population"]
        self.assertEqual(population["files"], 8)
        self.assertEqual(population["repositories"], 4)
        self.assertEqual(population["root_files"], 2)
        self.assertEqual(population["nested_files"], 6)
        self.assertEqual(population["multi_file_repositories"], 2)
        self.assertEqual(population["files_in_multi_file_repositories"], 6)

        configurations = summary["location_configurations"]
        self.assertEqual(configurations["root_only"]["repositories"], 1)
        self.assertEqual(configurations["nested_only"]["repositories"], 2)
        self.assertEqual(configurations["root_and_nested"]["repositories"], 1)

        depth_counts = {
            item["directory_depth"]: item["files"]
            for item in summary["directory_depth"]["counts"]
        }
        self.assertEqual(depth_counts, {0: 2, 1: 3, 2: 3})
        self.assertEqual(
            summary["directory_depth"]["nested_file_distribution"]["median"],
            1,
        )

        by_repository = {
            repository.repo_full_name: repository
            for repository in repositories
        }
        mixed = by_repository["mixed/repo"]
        self.assertEqual(mixed.exact_files, 4)
        self.assertEqual(mixed.hierarchy_edges, 3)
        self.assertEqual(mixed.files_with_nearest_ancestor, 3)
        self.assertEqual(mixed.files_serving_as_nearest_ancestor, 2)
        self.assertEqual(mixed.maximum_instruction_scope_chain, 3)
        self.assertEqual(mixed.exact_duplicate_groups, 1)
        self.assertEqual(mixed.files_in_exact_duplicate_groups, 2)
        self.assertEqual(mixed.repeated_exact_instances, 1)
        self.assertEqual(mixed.exact_duplicate_pairs, 1)

        hierarchy = summary["hierarchy"]
        self.assertEqual(hierarchy["nearest_ancestor_edges"], 3)
        self.assertEqual(hierarchy["root_parent_edges"], 2)
        self.assertEqual(hierarchy["nested_parent_edges"], 1)
        self.assertEqual(
            hierarchy["repositories_with_nearest_ancestor_edges"],
            1,
        )
        self.assertEqual(
            hierarchy["multi_file_repositories_without_ancestor_relation"],
            1,
        )
        self.assertEqual(hierarchy["maximum_observed_instruction_scope_chain"], 3)
        self.assertEqual(
            hierarchy["repositories_with_edges_by_location_configuration"],
            {"root_only": 0, "nested_only": 0, "root_and_nested": 1},
        )
        self.assertEqual(len(edges), 3)

        exact_reuse = summary["within_repository_exact_reuse"]
        self.assertEqual(exact_reuse["repositories_with_exact_duplicate_groups"], 1)
        self.assertEqual(exact_reuse["exact_duplicate_groups"], 1)
        self.assertEqual(exact_reuse["files_in_exact_duplicate_groups"], 2)
        self.assertEqual(exact_reuse["file_share_in_exact_duplicate_groups"], 0.25)

        first_directories = {
            row["component_casefolded"]: row["file_count"]
            for row in components
            if row["dimension"] == "first_directory"
        }
        self.assertEqual(first_directories["apps"], 3)
        self.assertEqual(first_directories["packages"], 2)
        self.assertEqual(first_directories["docs"], 1)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM files").fetchone()[0],
                9,
            )
        finally:
            connection.close()

    def test_reports_transparent_nonexact_line_overlap_categories(self):
        summary, repositories, _components, _edges, overlaps = (
            analyze_repository_structure(self.db_path)
        )

        line_overlap = summary["within_repository_normalized_line_overlap"]
        self.assertEqual(line_overlap["candidate_file_pairs"], 7)
        self.assertEqual(line_overlap["evaluated_different_hash_pairs"], 6)
        self.assertEqual(line_overlap["reported_pairs"], 5)
        self.assertEqual(
            line_overlap["pair_counts_by_relation"],
            {
                "normalized_equivalent": 1,
                "high_jaccard": 2,
                "high_containment": 2,
            },
        )
        self.assertEqual(line_overlap["repositories_with_reported_pairs"], 2)

        by_repository = {
            repository.repo_full_name: repository
            for repository in repositories
        }
        self.assertEqual(by_repository["mixed/repo"].high_jaccard_pairs, 2)
        self.assertEqual(by_repository["mixed/repo"].high_containment_pairs, 2)
        self.assertEqual(
            by_repository["siblings/repo"].normalized_equivalent_pairs,
            1,
        )
        self.assertEqual(len(overlaps), 5)
        self.assertEqual(
            _normalize_lines(" A  B \n\nC\n"),
            frozenset(("a b", "c")),
        )

    def test_ambiguous_same_directory_parents_are_explicit(self):
        files = [
            FileRecord("case/repo", "CLAUDE.md", "a", 1),
            FileRecord("case/repo", "claude.MD", "b", 1),
            FileRecord("case/repo", "child/CLAUDE.md", "c", 1),
        ]

        edges, child_files, parent_files, maximum_chain, same_directory = (
            _build_hierarchy("case/repo", files)
        )

        self.assertEqual(len(edges), 2)
        self.assertEqual(child_files, 1)
        self.assertEqual(parent_files, 2)
        self.assertEqual(maximum_chain, 2)
        self.assertEqual(same_directory, 1)
        self.assertEqual(
            {edge.parent_candidates_at_nearest_scope for edge in edges},
            {2},
        )

    def test_maps_hidden_claude_directory_to_its_containing_scope(self):
        files_by_repository = {
            "root/direct": [
                FileRecord("root/direct", "CLAUDE.md", "a", 1),
            ],
            "root/hidden": [
                FileRecord("root/hidden", ".claude/CLAUDE.md", "b", 1),
            ],
            "both/forms": [
                FileRecord("both/forms", "CLAUDE.md", "c", 1),
                FileRecord("both/forms", ".claude/CLAUDE.md", "d", 1),
                FileRecord("both/forms", "pkg/CLAUDE.md", "e", 1),
                FileRecord("both/forms", "pkg/.claude/CLAUDE.md", "e", 1),
            ],
            "nested/only": [
                FileRecord("nested/only", "pkg/.CLAUDE/CLAUDE.md", "f", 1),
            ],
        }

        summary = _documented_scope_mapping(files_by_repository)

        self.assertEqual(
            summary["placement_forms"],
            {
                "direct_files": 3,
                "direct_repositories": 2,
                "hidden_claude_directory_files": 4,
                "hidden_claude_directory_exact_case_files": 3,
                "hidden_claude_directory_case_variant_files": 1,
                "hidden_claude_directory_repositories": 3,
                "hidden_claude_directory_file_share": 4 / 7,
            },
        )
        logical = summary["logical_file_location"]
        self.assertEqual(logical["project_root_scope_files"], 4)
        self.assertEqual(logical["nonroot_scope_files"], 3)
        self.assertEqual(logical["project_root_hidden_exact_case_files"], 2)
        self.assertEqual(logical["project_root_hidden_case_variant_files"], 0)
        configurations = summary["logical_location_configurations"]
        self.assertEqual(configurations["project_root_only"]["repositories"], 2)
        self.assertEqual(configurations["nonroot_only"]["repositories"], 1)
        self.assertEqual(
            configurations["project_root_and_nonroot"]["repositories"],
            1,
        )
        alternatives = summary["same_logical_scope_alternative_placements"]
        self.assertEqual(alternatives["scope_groups"], 2)
        self.assertEqual(alternatives["project_root_scope_groups"], 1)
        self.assertEqual(alternatives["nonroot_scope_groups"], 1)
        self.assertEqual(alternatives["file_pairs"], 2)
        self.assertEqual(alternatives["same_content_hash_file_pairs"], 1)
        topology = summary["logical_scope_topology"]
        self.assertEqual(topology["nearest_ancestor_scope_edges"], 1)
        self.assertEqual(
            topology["repositories_with_nearest_ancestor_scope_edges"],
            1,
        )
        self.assertEqual(topology["maximum_scope_chain"], 2)

    def test_writes_deterministic_machine_readable_artifacts(self):
        summary, repositories, components, edges, overlaps = (
            analyze_repository_structure(self.db_path)
        )
        output = Path(self.temporary_directory.name) / "reports"

        repository_path = write_repository_csv(
            repositories,
            output / "repositories.csv",
        )
        component_path = write_component_csv(components, output / "components.csv")
        hierarchy_path = write_hierarchy_csv(edges, output / "hierarchy.csv")
        overlap_path = write_overlap_csv(overlaps, output / "overlap.csv")
        summary_path = write_summary_json(summary, output / "summary.json")
        repeated_output = Path(self.temporary_directory.name) / "repeated"
        repeated_paths = (
            write_repository_csv(
                repositories,
                repeated_output / "repositories.csv",
            ),
            write_component_csv(
                components,
                repeated_output / "components.csv",
            ),
            write_hierarchy_csv(edges, repeated_output / "hierarchy.csv"),
            write_overlap_csv(overlaps, repeated_output / "overlap.csv"),
            write_summary_json(summary, repeated_output / "summary.json"),
        )

        with repository_path.open(encoding="utf-8", newline="") as handle:
            repository_rows = list(csv.DictReader(handle))
        with component_path.open(encoding="utf-8", newline="") as handle:
            component_rows = list(csv.DictReader(handle))
        with hierarchy_path.open(encoding="utf-8", newline="") as handle:
            hierarchy_rows = list(csv.DictReader(handle))
        with overlap_path.open(encoding="utf-8", newline="") as handle:
            overlap_rows = list(csv.DictReader(handle))
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(repository_rows), 4)
        self.assertGreater(len(component_rows), 0)
        self.assertEqual(len(hierarchy_rows), 3)
        self.assertEqual(len(overlap_rows), 5)
        self.assertEqual(summary_data["population"]["files"], 8)
        for original, repeated in zip(
            (
                repository_path,
                component_path,
                hierarchy_path,
                overlap_path,
                summary_path,
            ),
            repeated_paths,
        ):
            self.assertEqual(original.read_bytes(), repeated.read_bytes())

    def test_parser_defaults_are_the_published_parameters(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.minimum_shared_lines, 5)
        self.assertEqual(arguments.jaccard_threshold, 0.80)
        self.assertEqual(arguments.containment_threshold, 0.90)


if __name__ == "__main__":
    unittest.main()
