from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from miner.markdown_references import extract_local_markdown_references
from scripts.claude_repeated_content_classes import (
    CATEGORIES,
    analyze_repeated_content,
    build_parser,
    classify_content,
    write_audit_csv,
    write_classes_csv,
    write_groups_csv,
    write_members_csv,
    write_summary_json,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ReferencePrimitiveTests(unittest.TestCase):
    def test_extracts_local_references_but_not_title_fences_or_urls(self):
        content = """# CLAUDE.md

@AGENTS.md
Follow [the rules](docs/rules.md) and `README.md`.
See https://example.com/external.md.
[External](https://example.com/external.md)
<!-- include: docs/extra.md -->
```text
@ignored.md
```
"""

        references = extract_local_markdown_references(content)

        self.assertEqual(
            {(item.syntax, item.normalized_target) for item in references},
            {
                ("direct_include", "AGENTS.md"),
                ("markdown_link", "docs/rules.md"),
                ("path_mention", "README.md"),
                ("comment_include", "docs/extra.md"),
            },
        )
        self.assertNotIn("claude.md", {item.target_basename for item in references})


class ContentClassificationTests(unittest.TestCase):
    def test_assigns_each_high_precision_form_and_fallback(self):
        examples = {
            "pointer": ("# CLAUDE.md\n\n@AGENTS.md\n", "pointer_shim"),
            "empty": (
                "<claude-mem-context>\n\n</claude-mem-context>\n",
                "empty_placeholder",
            ),
            "generated": (
                "<claude-mem-context>\nRecent activity\n</claude-mem-context>\n",
                "generated_context",
            ),
            "template": (
                "# Guide\n\nThis project is a template for API connectors and is "
                "meant to be cloned and customized.\n",
                "template_scaffold",
            ),
            "substantive": (
                "# Commands\n" + "You must run tests before every change.\n" * 30,
                "substantive_repeated_document",
            ),
            "short": ("# Notes\nUse pnpm.\n", "short_ambiguous"),
            "path_only": ("../../apps/web/CLAUDE.md\n", "pointer_shim"),
            "legacy_rules": (".rules\n", "pointer_shim"),
            "empty_sections": (
                "## Issues\n\n*(None)*\n\n## Warnings\n\n*(None)*\n",
                "empty_placeholder",
            ),
            "front_matter_pointer": (
                "---\ntitle: Claude\nsummary: AGENTS.md\ncategory: docs\n---\n\nAGENTS.md\n",
                "pointer_shim",
            ),
            "generic_at_pointer": ("@docs/fake-services\n", "pointer_shim"),
        }

        results = {
            name: classify_content(name, content, len(content.encode()))
            for name, (content, _category) in examples.items()
        }

        for name, (_content, category) in examples.items():
            self.assertEqual(results[name].primary_category, category)
        self.assertEqual(results["pointer"].target_category, "agents_md")
        self.assertEqual(results["pointer"].direct_include_count, 1)
        self.assertEqual(results["empty"].subtype, "empty_generated_block")
        self.assertEqual(results["path_only"].subtype, "path_only")
        self.assertEqual(results["legacy_rules"].target_category, "rules_file")
        self.assertEqual(results["empty_sections"].subtype, "empty_sections")
        self.assertEqual(results["front_matter_pointer"].subtype, "path_only")
        self.assertEqual(results["generic_at_pointer"].subtype, "generic_at_pointer")


class RepeatedContentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        pointer = "# CLAUDE.md\n\n@AGENTS.md\n"
        contents = {
            "empty": "<claude-mem-context>\n\n</claude-mem-context>\n",
            "generated": (
                "<claude-mem-context>\nRecent activity\n</claude-mem-context>\n"
            ),
            "template": (
                "# Guide\n\nThis project is a template for API connectors and is "
                "meant to be cloned and customized.\n"
            ),
            "substantive": (
                "# Commands\n" + "You must run tests before every change.\n" * 30
            ),
            "short": "# Notes\nUse pnpm.\n",
        }
        rows: list[tuple[str, str, str]] = [
            ("pointer/one", "skills/a/CLAUDE.md", pointer),
            ("pointer/one", "skills/b/CLAUDE.md", pointer),
            ("pointer/two", "packages/a/CLAUDE.md", pointer),
            ("pointer/two", "packages/b/CLAUDE.md", pointer),
        ]
        for name, content in contents.items():
            rows.extend(
                [
                    (f"category/{name}", "a/CLAUDE.md", content),
                    (f"category/{name}", "b/claude.MD", content),
                ]
            )
        rows.extend(
            [
                ("excluded/unique", "CLAUDE.md", "unique content\n"),
                ("excluded/variant", "CLAUDE.md.template", pointer),
                ("excluded/variant", "other/CLAUDE.md.template", pointer),
            ]
        )

        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       repo_full_name TEXT NOT NULL,
                       path TEXT NOT NULL,
                       content_hash TEXT,
                       size_bytes INTEGER,
                       content TEXT,
                       html_url TEXT
                   )"""
            )
            connection.executemany(
                """INSERT INTO files
                   (repo_full_name, path, content_hash, size_bytes, content, html_url)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        repo,
                        path,
                        _hash(content),
                        len(content.encode()),
                        content,
                        f"https://example.test/{index}",
                    )
                    for index, (repo, path, content) in enumerate(rows, 1)
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_classifies_once_per_hash_and_aggregates_all_units(self):
        summary, classes, groups, members, audit = analyze_repeated_content(
            self.db_path,
            audit_per_category=1,
        )

        self.assertEqual(summary["population"]["unique_contents"], 6)
        self.assertEqual(summary["population"]["repository_groups"], 7)
        self.assertEqual(summary["population"]["repositories"], 7)
        self.assertEqual(summary["population"]["files"], 14)
        self.assertEqual(summary["population"]["repeated_instances"], 7)
        self.assertEqual(len(classes), 6)
        self.assertEqual(len(groups), 7)
        self.assertEqual(len(members), 14)
        self.assertEqual(len(audit), 6)

        by_category = {
            row["category"]: row for row in summary["categories"]
        }
        self.assertEqual(set(by_category), set(CATEGORIES))
        self.assertEqual(by_category["pointer_shim"]["unique_contents"], 1)
        self.assertEqual(by_category["pointer_shim"]["repository_groups"], 2)
        self.assertEqual(by_category["pointer_shim"]["files"], 4)
        self.assertEqual(by_category["pointer_shim"]["repeated_instances"], 2)
        for category in CATEGORIES[1:]:
            self.assertEqual(by_category[category]["unique_contents"], 1)
            self.assertEqual(by_category[category]["repository_groups"], 1)

        pointer = next(
            row for row in classes if row.primary_category == "pointer_shim"
        )
        self.assertEqual(pointer.repository_groups, 2)
        self.assertEqual(pointer.repositories, 2)
        self.assertEqual(pointer.files, 4)
        self.assertEqual(pointer.target_basenames, "agents.md")
        self.assertEqual(
            {row.path_context_tags for row in groups if row.primary_category == "pointer_shim"},
            {"skill_plugin_or_agent", "package_app_or_component"},
        )

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM files").fetchone()[0],
                17,
            )
        finally:
            connection.close()

    def test_writes_byte_deterministic_csv_and_json_artifacts(self):
        summary, classes, groups, members, audit = analyze_repeated_content(
            self.db_path,
            audit_per_category=1,
        )
        first = Path(self.temporary_directory.name) / "first"
        second = Path(self.temporary_directory.name) / "second"

        first_paths = (
            write_classes_csv(classes, first / "classes.csv"),
            write_groups_csv(groups, first / "groups.csv"),
            write_members_csv(members, first / "members.csv"),
            write_audit_csv(audit, first / "audit.csv"),
            write_summary_json(summary, first / "summary.json"),
        )
        second_paths = (
            write_classes_csv(classes, second / "classes.csv"),
            write_groups_csv(groups, second / "groups.csv"),
            write_members_csv(members, second / "members.csv"),
            write_audit_csv(audit, second / "audit.csv"),
            write_summary_json(summary, second / "summary.json"),
        )

        for left, right in zip(first_paths, second_paths):
            self.assertEqual(left.read_bytes(), right.read_bytes())
        with first_paths[0].open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)
        data = json.loads(first_paths[-1].read_text(encoding="utf-8"))
        self.assertEqual(data["population"]["repository_groups"], 7)

    def test_parser_defaults_are_published_parameters(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(arguments.pointer_max_residual_words, 60)
        self.assertEqual(arguments.substantive_min_words, 50)
        self.assertEqual(arguments.substantive_min_nonempty_lines, 8)
        self.assertEqual(arguments.substantive_min_bytes, 500)
        self.assertEqual(arguments.audit_per_category, 10)
        self.assertEqual(arguments.audit_seed, 20260727)


if __name__ == "__main__":
    unittest.main()
