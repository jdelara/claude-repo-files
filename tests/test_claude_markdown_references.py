from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from miner.markdown_references import (
    extract_external_markdown_references,
    extract_local_markdown_reference_occurrences,
    extract_local_markdown_references,
)
from scripts.claude_markdown_references import (
    aggregate_strata,
    aggregate_targets,
    analyze_references,
    build_parser,
    build_summary,
    classify_local_intent,
    local_path_relation,
    resolve_local_target,
    target_category,
)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class OccurrencePrimitiveTests(unittest.TestCase):
    def test_preserves_duplicate_local_mentions_without_changing_unique_api(self):
        content = "Compare rules.md with rules.md.\n"

        occurrences = extract_local_markdown_reference_occurrences(content)
        unique = extract_local_markdown_references(content)

        self.assertEqual(len(occurrences), 2)
        self.assertEqual(len(unique), 1)
        self.assertEqual({item.normalized_target for item in occurrences}, {"rules.md"})

    def test_extracts_external_markdown_urls_but_not_comments_or_fences(self):
        content = """[Policy](https://example.org/docs/POLICY.md#rules)
Bare https://example.org/AGENTS.md.
<!-- https://example.org/ignored.md -->
```text
https://example.org/fenced.md
```
"""

        references = extract_external_markdown_references(content)

        self.assertEqual(
            [(item.syntax, item.target_basename) for item in references],
            [
                ("external_markdown_link", "policy.md"),
                ("external_url", "agents.md"),
            ],
        )
        self.assertTrue(references[0].normalized_target.endswith("#rules"))
        self.assertFalse(references[1].normalized_target.endswith("."))

    def test_abstains_on_malformed_external_url(self):
        references = extract_external_markdown_references(
            "Malformed https://[invalid/thing.md should not abort.\n"
        )

        self.assertEqual(references, ())


class PathAndIntentTests(unittest.TestCase):
    def test_resolves_and_classifies_relative_paths(self):
        self.assertEqual(
            resolve_local_target("docs/api/CLAUDE.md", "../../AGENTS.md"),
            "AGENTS.md",
        )
        self.assertEqual(
            local_path_relation("docs/api/CLAUDE.md", "AGENTS.md"),
            "ancestor_directory",
        )
        self.assertEqual(
            local_path_relation("CLAUDE.md", "docs/rules.md"),
            "descendant_directory",
        )
        self.assertEqual(
            local_path_relation("CLAUDE.md", "../rules.md"),
            "outside_repository",
        )

    def test_assigns_direct_delegation_and_contextual_categories(self):
        direct, delegation, contextual = (
            extract_local_markdown_reference_occurrences(
                "@AGENTS.md\nYou must read docs/rules.md before editing.\n"
                "[Architecture](docs/architecture.md)\n"
            )
        )

        self.assertEqual(
            classify_local_intent(
                direct,
                target_category(direct.target_basename, direct.normalized_target),
                direct.source_line,
            ),
            ("direct_inclusion", "high", "explicit_include_syntax_v1", 1),
        )
        self.assertEqual(
            classify_local_intent(
                delegation,
                target_category(delegation.target_basename, delegation.normalized_target),
                delegation.source_line,
            )[0],
            "instructional_delegation",
        )
        self.assertEqual(
            classify_local_intent(
                contextual,
                target_category(contextual.target_basename, contextual.normalized_target),
                contextual.source_line,
            )[-1],
            0,
        )

    def test_requires_target_local_cue_and_excludes_generic_file_maintenance(self):
        unrelated, maintenance, directory_label, canonical = (
            extract_local_markdown_reference_occurrences(
                "Python must not parse or interpret LLM-generated content "
                "(review reports, findings, assessed.md).\n"
                "After changes, you MUST check if README.md needs to be updated.\n"
                "CustomInstructions contains GPT instructions, including an index "
                "in CustomInstructions/README.md.\n"
                "Canonical agent instructions live in AGENTS.md.\n"
            )
        )

        for reference in (unrelated, maintenance, directory_label):
            category = target_category(reference.target_basename, reference.normalized_target)
            self.assertEqual(
                classify_local_intent(reference, category, reference.source_line)[-1],
                0,
            )
        category = target_category(canonical.target_basename, canonical.normalized_target)
        self.assertEqual(
            classify_local_intent(canonical, category, canonical.source_line)[0],
            "instructional_delegation",
        )


class CorpusAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "sample.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE repos (
                full_name TEXT PRIMARY KEY,
                stars INTEGER,
                language TEXT,
                fetched_at TEXT
            );
            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                ("owner/alpha", 5, "Python"),
                ("owner/beta", 0, "TypeScript"),
                ("owner/gamma", None, None),
            ],
        )
        pointer = "# Instructions\n\n@AGENTS.md\n"
        substantive = (
            "# Guide\n\nBefore editing, you must read ../rules.md.\n"
            "[Architecture](architecture.md)\n"
            "This CLAUDE.md describes the current file.\n\n"
            + "Project-specific operational detail.\n" * 55
        )
        external = (
            "[External policy](https://example.org/POLICY.md)\n"
            "```text\n@ignored.md\n```\n"
        )
        records = [
            ("owner/alpha", "CLAUDE.md", pointer),
            ("owner/alpha", "other/CLAUDE.md", pointer),
            ("owner/alpha", "docs/CLAUDE.md", substantive),
            ("owner/beta", "CLAUDE.md", external),
            ("owner/beta", "copy/CLAUDE.md", pointer),
            ("owner/gamma", "CLAUDE.md", "# No references\nRun tests.\n"),
            ("owner/gamma", "not-claude.md", "@ignored.md\n"),
        ]
        connection.executemany(
            """INSERT INTO files(
                   repo_full_name, path, content, content_hash, size_bytes
               ) VALUES (?, ?, ?, ?, ?)""",
            [
                (repo, path, content, _hash(content), len(content.encode("utf-8")))
                for repo, path, content in records
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_analyzes_exact_scope_and_excludes_self_from_candidate_numerator(self):
        references, files, changes = analyze_references(self.db_path)

        self.assertEqual(len(files), 6)
        self.assertEqual(changes, 0)
        self.assertEqual(
            {(item.repo_full_name, item.source_path) for item in files},
            {
                ("owner/alpha", "CLAUDE.md"),
                ("owner/alpha", "other/CLAUDE.md"),
                ("owner/alpha", "docs/CLAUDE.md"),
                ("owner/beta", "CLAUDE.md"),
                ("owner/beta", "copy/CLAUDE.md"),
                ("owner/gamma", "CLAUDE.md"),
            },
        )
        docs_file = next(item for item in files if item.source_path == "docs/CLAUDE.md")
        self.assertEqual(docs_file.self_reference_occurrences, 1)
        self.assertEqual(docs_file.instructional_fan_out, 1)
        self.assertEqual(docs_file.document_form, "instructions_plus_delegation")
        self.assertEqual(
            {item.resolved_target for item in references if item.source_path == "docs/CLAUDE.md" and item.high_confidence_local_instructional},
            {"rules.md"},
        )

        root_pointer = next(
            item
            for item in files
            if item.repo_full_name == "owner/alpha" and item.source_path == "CLAUDE.md"
        )
        self.assertEqual(root_pointer.document_form, "pointer_only")
        self.assertEqual(root_pointer.repository_content_copies, 2)
        self.assertEqual(root_pointer.global_content_copies, 3)

    def test_aggregates_deterministically_and_preserves_invariants(self):
        first_references, first_files, first_changes = analyze_references(self.db_path)
        second_references, second_files, second_changes = analyze_references(self.db_path)
        self.assertEqual(first_references, second_references)
        self.assertEqual(first_files, second_files)
        self.assertEqual(first_changes, second_changes)

        targets = aggregate_targets(first_references)
        strata = aggregate_strata(first_files)
        summary = build_summary(
            first_references,
            first_files,
            targets,
            strata,
            database_sha256_before="same",
            database_sha256_after="same",
            database_changes=first_changes,
        )

        self.assertTrue(summary["database_unchanged"])
        self.assertEqual(summary["population"], {"files": 6, "repositories": 3})
        self.assertEqual(
            summary["reference_counts"]["files_with_external_markdown_reference"],
            1,
        )
        candidate = summary["high_confidence_local_instructional_candidates"]
        self.assertEqual(candidate["files"], 4)
        self.assertEqual(candidate["repositories"], 2)
        self.assertEqual(
            sum(item["files"] for item in summary["document_forms"]),
            6,
        )
        location_rows = [item for item in strata if item.dimension == "source_location"]
        self.assertEqual(sum(item.files for item in location_rows), 6)

    def test_parser_defaults_to_phase2_artifacts(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(
            arguments.occurrences_output,
            "article/claude_markdown_reference_occurrences.csv",
        )
        self.assertEqual(
            arguments.summary_output,
            "article/claude_markdown_reference_summary.json",
        )


if __name__ == "__main__":
    unittest.main()
