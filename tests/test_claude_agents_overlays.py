from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.claude_agents_overlays import (
    OverlayFileRow,
    analyze_content,
    analyze_database,
    build_parser,
    build_summary,
    is_explicit_claude_overlay_heading,
    write_files_csv,
    write_sections_csv,
    write_summary_json,
)


class OverlayContentTests(unittest.TestCase):
    def test_classifies_a_strict_literal_import_shim(self) -> None:
        result = analyze_content("@AGENTS.md\n", "CLAUDE.md")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result.import_spans), 1)
        self.assertEqual(result.exact_literal_at_agents_occurrences, 1)
        self.assertEqual(result.strict_import_only, 1)
        self.assertEqual(result.no_residual_outside_html_comments, 1)
        self.assertEqual(result.residual_word_count, 0)
        self.assertEqual(result.sections, ())

    def test_separates_html_comments_from_raw_strict_shims(self) -> None:
        result = analyze_content(
            "@AGENTS.md\n<!-- Maintainer-only compatibility note. -->\n",
            "CLAUDE.md",
        )

        assert result is not None
        self.assertEqual(result.strict_import_only, 0)
        self.assertEqual(result.no_residual_outside_html_comments, 1)
        self.assertEqual(result.residual_word_count, 0)

    def test_detects_a_concrete_claude_specific_section_after_import(self) -> None:
        content = """# CLAUDE.md

@AGENTS.md

## Claude Code-specific instructions

Use plan mode before non-trivial changes. Run `/review` before finishing.
"""
        result = analyze_content(content, "CLAUDE.md")

        assert result is not None
        self.assertEqual(len(result.sections), 1)
        section = result.sections[0]
        self.assertEqual(section.heading_line, 5)
        self.assertEqual(section.label_kind, "explicit_specificity_or_adapter")
        self.assertEqual(section.has_english_directive_cue, 1)
        self.assertEqual(section.has_placeholder_language, 0)
        self.assertEqual(section.concrete_directive_candidate, 1)
        self.assertIn("plan_mode", section.themes)
        self.assertIn("skills_or_slash_commands", section.themes)
        self.assertGreater(result.residual_after_final_import_word_count, 10)

    def test_marks_placeholder_sections_without_calling_them_concrete(self) -> None:
        content = """@AGENTS.md

## Claude-specific instructions

_(None yet. Add Claude-Code-specific guidance here.)_
"""
        result = analyze_content(content, "CLAUDE.md")

        assert result is not None
        section = result.sections[0]
        self.assertEqual(section.has_placeholder_language, 1)
        self.assertEqual(section.concrete_directive_candidate, 0)

    def test_requires_the_labeled_section_to_follow_the_final_import(self) -> None:
        content = """## Claude Code workflow
Always run the tests.

@AGENTS.md

```markdown
## Claude Code-specific instructions
Use plan mode.
```

<!--
## Claude-specific notes
Always delegate.
-->
"""
        result = analyze_content(content, "CLAUDE.md")

        assert result is not None
        self.assertEqual(result.sections, ())

    def test_accepts_relative_agents_targets_but_not_inline_code(self) -> None:
        relative = analyze_content("@../AGENTS.md\n", "docs/CLAUDE.md")
        inline = analyze_content("See `@AGENTS.md` for context.\n", "CLAUDE.md")

        assert relative is not None
        self.assertEqual(relative.exact_literal_at_agents_occurrences, 0)
        self.assertEqual(relative.strict_import_only, 1)
        self.assertIsNone(inline)

    def test_heading_rule_avoids_dot_claude_paths(self) -> None:
        self.assertTrue(
            is_explicit_claude_overlay_heading(
                "claude code — additional instructions"
            )
        )
        self.assertTrue(
            is_explicit_claude_overlay_heading("claude-specific workflow")
        )
        self.assertFalse(is_explicit_claude_overlay_heading("claude.md"))
        self.assertFalse(
            is_explicit_claude_overlay_heading(
                "hard rules — see .claude/rules for details"
            )
        )
        self.assertFalse(
            is_explicit_claude_overlay_heading(
                "read-only glob search (claude-glob equivalent)"
            )
        )


class OverlayDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                repo_full_name TEXT NOT NULL,
                path TEXT NOT NULL,
                html_url TEXT,
                content_hash TEXT,
                size_bytes INTEGER,
                content TEXT
            );
            CREATE TABLE repos (
                full_name TEXT PRIMARY KEY,
                stars INTEGER,
                language TEXT
            );
            """
        )
        source_rows = [
            (1, "org/one", "CLAUDE.md", "@AGENTS.md\n"),
            (
                2,
                "org/two",
                "CLAUDE.md",
                "@AGENTS.md\n\n## Claude Code workflow\n"
                "Use plan mode and run `/review` before finishing.\n",
            ),
            (3, "org/two", "docs/claude.MD", "@../AGENTS.md\n"),
            (4, "org/three", "README.md", "@AGENTS.md\n"),
            (5, "org/four", "CLAUDE.md", "See `@AGENTS.md` for context.\n"),
        ]
        for file_id, repo, path, content in source_rows:
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            connection.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    repo,
                    path,
                    f"https://example.test/{file_id}",
                    digest,
                    len(content.encode("utf-8")),
                    content,
                ),
            )
        connection.executemany(
            "INSERT INTO repos VALUES (?, ?, ?)",
            [
                ("org/one", 1, "Python"),
                ("org/two", 2, "Rust"),
                ("org/three", 3, "Go"),
                ("org/four", 4, "Java"),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_analyzes_only_exact_claude_files_and_active_imports_read_only(self) -> None:
        result = analyze_database(self.db_path)

        self.assertEqual(result.exact_population_files, 4)
        self.assertEqual(result.exact_population_repositories, 3)
        self.assertEqual(len(result.files), 3)
        self.assertEqual(len(result.sections), 1)
        self.assertEqual(result.sqlite_total_changes, 0)
        self.assertEqual(
            [row.file_id for row in result.files],
            [1, 2, 3],
        )

        rows = {row.file_id: row for row in result.files}
        self.assertEqual(rows[1].strict_import_only, 1)
        self.assertEqual(rows[3].has_exact_literal_at_agents_import, 0)
        self.assertEqual(rows[2].has_concrete_labeled_directive_candidate, 1)
        self.assertEqual(rows[2].repository_exact_claude_files, 2)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                5,
            )
        finally:
            connection.close()

    def test_builds_summary_with_literal_and_overlay_cohorts(self) -> None:
        result = analyze_database(self.db_path)
        digest = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        summary = build_summary(
            result,
            database_path=self.db_path,
            database_sha256_before=digest,
            database_sha256_after=digest,
        )

        self.assertEqual(summary["source_population"]["files"], 4)
        self.assertEqual(
            summary["agents_target_basename_candidates"]["files"], 3
        )
        self.assertEqual(summary["exact_literal_at_agents_subset"]["files"], 2)
        self.assertEqual(
            summary["classification_cohorts"][
                "concrete_labeled_directive_candidate"
            ]["files"],
            1,
        )
        self.assertTrue(
            summary["invariants"][
                "exclusive_residual_partition_matches_candidate_files"
            ]
        )
        self.assertTrue(summary["database"]["sha256_unchanged"])


class OverlayOutputTests(unittest.TestCase):
    def test_defaults_target_article_artifacts(self) -> None:
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.db, "mined.db")
        self.assertEqual(
            arguments.files_output,
            "article/claude_agents_overlay_files.csv",
        )
        self.assertEqual(
            arguments.sections_output,
            "article/claude_agents_overlay_sections.csv",
        )
        self.assertEqual(
            arguments.summary_output,
            "article/claude_agents_overlay_summary.json",
        )
        self.assertEqual(arguments.substantive_word_threshold, 10)
        self.assertEqual(arguments.minimum_section_body_words, 3)

    def test_writers_are_deterministic_and_sections_can_be_empty(self) -> None:
        row = OverlayFileRow(
            file_id=1,
            repo_full_name="org/repo",
            source_path="CLAUDE.md",
            html_url="",
            content_hash="abc",
            size_bytes=11,
            repository_stars=0,
            repository_language="Python",
            repository_exact_claude_files=1,
            exact_population_content_copies=1,
            agents_import_occurrences=1,
            exact_literal_at_agents_occurrences=1,
            has_exact_literal_at_agents_import=1,
            target_form_counts_json=(
                '[{"occurrences":1,"raw_token":"@AGENTS.md"}]'
            ),
            first_agents_import_line=1,
            last_agents_import_line=1,
            strict_import_only=1,
            no_residual_outside_html_comments=1,
            residual_class="strict_import_only",
            residual_word_count=0,
            residual_after_final_import_word_count=0,
            substantive_residual=0,
            substantive_residual_after_final_import=0,
            explicit_claude_labeled_sections_after_final_import=0,
            labeled_sections_with_nonempty_body=0,
            labeled_sections_with_placeholder_language=0,
            concrete_labeled_directive_sections=0,
            has_concrete_labeled_directive_candidate=0,
            overlay_headings_json="[]",
            overlay_themes_json="[]",
            residual_excerpt="",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            files_path = write_files_csv([row], root / "files.csv")
            sections_path = write_sections_csv([], root / "sections.csv")
            left = write_summary_json({"b": 2, "a": 1}, root / "left.json")
            right = write_summary_json({"a": 1, "b": 2}, root / "right.json")

            with files_path.open(encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(written[0]["repo_full_name"], "org/repo")
            self.assertEqual(
                len(sections_path.read_text(encoding="utf-8").splitlines()),
                1,
            )
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(
                json.loads(left.read_text(encoding="utf-8")),
                {"a": 1, "b": 2},
            )


if __name__ == "__main__":
    unittest.main()
