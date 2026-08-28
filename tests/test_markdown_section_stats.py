from __future__ import annotations

import hashlib
import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.markdown_section_stats import (
    analyze_markdown_sections,
    build_parser,
    normalize_heading_name,
    parse_document_structure,
    parse_headings,
    write_content_style_csv,
    write_list_item_distribution_csv,
)


class MarkdownHeadingParserTests(unittest.TestCase):
    def test_parses_atx_and_setext_but_ignores_fenced_headings(self):
        content = """# Title
intro words
## Alpha
one two
```markdown
# ignored
```
#### Deep
deep words
Beta
----
beta body
"""

        headings = parse_headings(content)
        structure = parse_document_structure(content)

        self.assertEqual([heading.text for heading in headings], ["Title", "Alpha", "Deep", "Beta"])
        self.assertEqual([heading.level for heading in headings], [1, 2, 4, 2])
        self.assertEqual(structure.top_level_sections, 1)
        self.assertEqual(structure.maximum_relative_depth, 3)
        self.assertEqual(structure.skipped_level_jumps, 1)
        self.assertEqual(structure.h1_count, 1)
        self.assertEqual(len(structure.section_word_counts), 4)
        self.assertEqual(structure.section_word_counts[-1], 2)

    def test_counts_multiple_root_sections_when_no_lower_parent_is_open(self):
        structure = parse_document_structure("## First\ntext\n## Second\ntext\n")

        self.assertEqual(structure.top_level_sections, 2)
        self.assertEqual(structure.maximum_relative_depth, 1)
        self.assertTrue(structure.starts_below_h1)
        self.assertEqual(structure.h1_count, 0)

    def test_normalizes_numbering_links_and_formatting(self):
        self.assertEqual(normalize_heading_name("2. **Testing:**"), "testing")
        self.assertEqual(normalize_heading_name("A Project Guide"), "a project guide")
        self.assertEqual(
            normalize_heading_name("[Project Architecture](docs/architecture.md)"),
            "project architecture",
        )

    def test_classifies_list_prose_and_other_section_content(self):
        content = """# List only
- first item
  wrapped continuation
  - nested item
1. ordered item
- [x] completed task

## Mixed
Introductory prose
continues here
- one item

## Other
```text
- ignored code marker
```
| name | value |
|---|---|

## Prose
First paragraph
continues here

Second paragraph
"""

        structure = parse_document_structure(content)
        list_only, mixed, other, prose = structure.section_bodies

        self.assertEqual(list_only.content_style, "list_only")
        self.assertEqual(list_only.list_item_count, 4)
        self.assertEqual(list_only.unordered_list_item_count, 3)
        self.assertEqual(list_only.ordered_list_item_count, 1)
        self.assertEqual(list_only.task_list_item_count, 1)
        self.assertEqual(list_only.list_content_line_count, 5)
        self.assertEqual(list_only.standalone_prose_block_count, 0)

        self.assertEqual(mixed.content_style, "list_and_prose")
        self.assertEqual(mixed.list_item_count, 1)
        self.assertEqual(mixed.standalone_prose_block_count, 1)

        self.assertEqual(other.content_style, "other_only")
        self.assertEqual(other.list_item_count, 0)
        self.assertEqual(other.prose_content_line_count, 0)

        self.assertEqual(prose.content_style, "prose_only")
        self.assertEqual(prose.standalone_prose_block_count, 2)

    def test_keeps_indented_and_lazy_continuations_in_their_list(self):
        structure = parse_document_structure(
            "# Lists\n- item\n\n  indented continuation\n- next\nlazy continuation\n"
        )

        body = structure.section_bodies[0]
        self.assertEqual(body.content_style, "list_only")
        self.assertEqual(body.list_item_count, 2)
        self.assertEqual(body.list_content_line_count, 4)

    def test_does_not_count_indented_or_fenced_code_as_list_items(self):
        structure = parse_document_structure(
            "# Code\n    - indented code\n~~~text\n1. fenced code\n~~~\n"
        )

        body = structure.section_bodies[0]
        self.assertEqual(body.content_style, "other_only")
        self.assertEqual(body.list_item_count, 0)

    def test_classifies_blockquotes_and_fences_contained_by_lists(self):
        structure = parse_document_structure(
            "# Quote\n> - quoted item\n>   continuation\n"
            "## List fence\n- ```text\n  - code marker\n  ```\n- real item\n"
        )

        quoted, fenced = structure.section_bodies
        self.assertEqual(quoted.content_style, "list_only")
        self.assertEqual(quoted.list_item_count, 1)
        self.assertEqual(quoted.list_content_line_count, 2)
        self.assertEqual(fenced.content_style, "list_only")
        self.assertEqual(fenced.list_item_count, 2)
        self.assertEqual(fenced.list_content_line_count, 4)


class MarkdownSectionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        shared = "# Title\nintro\n## Testing\n- run tests\n"
        unique = "# Other\nbody\n"
        shared_hash = hashlib.sha256(shared.encode()).hexdigest()
        unique_hash = hashlib.sha256(unique.encode()).hexdigest()
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       id INTEGER PRIMARY KEY,
                       repo_full_name TEXT NOT NULL,
                       path TEXT NOT NULL,
                       content_hash TEXT,
                       size_bytes INTEGER,
                       content TEXT NOT NULL
                   )"""
            )
            connection.executemany(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "a/repo", "CLAUDE.md", shared_hash, len(shared), shared),
                    (2, "b/repo", "docs/CLAUDE.md", shared_hash, len(shared), shared),
                    (3, "c/repo", "CLAUDE.md", unique_hash, len(unique), unique),
                    (4, "d/repo", "CLAUDE.md.template", unique_hash, len(unique), unique),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_weights_full_population_and_reports_unique_content(self):
        summary, families, headings = analyze_markdown_sections(self.db_path)

        full = summary["full_population"]
        unique = summary["unique_content_population"]
        self.assertEqual(summary["scoped_files"], 3)
        self.assertEqual(summary["content_families"], 2)
        self.assertEqual(full["files"], 3)
        self.assertEqual(unique["files"], 2)
        self.assertEqual(full["explicit_sections"], 5)
        self.assertEqual(unique["explicit_sections"], 3)
        self.assertEqual(full["headed_files_without_h1"], 0)
        self.assertEqual(full["sections_per_headed_file"]["count"], 3)
        self.assertEqual(len(families), 2)

        full_content = full["section_content"]
        unique_content = unique["section_content"]
        self.assertEqual(full_content["sections_with_list_items"], 2)
        self.assertEqual(unique_content["sections_with_list_items"], 1)
        self.assertEqual(full_content["list_items"]["total"], 2)
        self.assertEqual(full_content["styles"]["list_only"]["sections"], 2)
        self.assertEqual(full_content["styles"]["prose_only"]["sections"], 3)
        self.assertEqual(
            sum(item["sections"] for item in full_content["styles"].values()),
            full["explicit_sections"],
        )

        by_name = {item["normalized_heading"]: item for item in headings}
        self.assertEqual(by_name["title"]["files_full"], 2)
        self.assertEqual(by_name["title"]["unique_contents"], 1)
        self.assertEqual(by_name["testing"]["occurrences_full"], 2)
        self.assertEqual(by_name["other"]["files_full"], 1)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM files").fetchone()[0], 4)
        finally:
            connection.close()

        styles_path = Path(self.temporary_directory.name) / "styles.csv"
        distribution_path = Path(self.temporary_directory.name) / "distribution.csv"
        write_content_style_csv(summary, styles_path)
        write_list_item_distribution_csv(summary, distribution_path)
        with styles_path.open(encoding="utf-8", newline="") as handle:
            style_rows = list(csv.DictReader(handle))
        with distribution_path.open(encoding="utf-8", newline="") as handle:
            distribution_rows = list(csv.DictReader(handle))
        self.assertEqual(len(style_rows), 8)
        self.assertEqual(
            next(row for row in style_rows if row["content_style"] == "list_only")[
                "sections_full"
            ],
            "2",
        )
        self.assertEqual(
            next(row for row in distribution_rows if row["list_item_count"] == "1")[
                "sections_full"
            ],
            "2",
        )

    def test_parser_defaults_to_exact_claude(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(arguments.heading_limit, 1000)
        self.assertEqual(arguments.styles_output, "section_content_styles.csv")
        self.assertEqual(
            arguments.list_distribution_output,
            "section_list_item_distribution.csv",
        )


if __name__ == "__main__":
    unittest.main()
