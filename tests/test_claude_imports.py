from __future__ import annotations

import unittest

from miner.claude_imports import (
    MAX_DOCUMENTED_IMPORT_HOPS,
    extract_claude_import_occurrences,
    resolve_import_target,
    target_shape,
)


class ClaudeImportLexerTests(unittest.TestCase):
    def test_detects_documented_shapes_anywhere_without_extension_filter(self):
        occurrences = extract_claude_import_occurrences(
            "@AGENTS.md\n"
            "See @README for the overview and @package.json for commands.\n"
            "- workflow @docs/git-instructions\n"
        )

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("AGENTS.md", "import_candidate"),
                ("README", "import_candidate"),
                ("package.json", "import_candidate"),
                ("docs/git-instructions", "import_candidate"),
            ],
        )
        self.assertEqual(occurrences[0].surface_form, "whole_line")
        self.assertEqual(occurrences[-1].surface_form, "prose_embedded")

    def test_excludes_inline_code_with_equal_and_multiline_backtick_runs(self):
        content = (
            "Literal `@README` and ``@package.json`` but active @AGENTS.md.\n"
            "Multiline ``code @first\nstill @second`` then @third.\n"
        )

        occurrences = extract_claude_import_occurrences(content)

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("README", "excluded_inline_code"),
                ("package.json", "excluded_inline_code"),
                ("AGENTS.md", "import_candidate"),
                ("first", "excluded_inline_code"),
                ("second", "excluded_inline_code"),
                ("third", "import_candidate"),
            ],
        )

    def test_unmatched_backtick_does_not_hide_following_candidate(self):
        occurrences = extract_claude_import_occurrences("Unclosed ` marker @README\n")

        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0].decision, "import_candidate")

    def test_inline_code_span_cannot_bridge_fence_or_comment_barriers(self):
        content = (
            "Unmatched ` before tilde fence\n"
            "~~~text\nno backticks here\n~~~\n"
            "@after-fence `\n"
            "Unmatched ` before comment\n"
            "<!-- no backticks here -->\n"
            "@after-comment `\n"
            "Unmatched ` before blank line\n"
            "\n"
            "@after-blank `\n"
        )

        occurrences = extract_claude_import_occurrences(content)

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("after-fence", "import_candidate"),
                ("after-comment", "import_candidate"),
                ("after-blank", "import_candidate"),
            ],
        )

    def test_inline_code_takes_precedence_over_inline_html_comment_markers(self):
        occurrences = extract_claude_import_occurrences(
            "`before <!-- @inside-comment --> and @inside-code`\n"
        )

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("inside-comment", "excluded_inline_code"),
                ("inside-code", "excluded_inline_code"),
            ],
        )

    def test_multiline_block_comment_remains_a_barrier_across_blank_lines(self):
        occurrences = extract_claude_import_occurrences(
            "Unmatched ` before comment\n"
            "<!--\n\n@inside-block-comment\n-->\n"
            "@after-block-comment `\n"
        )

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("inside-block-comment", "excluded_html_comment"),
                ("after-block-comment", "import_candidate"),
            ],
        )

    def test_excludes_backtick_and_tilde_fences_and_html_comments(self):
        content = """```text
@fenced.json
```
~~~
@also-fenced
~~~
<!-- import: @commented.md -->
@active.md
"""

        occurrences = extract_claude_import_occurrences(content)

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("fenced.json", "excluded_fenced_code"),
                ("also-fenced", "excluded_fenced_code"),
                ("commented.md", "excluded_html_comment"),
                ("active.md", "import_candidate"),
            ],
        )

    def test_filters_email_addresses_but_retains_whitespace_delimited_mentions(self):
        occurrences = extract_claude_import_occurrences(
            "Email dev@example.com, identifier foo@bar, candidate @Claude.\n"
        )

        self.assertEqual([item.raw_target for item in occurrences], ["Claude"])

    def test_handles_multiple_targets_and_sentence_punctuation(self):
        occurrences = extract_claude_import_occurrences(
            "Use @README, @docs/rules.md; then @package.json.\n"
        )

        self.assertEqual(
            [item.raw_target for item in occurrences],
            ["README", "docs/rules.md", "package.json"],
        )

    def test_preserves_nested_at_signs_in_filesystem_paths(self):
        occurrences = extract_claude_import_occurrences(
            "@./node_modules/@scope/package/CLAUDE.md\n"
        )

        self.assertEqual(len(occurrences), 1)
        self.assertEqual(
            occurrences[0].raw_target,
            "./node_modules/@scope/package/CLAUDE.md",
        )
        self.assertEqual(occurrences[0].decision, "import_candidate")

    def test_preserves_but_excludes_undocumented_patterns_and_variables(self):
        occurrences = extract_claude_import_occurrences(
            "@docs/*.md\n@{{ROOT}}/rules.md\n@$HOME/rules.md\n"
        )

        self.assertEqual(
            [(item.raw_target, item.decision) for item in occurrences],
            [
                ("docs/*.md", "excluded_unsupported_pattern"),
                ("{{ROOT}}/rules.md", "excluded_unsupported_pattern"),
                ("$HOME/rules.md", "excluded_unsupported_variable"),
            ],
        )

    def test_allows_markdown_emphasis_around_an_import_token(self):
        occurrence = extract_claude_import_occurrences("**@AGENTS.md**\n")[0]

        self.assertEqual(occurrence.raw_target, "AGENTS.md")
        self.assertEqual(occurrence.decision, "import_candidate")
        self.assertEqual(occurrence.surface_form, "standalone_formatted")

    def test_classifies_urls_as_non_filesystem_candidates(self):
        occurrences = extract_claude_import_occurrences(
            "Do not import @https://example.com/rules.md but use @rules.md.\n"
        )

        self.assertEqual(occurrences[0].path_kind, "unsupported_uri")
        self.assertEqual(occurrences[0].decision, "excluded_unsupported_uri")
        self.assertEqual(occurrences[1].decision, "import_candidate")


class ImportTargetResolutionTests(unittest.TestCase):
    def test_resolves_relative_target_against_containing_file(self):
        resolution = resolve_import_target(
            "packages/api/CLAUDE.md",
            "../../AGENTS.md",
        )

        self.assertEqual(resolution.normalized_target, "../../AGENTS.md")
        self.assertEqual(resolution.resolved_target, "AGENTS.md")
        self.assertEqual(resolution.path_relation, "ancestor_directory")
        self.assertEqual(resolution.target_extension_class, "markdown")

    def test_distinguishes_extensionless_other_extension_and_self_reference(self):
        readme = resolve_import_target("CLAUDE.md", "README")
        package = resolve_import_target("CLAUDE.md", "package.json")
        self_reference = resolve_import_target("docs/CLAUDE.md", "CLAUDE.md")

        self.assertEqual(readme.target_extension_class, "extensionless")
        self.assertEqual(package.target_extension_class, "other_extension")
        self.assertEqual(self_reference.is_self_reference, 1)
        self.assertEqual(target_shape("README", "extensionless"), "bare_extensionless")
        self.assertEqual(
            target_shape("docs/workflow", "extensionless"),
            "path_extensionless",
        )

    def test_treats_leading_slash_and_home_paths_as_external(self):
        absolute = resolve_import_target("CLAUDE.md", "/etc/team-rules")
        home = resolve_import_target("CLAUDE.md", "~/.claude/preferences.md")

        self.assertEqual(absolute.path_kind, "posix_absolute")
        self.assertEqual(absolute.path_relation, "external_absolute")
        self.assertEqual(home.path_kind, "home_relative")
        self.assertEqual(home.path_relation, "external_home")

    def test_preserves_windows_paths_as_platform_sensitive(self):
        absolute = resolve_import_target("CLAUDE.md", r"C:\team\rules.md")
        relative = resolve_import_target("CLAUDE.md", r"docs\rules.md")

        self.assertEqual(absolute.path_kind, "windows_absolute")
        self.assertEqual(absolute.path_relation, "external_absolute")
        self.assertEqual(absolute.target_basename, "rules.md")
        self.assertEqual(relative.path_kind, "platform_relative")
        self.assertEqual(relative.path_relation, "platform_dependent")

    def test_marks_relative_escape_outside_repository(self):
        resolution = resolve_import_target("CLAUDE.md", "../shared/rules.md")

        self.assertEqual(resolution.path_relation, "outside_repository")

    def test_normalizes_unsupported_relative_patterns_for_comparison(self):
        resolution = resolve_import_target(
            "CLAUDE.md",
            "./.github/instructions/*.md",
        )

        self.assertEqual(resolution.path_kind, "unsupported_pattern")
        self.assertEqual(resolution.normalized_target, ".github/instructions/*.md")

    def test_records_documented_recursion_limit(self):
        self.assertEqual(MAX_DOCUMENTED_IMPORT_HOPS, 4)


if __name__ == "__main__":
    unittest.main()
