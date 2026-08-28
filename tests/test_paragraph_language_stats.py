from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.paragraph_language_stats import (
    AnalysisConfig,
    Prediction,
    ProseParagraph,
    analyze_paragraph_languages,
    build_parser,
    classify_document,
    classify_paragraph,
    extract_prose_paragraphs,
)


class RuleDetector:
    version = "test"
    language_names = {"en": "English", "es": "Spanish"}

    def __init__(self, name: str, *, force_language: str | None = None):
        self.name = name
        self.force_language = force_language

    def predict(self, text: str) -> Prediction:
        if self.force_language is not None:
            language = self.force_language
        elif any(word in text.casefold() for word in ("este", "español", "proyecto")):
            language = "es"
        else:
            language = "en"
        return Prediction(language=language, confidence=0.99, margin=0.95)


class MarkdownProseExtractionTests(unittest.TestCase):
    def test_excludes_code_tables_and_short_markdown_tokens(self):
        content = """---
title: metadata
---
# Guide

This paragraph explains how contributors should build and test the project before submitting a proposed change to the repository.

```python
print("This code must not become prose")
```

| command | meaning |
| --- | --- |
| test | run it |

- Follow the detailed review instructions in `docs/review.md` before opening a pull request for the maintainers.
"""
        paragraphs = extract_prose_paragraphs(content)
        text = "\n".join(paragraph.text for paragraph in paragraphs)

        self.assertIn("contributors should build", text)
        self.assertIn("Follow the detailed review", text)
        self.assertNotIn("print", text)
        self.assertNotIn("command meaning", text)
        self.assertNotIn("metadata", text)
        self.assertNotIn("docs/review.md", text)


class ConservativeClassificationTests(unittest.TestCase):
    def setUp(self):
        self.config = AnalysisConfig(
            min_words=5,
            min_alpha_characters=20,
            min_document_accepted_words=5,
            min_document_coverage=0.5,
            multilingual_secondary_words=3,
        )
        self.paragraph = ProseParagraph(
            index=0,
            text=(
                "This paragraph explains the testing procedure clearly and "
                "asks contributors to review every proposed change."
            ),
            word_count=15,
            alphabetic_characters=90,
        )

    def test_accepts_only_agreement_that_is_stable(self):
        decision = classify_paragraph(
            self.paragraph,
            RuleDetector("fastText"),
            RuleDetector("Lingua"),
            self.config,
        )

        self.assertEqual(decision.decision, "accepted")
        self.assertEqual(decision.accepted_language, "en")
        self.assertTrue(decision.stable_after_normalization)

    def test_abstains_when_detectors_disagree(self):
        decision = classify_paragraph(
            self.paragraph,
            RuleDetector("fastText", force_language="en"),
            RuleDetector("Lingua", force_language="es"),
            self.config,
        )

        self.assertEqual(decision.decision, "detector_disagreement")
        self.assertIsNone(decision.accepted_language)

    def test_rejects_a_prediction_incompatible_with_dominant_script(self):
        russian = ProseParagraph(
            index=0,
            text=(
                "Этот абзац подробно объясняет правила тестирования проекта "
                "и порядок проверки предлагаемых изменений участниками."
            ),
            word_count=14,
            alphabetic_characters=95,
        )
        decision = classify_paragraph(
            russian,
            RuleDetector("fastText", force_language="en"),
            RuleDetector("Lingua", force_language="en"),
            self.config,
        )

        self.assertEqual(decision.decision, "script_mismatch")

    def test_document_aggregation_can_report_primary_multilingual_or_unknown(self):
        primary = classify_document(
            {"en": 900, "es": 100}, 1_100, self.config
        )
        multilingual = classify_document(
            {"en": 600, "es": 400}, 1_100, self.config
        )
        unknown = classify_document({}, 0, self.config)

        self.assertEqual(primary[0:2], ("primary", "en"))
        self.assertEqual(multilingual[0], "multilingual")
        self.assertEqual(unknown[-1], "no_eligible_prose")


class LanguageAnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary_directory.name) / "mined.db"
        english = (
            "# Guide\n\nThis paragraph explains how contributors should build "
            "and test the application before submitting changes to the repository.\n"
        )
        spanish = (
            "# Guía\n\nEste párrafo explica cómo los colaboradores deben compilar "
            "y probar el proyecto antes de enviar cambios al repositorio.\n"
        )
        english_hash = hashlib.sha256(english.encode()).hexdigest()
        spanish_hash = hashlib.sha256(spanish.encode()).hexdigest()
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """CREATE TABLE files (
                       id INTEGER PRIMARY KEY,
                       repo_full_name TEXT NOT NULL,
                       path TEXT NOT NULL,
                       html_url TEXT,
                       content_hash TEXT,
                       size_bytes INTEGER,
                       content TEXT NOT NULL
                   )"""
            )
            connection.executemany(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (1, "a/repo", "CLAUDE.md", "https://example/a", english_hash, len(english), english),
                    (2, "b/repo", "docs/CLAUDE.md", "https://example/b", english_hash, len(english), english),
                    (3, "c/repo", "CLAUDE.md", "https://example/c", spanish_hash, len(spanish), spanish),
                    (4, "d/repo", "CLAUDE.md.template", None, spanish_hash, len(spanish), spanish),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_weights_duplicates_and_never_changes_the_database(self):
        config = AnalysisConfig(
            min_words=5,
            min_alpha_characters=20,
            min_document_accepted_words=5,
            min_document_coverage=0.5,
            multilingual_secondary_words=3,
        )
        summary, families, distribution, audit = analyze_paragraph_languages(
            self.db_path,
            RuleDetector("fastText"),
            RuleDetector("Lingua"),
            config=config,
            audit_top_languages=2,
            audit_accepted_per_language=1,
            audit_disagreements=1,
            audit_threshold_rejections=1,
            audit_stability_or_script=1,
        )

        self.assertEqual(summary["scoped_files"], 3)
        self.assertEqual(summary["content_families"], 2)
        self.assertEqual(summary["full_population"]["files"], 3)
        self.assertEqual(summary["unique_content_population"]["files"], 2)
        self.assertEqual(
            summary["full_population"]["documents"]["primary_languages"],
            {"en": 2, "es": 1},
        )
        self.assertEqual(len(families), 2)
        self.assertEqual([row["language_code"] for row in distribution], ["en", "es"])
        self.assertEqual(len(audit), 2)

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM files").fetchone()[0], 4)
        finally:
            connection.close()

    def test_parser_defaults_to_exact_claude(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.scope, "exact-claude")
        self.assertEqual(arguments.min_confidence, 0.90)


if __name__ == "__main__":
    unittest.main()
