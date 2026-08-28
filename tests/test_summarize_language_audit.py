from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_language_audit import summarize_audit, wilson_interval


class LanguageAuditSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temporary_directory.name) / "audit.csv"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_rows(self, rows):
        fieldnames = (
            "audit_stratum",
            "accepted_language",
            "fasttext_language",
            "lingua_language",
            "manual_language",
            "manual_judgment",
        )
        with self.audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_summarizes_accepted_precision_and_abstentions(self):
        self.write_rows(
            [
                {
                    "audit_stratum": "accepted:en",
                    "accepted_language": "en",
                    "fasttext_language": "en",
                    "lingua_language": "en",
                    "manual_language": "en",
                    "manual_judgment": "",
                },
                {
                    "audit_stratum": "accepted:es",
                    "accepted_language": "es",
                    "fasttext_language": "es",
                    "lingua_language": "es",
                    "manual_language": "pt",
                    "manual_judgment": "",
                },
                {
                    "audit_stratum": "detector_disagreement",
                    "accepted_language": "",
                    "fasttext_language": "en",
                    "lingua_language": "de",
                    "manual_language": "en",
                    "manual_judgment": "",
                },
                {
                    "audit_stratum": "threshold_rejection",
                    "accepted_language": "",
                    "fasttext_language": "fr",
                    "lingua_language": "fr",
                    "manual_language": "",
                    "manual_judgment": "",
                },
            ]
        )

        summary = summarize_audit(self.audit_path)

        self.assertEqual(summary["reviewed_rows"], 3)
        self.assertEqual(summary["accepted_labels"]["correct"], 1)
        self.assertEqual(summary["accepted_labels"]["incorrect"], 1)
        self.assertEqual(summary["accepted_labels"]["observed_precision"], 0.5)
        self.assertEqual(
            summary["abstentions"]["detector_matches_manual_language"],
            {"fasttext_only": 1},
        )

    def test_empty_review_remains_pending(self):
        self.write_rows([])

        summary = summarize_audit(self.audit_path)

        self.assertEqual(summary["status"], "awaiting_manual_review")
        self.assertIsNone(summary["accepted_labels"]["observed_precision"])

    def test_reads_excel_style_cp1252_semicolon_csv(self):
        fieldnames = (
            "audit_stratum",
            "accepted_language",
            "fasttext_language",
            "lingua_language",
            "manual_language",
            "manual_judgment",
            "manual_notes",
        )
        with self.audit_path.open("w", encoding="cp1252", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerow(
                {
                    "audit_stratum": "accepted:es",
                    "accepted_language": "es",
                    "fasttext_language": "es",
                    "lingua_language": "es",
                    "manual_language": "es",
                    "manual_judgment": "",
                    "manual_notes": "revisé",
                }
            )

        summary = summarize_audit(self.audit_path)

        self.assertEqual(summary["reviewed_rows"], 1)
        self.assertEqual(summary["accepted_labels"]["correct"], 1)
        self.assertEqual(summary["input_format"]["encoding"], "cp1252")
        self.assertEqual(summary["input_format"]["delimiter"], "semicolon")

    def test_wilson_interval_is_bounded(self):
        interval = wilson_interval(10, 10)

        self.assertIsNotNone(interval)
        self.assertGreater(interval[0], 0.7)
        self.assertAlmostEqual(interval[1], 1.0)


if __name__ == "__main__":
    unittest.main()
