from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_repeated_content_audit import (
    build_parser,
    summarize_audit,
    write_summary,
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RepeatedContentAuditSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.audit_path = root / "audit.csv"
        self.labels_path = root / "labels.csv"
        _write_csv(
            self.audit_path,
            ("audit_id", "content_hash", "auto_category"),
            [
                {"audit_id": "RC001", "content_hash": "a", "auto_category": "pointer_shim"},
                {"audit_id": "RC002", "content_hash": "b", "auto_category": "substantive_repeated_document"},
                {"audit_id": "RC003", "content_hash": "c", "auto_category": "short_ambiguous"},
            ],
        )
        _write_csv(
            self.labels_path,
            (
                "audit_id",
                "content_hash",
                "review_disposition",
                "review_category",
                "notes",
            ),
            [
                {
                    "audit_id": "RC001",
                    "content_hash": "a",
                    "review_disposition": "accepted_correct",
                    "review_category": "pointer_shim",
                    "notes": "",
                },
                {
                    "audit_id": "RC002",
                    "content_hash": "b",
                    "review_disposition": "accepted_incorrect",
                    "review_category": "generated_context",
                    "notes": "",
                },
                {
                    "audit_id": "RC003",
                    "content_hash": "c",
                    "review_disposition": "classifier_abstention",
                    "review_category": "substantive_repeated_document",
                    "notes": "",
                },
            ],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_summarizes_accepted_precision_and_abstention_assignments(self):
        summary = summarize_audit(self.audit_path, self.labels_path)

        self.assertEqual(summary["audit_rows"], 3)
        self.assertEqual(summary["accepted_labels"], 2)
        self.assertEqual(summary["accepted_correct"], 1)
        self.assertEqual(summary["accepted_incorrect"], 1)
        self.assertEqual(summary["accepted_precision"], 0.5)
        self.assertEqual(summary["classifier_abstentions"], 1)
        self.assertEqual(
            summary["abstention_review_categories"],
            [{"review_category": "substantive_repeated_document", "rows": 1}],
        )

        output = write_summary(
            summary,
            Path(self.temporary_directory.name) / "summary.json",
        )
        loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["accepted_correct"], 1)

    def test_rejects_hash_mismatch(self):
        with self.labels_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["content_hash"] = "wrong"
        _write_csv(
            self.labels_path,
            tuple(rows[0]),
            rows,
        )

        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            summarize_audit(self.audit_path, self.labels_path)

    def test_parser_defaults_to_phase1b_artifacts(self):
        arguments = build_parser().parse_args([])

        self.assertEqual(arguments.audit, "article/claude_repeated_content_audit.csv")
        self.assertEqual(
            arguments.labels,
            "article/claude_repeated_content_audit_labels.csv",
        )
        self.assertEqual(
            arguments.output,
            "article/claude_repeated_content_audit_summary.json",
        )


if __name__ == "__main__":
    unittest.main()
