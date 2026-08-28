"""Validate and summarize the manual Phase 1b repeated-content audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


CATEGORIES = (
    "pointer_shim",
    "empty_placeholder",
    "generated_context",
    "template_scaffold",
    "substantive_repeated_document",
    "short_ambiguous",
)
DISPOSITIONS = {
    "accepted_correct",
    "accepted_incorrect",
    "classifier_abstention",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV not found: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize_audit(
    audit_path: str | Path,
    labels_path: str | Path,
) -> dict[str, object]:
    audit = _read_csv(audit_path)
    labels = _read_csv(labels_path)
    audit_by_id = {row["audit_id"]: row for row in audit}
    labels_by_id = {row["audit_id"]: row for row in labels}
    if len(audit_by_id) != len(audit):
        raise ValueError("duplicate audit_id in generated audit")
    if len(labels_by_id) != len(labels):
        raise ValueError("duplicate audit_id in labels")
    missing = sorted(set(audit_by_id) - set(labels_by_id))
    extra = sorted(set(labels_by_id) - set(audit_by_id))
    if missing or extra:
        raise ValueError(f"audit/label ID mismatch: missing={missing}, extra={extra}")

    joined: list[dict[str, str]] = []
    for audit_id in sorted(audit_by_id):
        generated = audit_by_id[audit_id]
        reviewed = labels_by_id[audit_id]
        if generated["content_hash"] != reviewed["content_hash"]:
            raise ValueError(f"content hash mismatch for {audit_id}")
        disposition = reviewed["review_disposition"]
        review_category = reviewed["review_category"]
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unsupported disposition for {audit_id}: {disposition}")
        if review_category not in CATEGORIES:
            raise ValueError(
                f"unsupported review category for {audit_id}: {review_category}"
            )
        if (
            disposition == "accepted_correct"
            and generated["auto_category"] != review_category
        ):
            raise ValueError(f"correct label disagrees with auto category for {audit_id}")
        if (
            disposition == "accepted_incorrect"
            and generated["auto_category"] == review_category
        ):
            raise ValueError(f"incorrect label equals auto category for {audit_id}")
        joined.append({**generated, **reviewed})

    accepted = [
        row for row in joined if row["review_disposition"].startswith("accepted_")
    ]
    correct = [
        row for row in accepted if row["review_disposition"] == "accepted_correct"
    ]
    incorrect = [
        row for row in accepted if row["review_disposition"] == "accepted_incorrect"
    ]
    abstentions = [
        row for row in joined if row["review_disposition"] == "classifier_abstention"
    ]
    lower, upper = _wilson_interval(len(correct), len(accepted))

    by_category = []
    for category in CATEGORIES:
        sampled = [row for row in joined if row["auto_category"] == category]
        category_accepted = [
            row
            for row in sampled
            if row["review_disposition"].startswith("accepted_")
        ]
        category_correct = [
            row
            for row in category_accepted
            if row["review_disposition"] == "accepted_correct"
        ]
        by_category.append(
            {
                "auto_category": category,
                "sampled": len(sampled),
                "accepted_labels": len(category_accepted),
                "correct": len(category_correct),
                "incorrect": len(category_accepted) - len(category_correct),
                "classifier_abstentions": sum(
                    row["review_disposition"] == "classifier_abstention"
                    for row in sampled
                ),
                "accepted_precision": (
                    len(category_correct) / len(category_accepted)
                    if category_accepted
                    else None
                ),
            }
        )

    confusion = Counter(
        (row["auto_category"], row["review_category"]) for row in accepted
    )
    abstention_categories = Counter(
        row["review_category"] for row in abstentions
    )
    return {
        "audit_rows": len(joined),
        "sampling": (
            "deterministic category-by-size stratified sample; not proportional "
            "to content, group, file, or repository prevalence"
        ),
        "accepted_labels": len(accepted),
        "accepted_correct": len(correct),
        "accepted_incorrect": len(incorrect),
        "accepted_precision": len(correct) / len(accepted) if accepted else None,
        "accepted_precision_wilson_95": {
            "lower": lower,
            "upper": upper,
        },
        "classifier_abstentions": len(abstentions),
        "by_auto_category": by_category,
        "accepted_confusion": [
            {
                "auto_category": auto,
                "review_category": review,
                "rows": count,
            }
            for (auto, review), count in sorted(confusion.items())
        ],
        "abstention_review_categories": [
            {"review_category": category, "rows": count}
            for category, count in sorted(abstention_categories.items())
        ],
        "interpretation": (
            "accepted-label precision is a targeted single-reviewer diagnostic; "
            "abstention assignments diagnose coverage and are not counted as errors"
        ),
    }


def write_summary(summary: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        default="article/claude_repeated_content_audit.csv",
    )
    parser.add_argument(
        "--labels",
        default="article/claude_repeated_content_audit_labels.csv",
    )
    parser.add_argument(
        "--output",
        default="article/claude_repeated_content_audit_summary.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_audit(args.audit, args.labels)
        output = write_summary(summary, args.output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Accepted labels: {summary['accepted_correct']}/{summary['accepted_labels']} "
        f"correct ({summary['accepted_precision']:.2%})"
    )
    print(f"Classifier abstentions reviewed: {summary['classifier_abstentions']}")
    print(f"Summary JSON: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
