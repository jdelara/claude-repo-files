"""Summarize completed manual checks in ``language_manual_audit.csv``.

Reviewers can fill either ``manual_language`` (preferably an ISO 639-1 code,
plus ``mixed``/``unknown`` when needed), ``manual_judgment``
(``correct``, ``incorrect``, or ``uncertain``), or both. The script reports the
precision of accepted automated labels on reviewed, decidable rows and uses a
Wilson 95% interval. Abstained rows are summarized separately because an
abstention is not itself a wrong language label.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


JUDGMENT_ALIASES = {
    "correct": "correct",
    "yes": "correct",
    "y": "correct",
    "true": "correct",
    "1": "correct",
    "incorrect": "incorrect",
    "no": "incorrect",
    "n": "incorrect",
    "false": "incorrect",
    "0": "incorrect",
    "uncertain": "uncertain",
    "unsure": "uncertain",
    "?": "uncertain",
}
LANGUAGE_ALIASES = {"nb": "no"}


def normalize_language(value: str) -> str:
    code = value.strip().casefold().replace("_", "-")
    return LANGUAGE_ALIASES.get(code, code)


def normalize_judgment(value: str) -> str:
    cleaned = value.strip().casefold()
    if not cleaned:
        return ""
    if cleaned not in JUDGMENT_ALIASES:
        raise ValueError(
            f"unsupported manual_judgment {value!r}; use correct, incorrect, "
            "or uncertain"
        )
    return JUDGMENT_ALIASES[cleaned]


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def read_audit_rows(audit_path: str | Path) -> tuple[list[dict[str, str]], str, str]:
    """Read UTF-8/comma or Excel-style Windows-1252/semicolon CSV files."""
    path = Path(audit_path)
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = data.decode("cp1252")
        encoding = "cp1252"

    header = text.splitlines()[0] if text.splitlines() else ""
    candidates = {",": header.count(","), ";": header.count(";"), "\t": header.count("\t")}
    delimiter = max(candidates, key=candidates.get)
    if candidates[delimiter] == 0:
        raise ValueError("could not detect comma, semicolon, or tab CSV delimiter")

    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    required = {
        "audit_stratum",
        "accepted_language",
        "fasttext_language",
        "lingua_language",
        "manual_language",
        "manual_judgment",
    }
    present = set(rows[0]) if rows else set(header.split(delimiter))
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"audit CSV is missing columns: {', '.join(missing)}")
    return rows, encoding, delimiter


def summarize_audit(audit_path: str | Path) -> dict[str, Any]:
    rows, encoding, delimiter = read_audit_rows(audit_path)

    reviewed = 0
    uncertain = 0
    reviewed_by_stratum: Counter[str] = Counter()
    accepted_reviewed = 0
    accepted_correct = 0
    accepted_incorrect = 0
    accepted_by_language: dict[str, Counter[str]] = {}
    abstained_reviewed = 0
    abstention_manual_languages: Counter[str] = Counter()
    abstention_detector_matches: Counter[str] = Counter()

    for row_number, row in enumerate(rows, 2):
        manual_language = normalize_language(row.get("manual_language", ""))
        try:
            judgment = normalize_judgment(row.get("manual_judgment", ""))
        except ValueError as exc:
            raise ValueError(f"row {row_number}: {exc}") from exc
        if not manual_language and not judgment:
            continue

        reviewed += 1
        stratum = row.get("audit_stratum", "") or "(missing)"
        reviewed_by_stratum[stratum] += 1
        accepted_language = normalize_language(row.get("accepted_language", ""))
        if judgment == "uncertain" or manual_language in {"unknown", "uncertain"}:
            uncertain += 1
            continue

        if accepted_language:
            accepted_reviewed += 1
            if judgment:
                is_correct = judgment == "correct"
            elif manual_language:
                is_correct = manual_language == accepted_language
            else:
                continue
            outcome = "correct" if is_correct else "incorrect"
            accepted_by_language.setdefault(
                accepted_language, Counter()
            )[outcome] += 1
            if is_correct:
                accepted_correct += 1
            else:
                accepted_incorrect += 1
            continue

        abstained_reviewed += 1
        if manual_language:
            abstention_manual_languages[manual_language] += 1
            fasttext = normalize_language(row.get("fasttext_language", ""))
            lingua = normalize_language(row.get("lingua_language", ""))
            if manual_language == fasttext == lingua:
                abstention_detector_matches["both"] += 1
            elif manual_language == fasttext:
                abstention_detector_matches["fasttext_only"] += 1
            elif manual_language == lingua:
                abstention_detector_matches["lingua_only"] += 1
            else:
                abstention_detector_matches["neither"] += 1

    decisive_accepted = accepted_correct + accepted_incorrect
    interval = wilson_interval(accepted_correct, decisive_accepted)
    status = "review_complete_or_in_progress" if reviewed else "awaiting_manual_review"
    return {
        "status": status,
        "input_format": {
            "encoding": encoding,
            "delimiter": {",": "comma", ";": "semicolon", "\t": "tab"}[
                delimiter
            ],
        },
        "audit_rows": len(rows),
        "reviewed_rows": reviewed,
        "unreviewed_rows": len(rows) - reviewed,
        "uncertain_reviewed_rows": uncertain,
        "reviewed_by_stratum": dict(
            sorted(reviewed_by_stratum.items(), key=lambda item: (-item[1], item[0]))
        ),
        "accepted_labels": {
            "reviewed": accepted_reviewed,
            "decisive": decisive_accepted,
            "correct": accepted_correct,
            "incorrect": accepted_incorrect,
            "observed_precision": (
                accepted_correct / decisive_accepted if decisive_accepted else None
            ),
            "wilson_95_percent_interval": list(interval) if interval else None,
            "by_language": {
                language: dict(counts)
                for language, counts in sorted(accepted_by_language.items())
            },
        },
        "abstentions": {
            "reviewed": abstained_reviewed,
            "manual_languages": dict(abstention_manual_languages.most_common()),
            "detector_matches_manual_language": dict(
                abstention_detector_matches.most_common()
            ),
        },
    }


def write_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize manual paragraph-language audit entries."
    )
    parser.add_argument(
        "--audit",
        default="article/language_manual_audit.csv",
        help="audit CSV generated by paragraph_language_stats.py",
    )
    parser.add_argument(
        "--output",
        default="article/language_manual_audit_summary.json",
        help="JSON summary output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_audit(args.audit)
        output = write_summary(summary, args.output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    accepted = summary["accepted_labels"]
    print(f"Audit rows: {summary['audit_rows']}")
    print(f"Reviewed rows: {summary['reviewed_rows']}")
    print(f"Accepted labels with decisive review: {accepted['decisive']}")
    if accepted["observed_precision"] is not None:
        print(f"Observed accepted-label precision: {accepted['observed_precision']:.2%}")
    print(f"Summary JSON: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
