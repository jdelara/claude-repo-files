"""Estimate ``CLAUDE.md`` token and context-window occupancy from stored data.

The analysis is deliberately offline and read-only.  It converts the stored
``structure`` analyzer character counts with Anthropic's documented heuristic
of approximately 3.5 English characters per token.  A second scenario applies
the documented approximate 30% token-count increase for the tokenizer
introduced with Claude Opus 4.7.  Neither scenario is an exact tokenizer count.

Examples:

    python scripts/claude_context_window_stats.py
    python scripts/claude_context_window_stats.py --scope all-query-matched
    python scripts/claude_context_window_stats.py \
        --output article/claude_context_window_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCOPES = ("exact-claude", "all-query-matched")
DEFAULT_CHARS_PER_TOKEN = 3.5
DEFAULT_TOKEN_UPLIFT = 0.30
DEFAULT_CONTEXT_WINDOWS = (200_000, 500_000, 1_000_000)
DEFAULT_LINE_GUIDELINE = 200
QUANTILES = (
    ("p25", 0.25),
    ("median", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)

ANTHROPIC_GLOSSARY_URL = (
    "https://platform.claude.com/docs/en/about-claude/glossary"
)
ANTHROPIC_TOKEN_COUNTING_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/token-counting"
)
ANTHROPIC_MODELS_URL = (
    "https://platform.claude.com/docs/en/about-claude/models/overview"
)
ANTHROPIC_CONTEXT_URL = (
    "https://support.claude.com/en/articles/"
    "8606394-how-large-is-the-context-window-on-paid-claude-plans"
)
ANTHROPIC_MEMORY_URL = "https://code.claude.com/docs/en/memory"
ANTHROPIC_HELP_GUIDANCE_URL = (
    "https://support.claude.com/en/articles/"
    "14553240-give-claude-context-claude-md-and-better-prompts"
)


@dataclass(frozen=True)
class FileMetrics:
    """Stored measurements required by the context-occupancy analysis."""

    file_id: int
    repo_full_name: str
    path: str
    size_bytes: int
    char_count: int
    line_count: int


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _scope_predicate(scope: str) -> str:
    if scope == "exact-claude":
        return "(lower(f.path) = 'claude.md' OR lower(f.path) LIKE '%/claude.md')"
    if scope == "all-query-matched":
        return "1 = 1"
    raise ValueError(f"unknown scope: {scope}")


def _scope_definition(scope: str) -> str:
    if scope == "exact-claude":
        return "case-insensitive exact basename CLAUDE.md"
    if scope == "all-query-matched":
        return "all stored files returned by the configured Claude Code query"
    raise ValueError(f"unknown scope: {scope}")


def load_file_metrics(
    db_path: str | Path,
    *,
    scope: str = "exact-claude",
) -> list[FileMetrics]:
    """Load stored byte, character, and line counts without modifying SQLite."""
    predicate = _scope_predicate(scope)
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = connection.execute(
            f"""
            SELECT f.id, f.repo_full_name, f.path, f.size_bytes, a.result_json
            FROM files AS f
            INNER JOIN analysis AS a ON a.file_id = f.id
            WHERE a.analyzer_id = 'structure'
              AND {predicate}
            ORDER BY f.id
            """
        )
        metrics: list[FileMetrics] = []
        for row in rows:
            result = json.loads(str(row["result_json"]))
            if row["size_bytes"] is None:
                raise ValueError(f"file {row['id']} has no stored size_bytes")
            try:
                size_bytes = int(row["size_bytes"])
                char_count = int(result["char_count"])
                line_count = int(result["line_count"])
            except KeyError as exc:
                raise ValueError(
                    f"file {row['id']} structure result lacks {exc.args[0]}"
                ) from exc
            if min(size_bytes, char_count, line_count) < 0:
                raise ValueError(
                    f"file {row['id']} has a negative stored structural count"
                )
            metrics.append(
                FileMetrics(
                    file_id=int(row["id"]),
                    repo_full_name=str(row["repo_full_name"]),
                    path=str(row["path"]),
                    size_bytes=size_bytes,
                    char_count=char_count,
                    line_count=line_count,
                )
            )
        return metrics
    finally:
        connection.close()


def nearest_rank(values: Sequence[int | float], probability: float) -> int | float:
    """Return an empirical nearest-rank quantile."""
    if not values:
        raise ValueError("cannot compute a quantile of an empty population")
    if not 0 < probability <= 1:
        raise ValueError("probability must be greater than zero and at most one")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def estimate_tokens(
    char_count: int,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    multiplier: float = 1.0,
) -> int:
    """Return a rounded-up character-based token estimate."""
    if char_count < 0:
        raise ValueError("char_count must be non-negative")
    if not math.isfinite(chars_per_token) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be greater than zero")
    return math.ceil((char_count / chars_per_token) * multiplier)


def _rounded(value: float) -> float:
    return round(value, 6)


def describe(values: Sequence[int | float]) -> dict[str, int | float]:
    """Describe a complete population with empirical nearest-rank quantiles."""
    if not values:
        raise ValueError("cannot describe an empty population")
    result: dict[str, int | float] = {
        "count": len(values),
        "minimum": min(values),
    }
    for label, probability in QUANTILES:
        result[label] = nearest_rank(values, probability)
    result["maximum"] = max(values)
    result["mean"] = _rounded(sum(values) / len(values))
    return result


def _describe_percentages(values: Sequence[float]) -> dict[str, int | float]:
    return {
        key: (_rounded(float(value)) if key != "count" else int(value))
        for key, value in describe(values).items()
    }


def _share(count: int, total: int) -> float:
    return _rounded(count / total) if total else 0.0


def build_summary(
    metrics: Sequence[FileMetrics],
    *,
    scope: str = "exact-claude",
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    token_uplift: float = DEFAULT_TOKEN_UPLIFT,
    context_windows: Sequence[int] = DEFAULT_CONTEXT_WINDOWS,
    line_guideline: int = DEFAULT_LINE_GUIDELINE,
) -> dict[str, object]:
    """Build a deterministic JSON-ready context-occupancy summary."""
    if not metrics:
        raise ValueError("no files were provided")
    _scope_definition(scope)
    if not math.isfinite(chars_per_token) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    if not math.isfinite(token_uplift) or token_uplift < 0:
        raise ValueError("token_uplift must be non-negative")
    if line_guideline <= 0:
        raise ValueError("line_guideline must be positive")
    windows = sorted(set(int(window) for window in context_windows))
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("context windows must contain positive integers")

    baseline_tokens = [
        estimate_tokens(item.char_count, chars_per_token=chars_per_token)
        for item in metrics
    ]
    uplift_multiplier = 1 + token_uplift
    uplift_tokens = [
        estimate_tokens(
            item.char_count,
            chars_per_token=chars_per_token,
            multiplier=uplift_multiplier,
        )
        for item in metrics
    ]

    below = sum(item.line_count < line_guideline for item in metrics)
    at = sum(item.line_count == line_guideline for item in metrics)
    above = sum(item.line_count > line_guideline for item in metrics)
    total = len(metrics)

    occupancy: dict[str, object] = {}
    for window in windows:
        baseline_percentages = [100 * tokens / window for tokens in baseline_tokens]
        uplift_percentages = [100 * tokens / window for tokens in uplift_tokens]
        occupancy[str(window)] = {
            "nominal_context_tokens": window,
            "baseline_percent": _describe_percentages(baseline_percentages),
            "uplift_sensitivity_percent": _describe_percentages(uplift_percentages),
            "baseline_files_at_or_above_nominal_window": sum(
                tokens >= window for tokens in baseline_tokens
            ),
            "uplift_files_at_or_above_nominal_window": sum(
                tokens >= window for tokens in uplift_tokens
            ),
            "baseline_headroom_after_largest_file_tokens": (
                window - max(baseline_tokens)
            ),
            "uplift_headroom_after_largest_file_tokens": (
                window - max(uplift_tokens)
            ),
        }

    return {
        "schema_version": 1,
        "scope": scope,
        "scope_definition": _scope_definition(scope),
        "files": total,
        "method": {
            "database_access": "SQLite mode=ro&immutable=1 with PRAGMA query_only=ON",
            "character_count": "Python len(content), stored by the structure analyzer",
            "line_count": "Python len(content.splitlines()), stored by the structure analyzer",
            "quantiles": "empirical nearest-rank over files",
            "token_estimate": (
                "ceil(character_count / chars_per_token * scenario_multiplier)"
            ),
            "chars_per_token": chars_per_token,
            "chars_per_token_source": ANTHROPIC_GLOSSARY_URL,
            "token_uplift_sensitivity": token_uplift,
            "token_uplift_multiplier": uplift_multiplier,
            "token_uplift_source": ANTHROPIC_TOKEN_COUNTING_URL,
            "exact_token_counting_source": ANTHROPIC_TOKEN_COUNTING_URL,
            "context_window_sources": [
                ANTHROPIC_MODELS_URL,
                ANTHROPIC_CONTEXT_URL,
            ],
            "line_guideline_source": ANTHROPIC_MEMORY_URL,
            "dated_line_guideline_source": ANTHROPIC_HELP_GUIDANCE_URL,
        },
        "distributions": {
            "size_bytes": describe([item.size_bytes for item in metrics]),
            "characters": describe([item.char_count for item in metrics]),
            "lines": describe([item.line_count for item in metrics]),
            "baseline_estimated_tokens": describe(baseline_tokens),
            "uplift_sensitivity_estimated_tokens": describe(uplift_tokens),
        },
        "context_occupancy": occupancy,
        "line_guideline": {
            "guideline": f"target under {line_guideline} lines per CLAUDE.md file",
            "threshold_lines": line_guideline,
            "files_below": below,
            "files_below_share": _share(below, total),
            "files_exactly_at": at,
            "files_exactly_at_share": _share(at, total),
            "files_above": above,
            "files_above_share": _share(above, total),
            "files_at_or_above": at + above,
            "files_at_or_above_share": _share(at + above, total),
            "files_above_500_lines": sum(item.line_count > 500 for item in metrics),
            "files_above_1000_lines": sum(item.line_count > 1000 for item in metrics),
        },
        "limitations": [
            "Character-based token values are estimates, not tokenizer outputs.",
            "The 30% uplift is a sensitivity scenario, not an upper bound.",
            (
                "Nominal context occupancy excludes system instructions, tool "
                "schemas, conversation history, other loaded files, and output "
                "headroom."
            ),
            (
                "The 200-line value is vendor authoring guidance, not a loading "
                "or performance threshold."
            ),
        ],
    }


def write_summary(summary: dict[str, object], output_path: str | Path) -> Path:
    """Write deterministic UTF-8 JSON and return its path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="exact-claude",
        help="file population (default: exact-claude)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="article/claude_context_window_summary.json",
        help="summary JSON path",
    )
    parser.add_argument(
        "--chars-per-token",
        type=positive_float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="character/token heuristic (default: 3.5)",
    )
    parser.add_argument(
        "--token-uplift",
        type=nonnegative_float,
        default=DEFAULT_TOKEN_UPLIFT,
        help="new-tokenizer sensitivity uplift as a fraction (default: 0.30)",
    )
    parser.add_argument(
        "--context-windows",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_WINDOWS),
        help="nominal token windows (default: 200000 500000 1000000)",
    )
    parser.add_argument(
        "--line-guideline",
        type=positive_int,
        default=DEFAULT_LINE_GUIDELINE,
        help="vendor line guideline threshold (default: 200)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        metrics = load_file_metrics(arguments.db, scope=arguments.scope)
        summary = build_summary(
            metrics,
            scope=arguments.scope,
            chars_per_token=arguments.chars_per_token,
            token_uplift=arguments.token_uplift,
            context_windows=arguments.context_windows,
            line_guideline=arguments.line_guideline,
        )
        output = write_summary(summary, arguments.output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    distributions = summary["distributions"]
    assert isinstance(distributions, dict)
    uplift = distributions["uplift_sensitivity_estimated_tokens"]
    assert isinstance(uplift, dict)
    guideline = summary["line_guideline"]
    assert isinstance(guideline, dict)
    print(f"Wrote context summary for {len(metrics):,} files to {output.resolve()}")
    print(
        "30% uplift sensitivity: "
        f"median={int(uplift['median']):,}, "
        f"p99={int(uplift['p99']):,}, "
        f"maximum={int(uplift['maximum']):,} estimated tokens"
    )
    print(
        f"At or above {arguments.line_guideline} lines: "
        f"{int(guideline['files_at_or_above']):,} "
        f"({100 * float(guideline['files_at_or_above_share']):.2f}%)"
    )
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
