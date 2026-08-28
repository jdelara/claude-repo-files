"""Plot estimated reading time for stored ``CLAUDE.md`` files.

The script reads stored ``structure`` analyzer word counts from SQLite without
modifying the database and writes a complementary cumulative distribution
(survival) curve.  Reading time is a linear word-count conversion, not an
estimate of comprehension or task-completion time.

Examples:

    python scripts/claude_reading_time_distribution.py
    python scripts/claude_reading_time_distribution.py --scope all-query-matched
    python scripts/claude_reading_time_distribution.py \
        --slow-wpm 175 --baseline-wpm 238 --fast-wpm 300 \
        --output article/claude_reading_time_distribution.png
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


SCOPES = ("exact-claude", "all-query-matched")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_word_counts(
    db_path: str | Path,
    *,
    scope: str = "exact-claude",
) -> list[int]:
    """Return stored non-negative structure word counts for ``scope``."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope: {scope}")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    clauses = ["a.analyzer_id = 'structure'"]
    if scope == "exact-claude":
        clauses.append(
            "(lower(f.path) = 'claude.md' "
            "OR lower(f.path) LIKE '%/claude.md')"
        )
    query = f"""
        SELECT a.result_json
        FROM files AS f
        INNER JOIN analysis AS a ON a.file_id = f.id
        WHERE {' AND '.join(clauses)}
        ORDER BY f.id
    """
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only=ON")
    try:
        counts: list[int] = []
        for (result_json,) in connection.execute(query):
            result = json.loads(result_json)
            count = int(result["word_count"])
            if count < 0:
                raise ValueError("stored word_count must be non-negative")
            counts.append(count)
        return counts
    finally:
        connection.close()


def nearest_rank(values: Sequence[int], probability: float) -> int:
    """Return an empirical nearest-rank quantile."""
    if not values:
        raise ValueError("cannot compute a quantile of an empty population")
    if not 0 < probability <= 1:
        raise ValueError("probability must be greater than zero and at most one")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return int(ordered[rank - 1])


def reading_time_survival(
    word_counts: Sequence[int],
    *,
    words_per_minute: float,
) -> tuple[list[float], list[float]]:
    """Return unique positive reading times and percentages at or above each."""
    if not word_counts:
        raise ValueError("no word counts were provided")
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be greater than zero")
    if any(count < 0 for count in word_counts):
        raise ValueError("word counts must be non-negative")

    frequencies = Counter(count for count in word_counts if count > 0)
    if not frequencies:
        raise ValueError("at least one positive word count is required")
    total = len(word_counts)
    remaining = sum(frequencies.values())
    minutes: list[float] = []
    percentages: list[float] = []
    for count in sorted(frequencies):
        minutes.append(count / words_per_minute)
        percentages.append(100 * remaining / total)
        remaining -= frequencies[count]
    return minutes, percentages


def _format_minutes(value: float, _position: float | None = None) -> str:
    if value < 1:
        seconds = value * 60
        return f"{seconds:g} s"
    if value >= 60:
        return f"{value / 60:g} h"
    return f"{value:g} min"


def _format_annotation_time(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    if minutes >= 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes:.1f} min"


def save_reading_time_distribution(
    word_counts: Sequence[int],
    output_path: str | Path,
    *,
    slow_wpm: float = 175,
    baseline_wpm: float = 238,
    fast_wpm: float = 300,
    dpi: int = 180,
    scope_label: str = "exact CLAUDE.md files",
) -> Path:
    """Write a reading-time survival plot and return its path."""
    if not word_counts:
        raise ValueError("no word counts were provided")
    if not 0 < slow_wpm < baseline_wpm < fast_wpm:
        raise ValueError("expected slow_wpm < baseline_wpm < fast_wpm")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install dependencies with "
            "'python -m pip install -r requirements.txt'"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    series = [
        (fast_wpm, "300 wpm (faster bound)", "#2a6fbb", (0, (1.5, 2.2)), 1.7),
        (slow_wpm, "175 wpm (slower bound)", "#d97706", (0, (5, 3)), 1.7),
        (baseline_wpm, "238 wpm baseline", "#6f42c1", "-", 2.6),
    ]

    figure, axis = plt.subplots(figsize=(12.5, 7.2))
    for rate, label, color, linestyle, linewidth in series:
        minutes, percentages = reading_time_survival(
            word_counts,
            words_per_minute=rate,
        )
        axis.step(
            minutes,
            percentages,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
        )

    positive_counts = [count for count in word_counts if count > 0]
    minimum = min(positive_counts) / fast_wpm
    maximum = max(positive_counts) / slow_wpm
    axis.set_xscale("log")
    axis.set_xlim(minimum * 0.8, maximum * 1.12)
    axis.set_ylim(0, 100.5)
    axis.set_xlabel("Estimated linear reading time (log scale)")
    axis.set_ylabel("Files requiring at least this much time")
    axis.set_title(f"Estimated reading time for {scope_label}", loc="left", pad=22)
    axis.text(
        0,
        1.01,
        (
            f"{len(word_counts):,} files; 238 wpm baseline and "
            "175–300 wpm English non-fiction sensitivity range"
        ),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    tick_candidates = (
        1 / 60,
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        30,
        60,
        120,
        240,
    )
    ticks = [tick for tick in tick_candidates if minimum <= tick <= maximum * 1.12]
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(FuncFormatter(_format_minutes))
    axis.set_yticks((1, 5, 10, 25, 50, 75, 100))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}%"))
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.8, alpha=0.65)
    axis.grid(axis="x", which="major", color="#e0e0e0", linewidth=0.7, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    annotations = (
        (0.50, "Median", (7, 9)),
        (0.90, "p90", (7, 10)),
        (0.95, "p95", (7, 10)),
        (0.99, "p99", (7, 10)),
    )
    for probability, label, offset in annotations:
        words = nearest_rank(word_counts, probability)
        minutes = words / baseline_wpm
        exceedance = 100 * (1 - probability)
        axis.scatter(
            [minutes],
            [exceedance],
            s=30,
            color="#6f42c1",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axis.annotate(
            f"{label}: {_format_annotation_time(minutes)}",
            xy=(minutes, exceedance),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    maximum_minutes = max(word_counts) / baseline_wpm
    axis.axvline(
        maximum_minutes,
        color="#777777",
        linewidth=1,
        linestyle=(0, (2, 3)),
        alpha=0.8,
    )
    axis.annotate(
        f"Maximum at baseline\n{maximum_minutes:.0f} min ({maximum_minutes / 60:.1f} h)",
        xy=(maximum_minutes, 0.18),
        xytext=(-8, 34),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#444444",
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.8},
    )

    axis.legend(loc="upper right", frameon=False, fontsize=9.5)
    zero_files = sum(count == 0 for count in word_counts)
    figure.text(
        0.99,
        0.012,
        (
            "Word counts include code and Markdown; not comprehension time. "
            f"{zero_files} zero-word files are omitted from the log axis. "
            "Reading-rate benchmark: Brysbaert (2019)."
        ),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    try:
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="mined.db", help="SQLite database")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="exact-claude",
        help="file population to plot (default: exact-claude)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="article/claude_reading_time_distribution.png",
        help="output image path",
    )
    parser.add_argument(
        "--slow-wpm",
        type=positive_float,
        default=175,
        help="slower sensitivity rate (default: 175)",
    )
    parser.add_argument(
        "--baseline-wpm",
        type=positive_float,
        default=238,
        help="central reading rate (default: 238)",
    )
    parser.add_argument(
        "--fast-wpm",
        type=positive_float,
        default=300,
        help="faster sensitivity rate (default: 300)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=180,
        help="output resolution (default: 180)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        word_counts = load_word_counts(arguments.db, scope=arguments.scope)
        label = (
            "exact CLAUDE.md files"
            if arguments.scope == "exact-claude"
            else "all query-matched files"
        )
        output = save_reading_time_distribution(
            word_counts,
            arguments.output,
            slow_wpm=arguments.slow_wpm,
            baseline_wpm=arguments.baseline_wpm,
            fast_wpm=arguments.fast_wpm,
            dpi=arguments.dpi,
            scope_label=label,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    median = nearest_rank(word_counts, 0.50) / arguments.baseline_wpm
    p90 = nearest_rank(word_counts, 0.90) / arguments.baseline_wpm
    maximum = max(word_counts) / arguments.baseline_wpm
    print(
        f"Wrote reading-time distribution for {len(word_counts):,} files to "
        f"{output.resolve()}"
    )
    print(
        f"Baseline {arguments.baseline_wpm:g} wpm: median={median:.2f} min, "
        f"p90={p90:.2f} min, maximum={maximum:.2f} min"
    )
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
