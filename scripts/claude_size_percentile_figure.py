"""Create a publication-ready size profile for exact ``CLAUDE.md`` files.

The script reads stored structural measurements from ``mined.db`` through an
immutable, query-only SQLite connection. It plots empirical cumulative
distribution functions (ECDFs) for bytes, lines, words, and a transparent
character-based token estimate, with nearest-rank percentiles annotated. The
default output is a vector PDF sized for a two-column IEEE Software figure.

Example:

    python scripts/claude_size_percentile_figure.py \
        --db mined.db \
        --output output/pdf/claude_size_percentile_figure.pdf \
        --preview-output tmp/pdfs/claude_size_percentile_figure.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCOPES = ("exact-claude", "all-query-matched")
DEFAULT_CHARS_PER_TOKEN = 3.5
DEFAULT_TOKEN_UPLIFT = 0.30
DEFAULT_READING_WPM = 238.0
DEFAULT_LINE_GUIDELINE = 200
PERCENTILES = (
    ("median", 0.50),
    ("p90", 0.90),
    ("p99", 0.99),
)
PERCENTILE_STYLES = {
    "median": {"color": "#0072B2", "marker": "o"},
    "p90": {"color": "#E69F00", "marker": "^"},
    "p99": {"color": "#D55E00", "marker": "s"},
}


@dataclass(frozen=True)
class SizeObservation:
    """Stored measurements for one file."""

    file_id: int
    path: str
    size_bytes: int
    line_count: int
    word_count: int
    char_count: int


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


def load_observations(
    db_path: str | Path,
    *,
    scope: str = "exact-claude",
) -> list[SizeObservation]:
    """Load stored size metrics without changing SQLite."""
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
            SELECT f.id, f.path, f.size_bytes, a.result_json
            FROM files AS f
            INNER JOIN analysis AS a ON a.file_id = f.id
            WHERE a.analyzer_id = 'structure'
              AND {predicate}
            ORDER BY f.id
            """
        )
        observations: list[SizeObservation] = []
        for row in rows:
            result = json.loads(str(row["result_json"]))
            if row["size_bytes"] is None:
                raise ValueError(f"file {row['id']} has no stored size_bytes")
            try:
                values = {
                    "size_bytes": int(row["size_bytes"]),
                    "line_count": int(result["line_count"]),
                    "word_count": int(result["word_count"]),
                    "char_count": int(result["char_count"]),
                }
            except KeyError as exc:
                raise ValueError(
                    f"file {row['id']} structure result lacks {exc.args[0]}"
                ) from exc
            if min(values.values()) < 0:
                raise ValueError(f"file {row['id']} has a negative size metric")
            observations.append(
                SizeObservation(
                    file_id=int(row["id"]),
                    path=str(row["path"]),
                    **values,
                )
            )
        if not observations:
            raise ValueError("the selected population contains no files")
        return observations
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


def estimate_tokens(
    char_count: int,
    *,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    multiplier: float = 1.0,
) -> int:
    """Estimate tokens from characters with explicit assumptions."""
    if char_count < 0:
        raise ValueError("char_count must be non-negative")
    if not math.isfinite(chars_per_token) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be greater than zero")
    return math.ceil(char_count / chars_per_token * multiplier)


def percentile_profile(values: Sequence[int]) -> dict[str, int]:
    """Return the three labeled percentiles displayed by the figure."""
    profile = {
        label: nearest_rank(values, probability)
        for label, probability in PERCENTILES
    }
    if min(profile.values()) <= 0:
        raise ValueError("displayed log-scale percentiles must be positive")
    return profile


def empirical_cdf(values: Sequence[int]) -> tuple[list[int], list[float]]:
    """Return sorted observed values and their cumulative percentages."""
    if not values:
        raise ValueError("cannot compute an ECDF of an empty population")
    if min(values) < 0:
        raise ValueError("ECDF values must be non-negative")

    counts = Counter(int(value) for value in values)
    total = len(values)
    cumulative = 0
    x_values: list[int] = []
    percentages: list[float] = []
    for value in sorted(counts):
        cumulative += counts[value]
        x_values.append(value)
        percentages.append(100 * cumulative / total)
    return x_values, percentages


def _format_axis_value(value: float, _position: float | None = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:g}k"
    return f"{value:g}"


def _format_minutes(words: int, reading_wpm: float) -> str:
    minutes = words / reading_wpm
    if minutes < 1:
        return f"{minutes * 60:.0f} s"
    if minutes >= 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes:.1f} min"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_percentile_figure(
    observations: Sequence[SizeObservation],
    output_path: str | Path,
    *,
    preview_output: str | Path | None = None,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    token_uplift: float = DEFAULT_TOKEN_UPLIFT,
    reading_wpm: float = DEFAULT_READING_WPM,
    line_guideline: int = DEFAULT_LINE_GUIDELINE,
    preview_dpi: int = 220,
) -> Path:
    """Write a vector PDF percentile figure and an optional PNG preview."""
    if not observations:
        raise ValueError("no observations were provided")
    if not math.isfinite(chars_per_token) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero")
    if not math.isfinite(token_uplift) or token_uplift < 0:
        raise ValueError("token_uplift must be non-negative")
    if not math.isfinite(reading_wpm) or reading_wpm <= 0:
        raise ValueError("reading_wpm must be greater than zero")
    if line_guideline <= 0:
        raise ValueError("line_guideline must be positive")
    if preview_dpi <= 0:
        raise ValueError("preview_dpi must be positive")

    output = Path(output_path)
    if output.suffix.lower() != ".pdf":
        raise ValueError("the publication output must use a .pdf extension")
    preview = Path(preview_output) if preview_output is not None else None
    if preview is not None and preview.suffix.lower() != ".png":
        raise ValueError("the preview output must use a .png extension")

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required; install dependencies with "
            "'python -m pip install -r requirements.txt'"
        ) from exc

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    bytes_values = [item.size_bytes for item in observations]
    line_values = [item.line_count for item in observations]
    word_values = [item.word_count for item in observations]
    baseline_tokens = [
        estimate_tokens(item.char_count, chars_per_token=chars_per_token)
        for item in observations
    ]
    uplift_tokens = [
        estimate_tokens(
            item.char_count,
            chars_per_token=chars_per_token,
            multiplier=1 + token_uplift,
        )
        for item in observations
    ]

    profiles = {
        "bytes": percentile_profile(bytes_values),
        "lines": percentile_profile(line_values),
        "words": percentile_profile(word_values),
        "tokens": percentile_profile(baseline_tokens),
    }
    guideline_count = sum(item.line_count >= line_guideline for item in observations)
    guideline_share = 100 * guideline_count / len(observations)

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.55), sharey=True)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.17,
        hspace=0.68,
        wspace=0.18,
    )

    panel_specs = (
        (
            axes[0, 0],
            "(a) Stored size",
            "Bytes per file (log scale)",
            bytes_values,
            profiles["bytes"],
            lambda value: f"{value:,} B",
        ),
        (
            axes[0, 1],
            "(b) Physical lines",
            "Lines per file (log scale)",
            line_values,
            profiles["lines"],
            lambda value: f"{value:,} lines",
        ),
        (
            axes[1, 0],
            "(c) Words and reading-time proxy",
            "Words per file (log scale)",
            word_values,
            profiles["words"],
            lambda value: (
                f"{value:,} words ({_format_minutes(value, reading_wpm)})"
            ),
        ),
        (
            axes[1, 1],
            "(d) Estimated tokens",
            "Estimated tokens per file (log scale)",
            baseline_tokens,
            profiles["tokens"],
            lambda value: f"{value:,}",
        ),
    )

    ecdf_lines = {}
    for panel_index, (
        axis,
        title,
        xlabel,
        values,
        profile,
        value_formatter,
    ) in enumerate(panel_specs):
        axis.set_xscale("log")
        axis.set_ylim(0, 102.5)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}%")
        )
        axis.set_title(title, loc="left", fontweight="bold", pad=5)
        axis.set_xlabel(xlabel, labelpad=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#777777")
        axis.spines["left"].set_linewidth(0.7)
        axis.spines["bottom"].set_color("#777777")
        axis.spines["bottom"].set_linewidth(0.7)
        axis.tick_params(axis="both", colors="#333333", length=3)
        axis.tick_params(axis="y", labelleft=panel_index % 2 == 0)
        if panel_index % 2 == 0:
            axis.set_ylabel("Cumulative share of files", labelpad=4)
        else:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False)
        axis.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        axis.xaxis.set_major_formatter(FuncFormatter(_format_axis_value))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(
            axis="both",
            which="major",
            color="#D9D9D9",
            linewidth=0.6,
        )
        axis.set_axisbelow(True)

        ecdf_x, ecdf_y = empirical_cdf(values)
        positive_points = [
            (value, percentage)
            for value, percentage in zip(ecdf_x, ecdf_y)
            if value > 0
        ]
        if not positive_points:
            raise ValueError("a log-scale ECDF requires a positive observation")
        positive_x = [point[0] for point in positive_points]
        positive_y = [point[1] for point in positive_points]
        ecdf_line = axis.step(
            positive_x,
            positive_y,
            where="post",
            color="#0072B2",
            linewidth=1.6,
            zorder=3,
        )[0]
        ecdf_lines[axis] = ecdf_line
        axis.fill_between(
            positive_x,
            positive_y,
            step="post",
            color="#56B4E9",
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )
        axis.set_xlim(min(positive_x) / 1.25, max(positive_x) * 1.12)

        annotation_positions = {
            "median": ((5, 5), "left", "bottom"),
            "p90": ((-5, -14), "right", "top"),
            "p99": ((-5, -2), "right", "top"),
        }
        for label, probability in PERCENTILES:
            value = profile[label]
            style = PERCENTILE_STYLES[label]
            percentile_y = 100 * probability
            axis.vlines(
                value,
                0,
                percentile_y,
                color=style["color"],
                linestyle=(0, (2, 2)),
                linewidth=0.8,
                alpha=0.8,
                zorder=2,
            )
            axis.scatter(
                [value],
                [percentile_y],
                s=26,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            label_name = "Median" if label == "median" else label
            offset, horizontal_alignment, vertical_alignment = (
                annotation_positions[label]
            )
            axis.annotate(
                f"{label_name}: {value_formatter(value)}",
                xy=(value, percentile_y),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=6.5,
                color="#222222",
                linespacing=1.05,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
                zorder=6,
            )

    line_axis = axes[0, 1]
    line_upper = line_axis.get_xlim()[1]
    line_axis.axvspan(
        line_guideline,
        line_upper,
        color="#E69F00",
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )
    line_axis.axvline(
        line_guideline,
        color="#555555",
        linestyle=(0, (3, 2)),
        linewidth=1,
        zorder=1,
    )
    line_axis.annotate(
        f"Guidance: under {line_guideline}",
        xy=(line_guideline, 0),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    line_axis.text(
        0.99,
        0.04,
        f"{guideline_share:.2f}% at or above",
        transform=line_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#805500",
    )

    token_axis = axes[1, 1]
    uplift_x, uplift_y = empirical_cdf(uplift_tokens)
    uplift_line = token_axis.step(
        uplift_x,
        uplift_y,
        where="post",
        color="#666666",
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        zorder=4,
    )[0]
    token_axis.set_xlim(
        token_axis.get_xlim()[0],
        max(uplift_x) * 1.12,
    )
    token_axis.legend(
        [ecdf_lines[token_axis], uplift_line],
        ["Baseline", "+30% sensitivity"],
        loc="lower right",
        frameon=False,
        fontsize=6.4,
        handlelength=2.6,
        borderaxespad=0.4,
    )

    zero_word_count = sum(value == 0 for value in word_values)
    zero_word_note = ""
    if zero_word_count:
        zero_word_note = (
            f" {zero_word_count:,} zero-word files at x=0 are not shown."
        )
    #figure.text(
    #    0.5,
    #    0.022,
    #    (
    #        f"Exact CLAUDE.md files (n={len(observations):,}); ECDF = share "
    #        "at or below x; empirical nearest-rank percentiles.\n"
    #        f"Reading time: {reading_wpm:g} wpm; tokens: "
    #        f"{chars_per_token:g} chars/token.{zero_word_note}"
    #    ),
    #    ha="center",
    #    va="bottom",
    #    fontsize=6.3,
    #    color="#4A4A4A",
    #    linespacing=1.25,
    #)

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Distributional profile of CLAUDE.md file size",
        "Author": "",
        "Subject": "Bytes, lines, words, token estimates, and reading time",
        "Keywords": "CLAUDE.md, file size, percentiles",
        "Creator": "scripts/claude_size_percentile_figure.py",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    try:
        figure.savefig(
            output,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.025,
            metadata=metadata,
        )
        if preview is not None:
            preview.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                preview,
                format="png",
                dpi=preview_dpi,
                bbox_inches="tight",
                pad_inches=0.025,
            )
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
        help="file population (default: exact-claude)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/pdf/claude_size_percentile_figure.pdf",
        help="vector PDF output path",
    )
    parser.add_argument(
        "--preview-output",
        help="optional PNG preview path",
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
        help="token sensitivity uplift as a fraction (default: 0.30)",
    )
    parser.add_argument(
        "--reading-wpm",
        type=positive_float,
        default=DEFAULT_READING_WPM,
        help="reading-rate proxy (default: 238)",
    )
    parser.add_argument(
        "--line-guideline",
        type=positive_int,
        default=DEFAULT_LINE_GUIDELINE,
        help="line-guideline threshold (default: 200)",
    )
    parser.add_argument(
        "--preview-dpi",
        type=positive_int,
        default=220,
        help="PNG preview resolution (default: 220)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        observations = load_observations(arguments.db, scope=arguments.scope)
        output = save_percentile_figure(
            observations,
            arguments.output,
            preview_output=arguments.preview_output,
            chars_per_token=arguments.chars_per_token,
            token_uplift=arguments.token_uplift,
            reading_wpm=arguments.reading_wpm,
            line_guideline=arguments.line_guideline,
            preview_dpi=arguments.preview_dpi,
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

    print(
        f"Wrote vector size-percentile figure for {len(observations):,} files "
        f"to {output.resolve()}"
    )
    print(f"PDF SHA-256: {_sha256(output)}")
    if arguments.preview_output:
        print(f"Wrote PNG preview to {Path(arguments.preview_output).resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
