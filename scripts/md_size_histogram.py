"""Create an image histogram of Markdown file sizes stored in ``mined.db``.

Examples:
    python scripts/md_size_histogram.py
    python scripts/md_size_histogram.py --db path/to/mined.db --output sizes.png
    python scripts/md_size_histogram.py --tool claude --bins 300 --x-intervals 20
    python scripts/md_size_histogram.py --max-size 50000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Sequence


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_markdown_sizes(db_path: str | Path, tool: str | None = None) -> list[int]:
    """Return non-negative ``size_bytes`` values for paths ending in ``.md``."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    clauses = [
        "size_bytes IS NOT NULL",
        "size_bytes >= 0",
        "lower(path) LIKE '%.md'",
    ]
    parameters: list[str] = []
    if tool:
        clauses.append("tool_id = ?")
        parameters.append(tool)

    query = "SELECT size_bytes FROM files WHERE " + " AND ".join(clauses)
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        cursor = connection.execute(query, parameters)
        try:
            return [int(row[0]) for row in cursor.fetchall()]
        finally:
            cursor.close()
    finally:
        connection.close()


def _format_bytes(value: float, _position: float | None = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.1f} KB"
    return f"{value:.0f} B"


def _x_tick_positions(maximum: int, intervals: int) -> list[float]:
    """Return evenly spaced ticks including zero and the exact maximum."""
    if intervals <= 0:
        raise ValueError("x intervals must be positive")
    if maximum <= 0:
        return [0.0]
    return [maximum * index / intervals for index in range(intervals + 1)]


def save_histogram(
    sizes: Sequence[int],
    output_path: str | Path,
    *,
    bins: int = 200,
    x_intervals: int = 16,
    dpi: int = 160,
) -> Path:
    """Render ``sizes`` as a histogram and return the written image path."""
    if not sizes:
        raise ValueError("no Markdown files with a size were found")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if x_intervals <= 0:
        raise ValueError("x intervals must be positive")
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

    maximum = max(sizes)
    plot_maximum = maximum if maximum > 0 else 1
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.hist(
        sizes,
        bins=bins,
        range=(0, plot_maximum),
        color="#3274a1",
        edgecolor="white",
        linewidth=0.25,
    )
    axis.set_title(f"Markdown file size distribution ({len(sizes):,} files)")
    axis.set_xlabel(".md file size (bytes)")
    axis.set_ylabel("Number of .md files")
    axis.set_xlim(0, plot_maximum)
    axis.set_xticks(_x_tick_positions(maximum, x_intervals))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_bytes))
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()

    try:
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query mined.db and save a histogram of Markdown file sizes as an image."
        )
    )
    parser.add_argument(
        "--db",
        default="mined.db",
        help="SQLite database created by miner (default: mined.db)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="md_size_histogram.png",
        help="output image path (default: md_size_histogram.png)",
    )
    parser.add_argument(
        "--tool",
        default=None,
        help="only include this tool_id (for example: claude; default: all tools)",
    )
    parser.add_argument(
        "--bins",
        type=positive_int,
        default=200,
        help="number of histogram bins (default: 200)",
    )
    parser.add_argument(
        "--x-intervals",
        type=positive_int,
        default=16,
        help="number of equal intervals marked on the x-axis (default: 16)",
    )
    parser.add_argument(
        "--max-size",
        type=positive_int,
        default=None,
        metavar="BYTES",
        help="exclude files larger than this many bytes",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=160,
        help="output resolution in dots per inch (default: 160)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        sizes = load_markdown_sizes(args.db, tool=args.tool)
        excluded = 0
        if args.max_size is not None:
            excluded = sum(size > args.max_size for size in sizes)
            sizes = [size for size in sizes if size <= args.max_size]
        output = save_histogram(
            sizes,
            args.output,
            bins=args.bins,
            x_intervals=args.x_intervals,
            dpi=args.dpi,
        )
    except (FileNotFoundError, sqlite3.Error, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    message = f"Wrote histogram for {len(sizes):,} .md files to {output.resolve()}"
    if excluded:
        message += f" ({excluded:,} files above --max-size excluded)"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
