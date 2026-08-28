"""Plot the size distribution of CLAUDE.md files stored in ``mined.db``.

The database is opened in SQLite read-only, immutable mode. The script never
creates, updates, or deletes database records.

Examples:
    python scripts/mined_claude_size_histogram.py
    python scripts/mined_claude_size_histogram.py --db path/to/mined.db
    python scripts/mined_claude_size_histogram.py --bins 200 --log-y
    python scripts/mined_claude_size_histogram.py --max-size 50000
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


def load_claude_sizes(db_path: str | Path) -> list[int]:
    """Return valid sizes for files whose case-insensitive basename is CLAUDE.md."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")

    query = """
        SELECT size_bytes
        FROM files
        WHERE size_bytes IS NOT NULL
          AND size_bytes >= 0
          AND (
              lower(path) = 'claude.md'
              OR lower(path) LIKE '%/claude.md'
          )
    """

    # mode=ro rejects writes; immutable=1 also prevents SQLite from creating
    # journal or shared-memory sidecar files beside the database.
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        return [int(row[0]) for row in connection.execute(query)]
    finally:
        connection.close()


def _format_bytes(value: float, _position: float | None = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g} MB"
    if value >= 1_000:
        return f"{value / 1_000:g} KB"
    return f"{value:g} B"


def save_histogram(
    sizes: Sequence[int],
    output_path: str | Path,
    *,
    bins: int = 100,
    dpi: int = 160,
    log_y: bool = False,
) -> Path:
    """Save a PNG histogram with file size on x and file count on y."""
    if not sizes:
        raise ValueError("no CLAUDE.md files with a valid size were found")
    if bins <= 0:
        raise ValueError("bins must be positive")
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

    figure, axis = plt.subplots(figsize=(13, 7))
    axis.hist(
        sizes,
        bins=bins,
        color="#6f42c1",
        edgecolor="white",
        linewidth=0.3,
        log=log_y,
    )
    axis.set_title(f"CLAUDE.md file-size distribution ({len(sizes):,} files)")
    axis.set_xlabel("CLAUDE.md file size")
    axis.set_ylabel("Number of files" + (" (log scale)" if log_y else ""))
    axis.set_xlim(left=0)
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
            "Read mined.db without modifying it and save a histogram of "
            "CLAUDE.md file sizes."
        )
    )
    parser.add_argument(
        "--db",
        default="mined.db",
        help="SQLite database produced by the miner (default: mined.db)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="claude_size_histogram.png",
        help="output image path (default: claude_size_histogram.png)",
    )
    parser.add_argument(
        "--bins",
        type=positive_int,
        default=100,
        help="number of size buckets (default: 100)",
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
        help="output image resolution (default: 160)",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="use a logarithmic y-axis so low-frequency size buckets remain visible",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        sizes = load_claude_sizes(args.db)
        excluded = 0
        if args.max_size is not None:
            excluded = sum(size > args.max_size for size in sizes)
            sizes = [size for size in sizes if size <= args.max_size]
        output = save_histogram(
            sizes,
            args.output,
            bins=args.bins,
            dpi=args.dpi,
            log_y=args.log_y,
        )
    except (FileNotFoundError, sqlite3.Error, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    message = (
        f"Wrote histogram for {len(sizes):,} CLAUDE.md files "
        f"to {output.resolve()}"
    )
    if excluded:
        message += f" ({excluded:,} files above --max-size excluded)"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
