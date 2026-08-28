"""Check that a local SQLite file is the frozen paper dataset."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Sequence


EXPECTED_SIZE_BYTES = 1_198_850_048
EXPECTED_SHA256 = "29bca85a0b5a9d9cf953461b0c0e2c90f7b8cf6855623ceb6e06eb9be2cb8442"
EXPECTED_COUNTS = {
    "repositories": 96_235,
    "files": 115_779,
    "analysis rows": 231_558,
    "case-insensitive exact CLAUDE.md files": 108_764,
    "repositories with a case-insensitive exact CLAUDE.md": 92_238,
    "non-exact query-matched files": 7_015,
    "repositories represented by non-exact query matches": 4_659,
    "strict-case exact CLAUDE.md files": 103_537,
    "repositories with a strict-case exact CLAUDE.md": 88_189,
}


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=30,
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def database_counts(path: Path) -> dict[str, int]:
    exact_predicate = (
        "lower(path) = 'claude.md' "
        "OR lower(path) LIKE '%/claude.md'"
    )
    strict_predicate = (
        "path = 'CLAUDE.md' "
        "OR substr(path, -10) = '/CLAUDE.md'"
    )
    connection = open_read_only(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = {"repos", "files", "analysis"} - tables
        if missing:
            raise ValueError("missing required table(s): " + ", ".join(sorted(missing)))

        counts = {
            "repositories": connection.execute("SELECT COUNT(*) FROM repos").fetchone()[0],
            "files": connection.execute("SELECT COUNT(*) FROM files").fetchone()[0],
            "analysis rows": connection.execute(
                "SELECT COUNT(*) FROM analysis"
            ).fetchone()[0],
        }
        counts["case-insensitive exact CLAUDE.md files"] = connection.execute(
            f"SELECT COUNT(*) FROM files WHERE {exact_predicate}"
        ).fetchone()[0]
        counts[
            "repositories with a case-insensitive exact CLAUDE.md"
        ] = connection.execute(
            "SELECT COUNT(DISTINCT repo_full_name) FROM files "
            f"WHERE {exact_predicate}"
        ).fetchone()[0]
        counts["non-exact query-matched files"] = connection.execute(
            f"SELECT COUNT(*) FROM files WHERE NOT ({exact_predicate})"
        ).fetchone()[0]
        counts[
            "repositories represented by non-exact query matches"
        ] = connection.execute(
            "SELECT COUNT(DISTINCT repo_full_name) FROM files "
            f"WHERE NOT ({exact_predicate})"
        ).fetchone()[0]

        counts["strict-case exact CLAUDE.md files"] = connection.execute(
            f"SELECT COUNT(*) FROM files WHERE {strict_predicate}"
        ).fetchone()[0]
        counts["repositories with a strict-case exact CLAUDE.md"] = connection.execute(
            "SELECT COUNT(DISTINCT repo_full_name) FROM files "
            f"WHERE {strict_predicate}"
        ).fetchone()[0]
    finally:
        connection.close()
    return {key: int(value) for key, value in counts.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the frozen SQLite database used by the paper."
    )
    parser.add_argument("--db", default="data/mined.db", help="path to mined.db")
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip the full-file SHA-256 pass (row counts are still checked)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.db)
    if not path.is_file():
        print(f"ERROR: database not found: {path}", file=sys.stderr)
        return 2

    failures: list[str] = []
    size = path.stat().st_size
    print(f"Database: {path.resolve()}")
    print(f"Size: {size:,} bytes")
    if size != EXPECTED_SIZE_BYTES:
        failures.append(
            f"size is {size:,}; expected {EXPECTED_SIZE_BYTES:,} bytes"
        )

    if not args.skip_hash:
        digest = file_sha256(path)
        print(f"SHA-256: {digest}")
        if digest != EXPECTED_SHA256:
            failures.append(f"SHA-256 is {digest}; expected {EXPECTED_SHA256}")
    else:
        print("SHA-256: skipped")

    try:
        observed = database_counts(path)
    except (sqlite3.Error, ValueError) as exc:
        failures.append(f"database inspection failed: {exc}")
        observed = {}

    for label, expected in EXPECTED_COUNTS.items():
        actual = observed.get(label)
        status = "OK" if actual == expected else "MISMATCH"
        shown = "unavailable" if actual is None else f"{actual:,}"
        print(f"{status:8} {label}: {shown} (expected {expected:,})")
        if actual is not None and actual != expected:
            failures.append(f"{label} is {actual:,}; expected {expected:,}")

    if failures:
        print("\nVerification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nVerification passed: this is the frozen paper dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
