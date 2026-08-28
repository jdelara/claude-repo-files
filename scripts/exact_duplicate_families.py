"""Export exact duplicate families from ``mined.db`` without modifying it.

Files are grouped by the SHA-256 ``content_hash`` already stored by the miner.
The default scope is every case-insensitive path ending in ``.md``. The script
opens SQLite with ``mode=ro&immutable=1`` and additionally enables
``PRAGMA query_only=ON``.

Example:

    python scripts/exact_duplicate_families.py \
        --db mined.db \
        --scope markdown \
        --families-output article/exact_duplicate_families.csv \
        --members-output article/exact_duplicate_family_members.csv \
        --summary-output article/exact_duplicate_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


SCOPES = ("markdown", "exact-claude", "all")


@dataclass(frozen=True)
class DuplicateMember:
    content_hash: str
    repo_full_name: str
    path: str
    size_bytes: int | None
    html_url: str | None
    fetched_at: str | None


@dataclass(frozen=True)
class DuplicateFamily:
    content_hash: str
    members: tuple[DuplicateMember, ...]

    @property
    def copies(self) -> int:
        return len(self.members)

    @property
    def repeated_instances(self) -> int:
        return self.copies - 1

    @property
    def repositories(self) -> int:
        return len({member.repo_full_name for member in self.members})

    @property
    def minimum_size_bytes(self) -> int | None:
        sizes = [
            member.size_bytes
            for member in self.members
            if member.size_bytes is not None
        ]
        return min(sizes, default=None)

    @property
    def maximum_size_bytes(self) -> int | None:
        sizes = [
            member.size_bytes
            for member in self.members
            if member.size_bytes is not None
        ]
        return max(sizes, default=None)

    @property
    def representative(self) -> DuplicateMember:
        return self.members[0]


@dataclass(frozen=True)
class DuplicateSummary:
    scope: str
    scope_definition: str
    scoped_files: int
    hashed_files: int
    unhashed_files: int
    distinct_content_hashes: int
    duplicate_families: int
    files_in_duplicate_families: int
    repeated_instances: int
    cross_repository_families: int
    single_repository_families: int
    largest_family_size: int
    largest_family_repositories: int
    largest_family_hash: str | None
    files_in_duplicate_family_share: float
    removable_repeated_instance_share: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _scope_predicate(scope: str, *, alias: str = "") -> str:
    if scope not in SCOPES:
        raise ValueError(f"unsupported scope: {scope!r}")
    prefix = f"{alias}." if alias else ""
    if scope == "markdown":
        return f"lower({prefix}path) LIKE '%.md'"
    if scope == "exact-claude":
        return (
            f"(lower({prefix}path) = 'claude.md' "
            f"OR lower({prefix}path) LIKE '%/claude.md')"
        )
    return "1 = 1"


def _scope_definition(scope: str) -> str:
    definitions = {
        "markdown": "case-insensitive paths ending in .md",
        "exact-claude": "case-insensitive basename equal to CLAUDE.md",
        "all": "all stored file records",
    }
    return definitions[scope]


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def analyze_exact_duplicates(
    db_path: str | Path,
    *,
    scope: str = "markdown",
) -> tuple[DuplicateSummary, list[DuplicateFamily]]:
    """Return deterministic exact-duplicate families for the selected scope."""
    predicate = _scope_predicate(scope)
    aliased_predicate = _scope_predicate(scope, alias="f")
    connection = _readonly_connection(db_path)
    try:
        counts = connection.execute(
            f"""
            SELECT count(*) AS scoped_files,
                   count(content_hash) AS hashed_files,
                   count(DISTINCT content_hash) AS distinct_content_hashes
            FROM files
            WHERE {predicate}
            """
        ).fetchone()
        assert counts is not None

        rows = connection.execute(
            f"""
            WITH duplicate_hashes AS (
                SELECT content_hash
                FROM files
                WHERE {predicate}
                  AND content_hash IS NOT NULL
                GROUP BY content_hash
                HAVING count(*) > 1
            )
            SELECT f.content_hash, f.repo_full_name, f.path, f.size_bytes,
                   f.html_url, f.fetched_at
            FROM files AS f
            INNER JOIN duplicate_hashes AS d
                    ON d.content_hash = f.content_hash
            WHERE {aliased_predicate}
            ORDER BY f.content_hash, lower(f.repo_full_name), lower(f.path),
                     f.repo_full_name, f.path
            """
        )

        grouped: dict[str, list[DuplicateMember]] = {}
        for row in rows:
            content_hash = str(row["content_hash"])
            grouped.setdefault(content_hash, []).append(
                DuplicateMember(
                    content_hash=content_hash,
                    repo_full_name=str(row["repo_full_name"]),
                    path=str(row["path"]),
                    size_bytes=(
                        int(row["size_bytes"])
                        if row["size_bytes"] is not None
                        else None
                    ),
                    html_url=row["html_url"],
                    fetched_at=row["fetched_at"],
                )
            )
    finally:
        connection.close()

    families = [
        DuplicateFamily(
            content_hash=content_hash,
            members=tuple(
                sorted(
                    members,
                    key=lambda member: (
                        member.repo_full_name.casefold(),
                        member.path.casefold(),
                        member.repo_full_name,
                        member.path,
                    ),
                )
            ),
        )
        for content_hash, members in grouped.items()
    ]
    families.sort(
        key=lambda family: (
            -family.copies,
            -family.repositories,
            family.content_hash,
        )
    )

    scoped_files = int(counts["scoped_files"])
    hashed_files = int(counts["hashed_files"])
    files_in_families = sum(family.copies for family in families)
    repeated_instances = sum(family.repeated_instances for family in families)
    largest = families[0] if families else None
    cross_repository = sum(family.repositories > 1 for family in families)
    summary = DuplicateSummary(
        scope=scope,
        scope_definition=_scope_definition(scope),
        scoped_files=scoped_files,
        hashed_files=hashed_files,
        unhashed_files=scoped_files - hashed_files,
        distinct_content_hashes=int(counts["distinct_content_hashes"]),
        duplicate_families=len(families),
        files_in_duplicate_families=files_in_families,
        repeated_instances=repeated_instances,
        cross_repository_families=cross_repository,
        single_repository_families=len(families) - cross_repository,
        largest_family_size=largest.copies if largest else 0,
        largest_family_repositories=largest.repositories if largest else 0,
        largest_family_hash=largest.content_hash if largest else None,
        files_in_duplicate_family_share=(
            files_in_families / scoped_files if scoped_files else 0.0
        ),
        removable_repeated_instance_share=(
            repeated_instances / scoped_files if scoped_files else 0.0
        ),
    )
    return summary, families


def _open_csv_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def write_family_csv(
    families: Sequence[DuplicateFamily],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    with _open_csv_output(output) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "family_rank",
                "content_hash",
                "copies",
                "repeated_instances",
                "repositories",
                "minimum_size_bytes",
                "maximum_size_bytes",
                "representative_repo",
                "representative_path",
            ),
        )
        writer.writeheader()
        for rank, family in enumerate(families, 1):
            writer.writerow(
                {
                    "family_rank": rank,
                    "content_hash": family.content_hash,
                    "copies": family.copies,
                    "repeated_instances": family.repeated_instances,
                    "repositories": family.repositories,
                    "minimum_size_bytes": family.minimum_size_bytes,
                    "maximum_size_bytes": family.maximum_size_bytes,
                    "representative_repo": family.representative.repo_full_name,
                    "representative_path": family.representative.path,
                }
            )
    return output


def write_member_csv(
    families: Sequence[DuplicateFamily],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    with _open_csv_output(output) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "family_rank",
                "content_hash",
                "family_copies",
                "family_repositories",
                "repo_full_name",
                "path",
                "size_bytes",
                "html_url",
                "fetched_at",
            ),
        )
        writer.writeheader()
        for rank, family in enumerate(families, 1):
            for member in family.members:
                writer.writerow(
                    {
                        "family_rank": rank,
                        "content_hash": family.content_hash,
                        "family_copies": family.copies,
                        "family_repositories": family.repositories,
                        "repo_full_name": member.repo_full_name,
                        "path": member.path,
                        "size_bytes": member.size_bytes,
                        "html_url": member.html_url,
                        "fetched_at": member.fetched_at,
                    }
                )
    return output


def write_summary_json(
    summary: DuplicateSummary,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read mined.db without modifying it and export exact duplicate "
            "content families."
        )
    )
    parser.add_argument(
        "--db",
        default="mined.db",
        help="SQLite database produced by the miner (default: mined.db)",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="markdown",
        help=(
            "files to analyze: markdown, exact-claude, or all "
            "(default: markdown)"
        ),
    )
    parser.add_argument(
        "--families-output",
        default="exact_duplicate_families.csv",
        help="one-row-per-family CSV output",
    )
    parser.add_argument(
        "--members-output",
        default="exact_duplicate_family_members.csv",
        help="one-row-per-family-member CSV output",
    )
    parser.add_argument(
        "--summary-output",
        default="exact_duplicate_summary.json",
        help="machine-readable JSON summary output",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=20,
        help="number of leading families to print (default: 20)",
    )
    return parser


def _print_report(
    summary: DuplicateSummary,
    families: Sequence[DuplicateFamily],
    *,
    top: int,
) -> None:
    print(f"Scope: {summary.scope} ({summary.scope_definition})")
    print(f"Scoped files: {summary.scoped_files:,}")
    print(f"Distinct content hashes: {summary.distinct_content_hashes:,}")
    print(f"Duplicate families: {summary.duplicate_families:,}")
    print(
        "Files in duplicate families: "
        f"{summary.files_in_duplicate_families:,} "
        f"({summary.files_in_duplicate_family_share:.2%})"
    )
    print(
        "Removable repeated instances: "
        f"{summary.repeated_instances:,} "
        f"({summary.removable_repeated_instance_share:.2%})"
    )
    print(f"Cross-repository families: {summary.cross_repository_families:,}")
    print("Top exact duplicate families:")
    for rank, family in enumerate(families[:top], 1):
        representative = family.representative
        print(
            f"  {rank:>3}. {family.copies:>5,} copies in "
            f"{family.repositories:>5,} repositories; "
            f"{representative.repo_full_name}/{representative.path}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, families = analyze_exact_duplicates(args.db, scope=args.scope)
        family_path = write_family_csv(families, args.families_output)
        member_path = write_member_csv(families, args.members_output)
        summary_path = write_summary_json(summary, args.summary_output)
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_report(summary, families, top=args.top)
    print(f"Families CSV: {family_path.resolve()}")
    print(f"Members CSV: {member_path.resolve()}")
    print(f"Summary JSON: {summary_path.resolve()}")
    print("Database writes: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
