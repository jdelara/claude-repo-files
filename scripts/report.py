"""Analyse mined data — reads directly from SQLite, parses result_json,
and prints a summary report. No pandas needed.

Usage:
    python scripts/report.py                          # uses mined.db
    python scripts/report.py --db path/to/other.db
    python scripts/report.py --export report.json     # also dump parsed data
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def load_data(db_path: str) -> dict:
    """Load files + parsed analysis results into a plain dict structure."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")

    files = {}
    for row in conn.execute("""
        SELECT f.id, f.repo_full_name, f.path, f.tool_id, f.size_bytes,
               f.html_url, r.stars, r.language
        FROM files f
        LEFT JOIN repos r ON r.full_name = f.repo_full_name
    """):
        files[row["id"]] = dict(row)
        files[row["id"]]["analysis"] = {}

    for row in conn.execute("SELECT file_id, analyzer_id, result_json FROM analysis"):
        fid = row["file_id"]
        if fid in files:
            files[fid]["analysis"][row["analyzer_id"]] = json.loads(row["result_json"])

    conn.close()
    return files


def describe_counts(values: list[int]) -> dict[str, float | int]:
    """Describe integer counts using empirical nearest-rank quantiles."""
    if not values:
        raise ValueError("cannot describe an empty count sequence")
    ordered = sorted(values)
    count = len(ordered)

    def quantile(probability: float) -> int:
        index = max(0, min(count - 1, math.ceil(probability * count) - 1))
        return ordered[index]

    return {
        "count": count,
        "total": sum(ordered),
        "min": ordered[0],
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "mean": sum(ordered) / count,
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "zero": sum(value == 0 for value in ordered),
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def report(db_path: str, export_path: str | None = None):
    files = load_data(db_path)
    if not files:
        print("No data found in", db_path)
        return

    # ------------------------------------------------------------------
    # 1. Overview
    # ------------------------------------------------------------------
    print_section("OVERVIEW")
    print(f"  Total files: {len(files)}")
    unique_repos = {f["repo_full_name"] for f in files.values()}
    print(f"  Unique repos: {len(unique_repos)}")

    tool_counts = Counter(f["tool_id"] for f in files.values())
    print(f"\n  Files by tool:")
    for tool, count in tool_counts.most_common():
        print(f"    {tool:20s} {count:>5d}")

    # ------------------------------------------------------------------
    # 2. Repo languages
    # ------------------------------------------------------------------
    print_section("REPO LANGUAGES (top 15)")
    lang_counts = Counter(f["language"] for f in files.values() if f["language"])
    for lang, count in lang_counts.most_common(15):
        print(f"    {lang:20s} {count:>5d}")

    # ------------------------------------------------------------------
    # 3. Structure analysis
    # ------------------------------------------------------------------
    print_section("STRUCTURE STATS")
    word_counts = []
    line_counts = []
    header_counts = []
    code_block_counts = []
    all_fence_langs = Counter()

    for f in files.values():
        s = f["analysis"].get("structure", {})
        if not s:
            continue
        word_counts.append(s.get("word_count", 0))
        line_counts.append(s.get("line_count", 0))
        header_counts.append(s.get("header_count", 0))
        code_block_counts.append(s.get("code_block_count", 0))
        for lang, cnt in s.get("code_block_languages", {}).items():
            all_fence_langs[lang] += cnt

    if word_counts:
        word_stats = describe_counts(word_counts)
        line_stats = describe_counts(line_counts)
        print(
            "  Word count:  "
            f"min={word_stats['min']}, p25={word_stats['p25']}, "
            f"median={word_stats['median']}, mean={word_stats['mean']:.0f}, "
            f"p75={word_stats['p75']}, p90={word_stats['p90']}, "
            f"p95={word_stats['p95']}, p99={word_stats['p99']}, "
            f"max={word_stats['max']}"
        )
        print(
            "  Line count:  "
            f"min={line_stats['min']}, p25={line_stats['p25']}, "
            f"median={line_stats['median']}, mean={line_stats['mean']:.0f}, "
            f"p75={line_stats['p75']}, p90={line_stats['p90']}, "
            f"p95={line_stats['p95']}, p99={line_stats['p99']}, "
            f"max={line_stats['max']}"
        )
        files_with_code = sum(1 for c in code_block_counts if c > 0)
        print(f"  Files with code blocks: {files_with_code}/{len(code_block_counts)} "
              f"({100*files_with_code/len(code_block_counts):.1f}%)")
        print(f"\n  Code-fence languages (top 15):")
        for lang, count in all_fence_langs.most_common(15):
            print(f"    {lang:20s} {count:>5d} blocks")

    # ------------------------------------------------------------------
    # 4. Notation / diagram analysis
    # ------------------------------------------------------------------
    print_section("NOTATION / DIAGRAM DETECTION")
    notation_counts = Counter()
    subtype_counts = Counter()
    confidence_counts = Counter()
    files_with_any = 0
    files_total_notation = 0
    notation_by_tool = defaultdict(Counter)

    for f in files.values():
        n = f["analysis"].get("notation", {})
        if not n:
            continue
        files_total_notation += 1
        if n.get("any_notation_detected"):
            files_with_any += 1
        for nid, summary in n.get("notations_summary", {}).items():
            notation_counts[nid] += summary.get("count", 0)
            notation_by_tool[f["tool_id"]][nid] += summary.get("count", 0)
            for st in summary.get("subtypes", []):
                subtype_counts[f"{nid}:{st}"] += 1
            for conf in summary.get("confidences", []):
                confidence_counts[conf] += 1

    if files_total_notation:
        print(f"  Files with ANY notation detected: {files_with_any}/{files_total_notation} "
              f"({100*files_with_any/files_total_notation:.1f}%)")
        print(f"\n  Notation types found (by count of occurrences):")
        for nid, count in notation_counts.most_common():
            print(f"    {nid:20s} {count:>5d}")
        print(f"\n  Subtypes (files containing each subtype):")
        for st, count in subtype_counts.most_common():
            print(f"    {st:30s} {count:>5d}")
        print(f"\n  Confidence levels (file-notation pairs containing each level):")
        for conf, count in confidence_counts.most_common():
            print(f"    {conf:20s} {count:>5d}")
        print(f"\n  Notations by tool:")
        for tool_id in sorted(notation_by_tool):
            notations = notation_by_tool[tool_id]
            print(f"    {tool_id}:")
            for nid, count in notations.most_common():
                print(f"      {nid:20s} {count:>5d}")

    # ------------------------------------------------------------------
    # 5. Biggest / most notable files
    # ------------------------------------------------------------------
    print_section("NOTABLE FILES")
    sorted_by_size = sorted(files.values(), key=lambda f: f["size_bytes"] or 0, reverse=True)

    print("  Largest files:")
    for f in sorted_by_size[:10]:
        wc = f["analysis"].get("structure", {}).get("word_count", "?")
        print(f"    {f['size_bytes']:>8d} bytes  {wc:>6} words  {f['repo_full_name']}/{f['path']}")

    print("\n  Most popular repos (by stars):")
    sorted_by_stars = sorted(files.values(), key=lambda f: f["stars"] or 0, reverse=True)
    seen_repos = set()
    for f in sorted_by_stars:
        if f["repo_full_name"] in seen_repos:
            continue
        seen_repos.add(f["repo_full_name"])
        print(f"    {f['stars'] or 0:>8d} stars  {f['repo_full_name']}")
        if len(seen_repos) >= 15:
            break

    # Files with the most diagram/notation content
    files_by_notation_count = []
    for f in files.values():
        n = f["analysis"].get("notation", {})
        total = n.get("notation_count_total", 0)
        if total > 0:
            files_by_notation_count.append((total, f))
    files_by_notation_count.sort(key=lambda x: x[0], reverse=True)

    if files_by_notation_count:
        print("\n  Files with most embedded diagrams/DSLs:")
        for count, f in files_by_notation_count[:10]:
            notations = ", ".join(f["analysis"].get("notation", {}).get("notations_summary", {}).keys())
            print(f"    {count:>3d} diagrams  {f['repo_full_name']}/{f['path']}  ({notations})")

    # ------------------------------------------------------------------
    # 6. Export parsed data (optional)
    # ------------------------------------------------------------------
    if export_path:
        out = list(files.values())
        Path(export_path).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\n  Exported parsed data to {export_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyse mined agentic-tool files")
    p.add_argument("--db", default="mined.db", help="SQLite DB path")
    p.add_argument("--export", default=None, help="Also dump all parsed data to this JSON file")
    args = p.parse_args()
    report(args.db, args.export)
