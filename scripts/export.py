from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path


def export(db_path: str, out_path: str, fmt: str = "csv", include_content: bool = True):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    content_col = "f.content," if include_content else ""

    # One row per file, with each analyzer's result as a separate column
    file_rows = list(conn.execute(f"""
        SELECT f.id AS file_id, f.repo_full_name, f.path, f.tool_id, f.html_url,
               {content_col}
               f.size_bytes, f.fetched_at, r.stars, r.language
        FROM files f
        LEFT JOIN repos r ON r.full_name = f.repo_full_name
    """))

    # Collect all analyzer IDs present in the DB
    analyzer_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT analyzer_id FROM analysis ORDER BY analyzer_id"
    )]

    # Build a lookup: file_id -> {analyzer_id: result_json}
    analysis_by_file = {}
    for row in conn.execute("SELECT file_id, analyzer_id, result_json FROM analysis"):
        analysis_by_file.setdefault(row["file_id"], {})[row["analyzer_id"]] = row["result_json"]

    conn.close()

    # Merge into flat rows: one row per file, analysis results as separate columns
    merged = []
    for row in file_rows:
        d = dict(row)
        fid = d["file_id"]
        file_analysis = analysis_by_file.get(fid, {})
        for aid in analyzer_ids:
            d[f"analysis_{aid}"] = file_analysis.get(aid, "")
        merged.append(d)

    if fmt == "json":
        # For JSON, parse the analysis strings into proper objects
        for d in merged:
            for aid in analyzer_ids:
                key = f"analysis_{aid}"
                if d[key]:
                    d[key] = json.loads(d[key])
        Path(out_path).write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    elif fmt == "csv":
        if not merged:
            Path(out_path).write_text("", encoding="utf-8")
        else:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
                writer.writeheader()
                writer.writerows(merged)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    print(f"Wrote {len(merged)} rows ({len(analyzer_ids)} analyzer columns: {', '.join(analyzer_ids)}) to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Export mined data to CSV/JSON")
    p.add_argument("--db", default="mined.db")
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["csv", "json"], default="csv")
    p.add_argument("--no-content", action="store_true",
                   help="Exclude the raw file content column (smaller export)")
    args = p.parse_args()

    export(args.db, args.out, args.format, include_content=not args.no_content)
