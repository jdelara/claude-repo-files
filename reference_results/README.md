# Reference results

This directory contains compact summaries produced for the paper and the
exact-`CLAUDE.md` duplicate tables used in its reuse analysis. They provide
known values against which to compare a new run.

`paper_dataset_summary.json`, `repository_languages.csv`, and
`top_repositories.csv` reproduce the paper's dataset, size, primary-language,
and stored-star tables with one row per repository where appropriate.

Most large row-level outputs are not committed. In particular, the complete
Markdown-reference and import-occurrence tables together occupy hundreds of
megabytes. The commands in the main `README.md` regenerate them from
`data/mined.db`.

The file `exact_claude_duplicate_summary.json` is the reference for the
paper's exact-basename population. The analysis program also supports an
all-Markdown run; these are different populations and should not be compared
as if they were the same result.
