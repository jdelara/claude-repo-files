# Fixed analysis inputs

These small files are committed because they cannot be recreated solely from
the SQLite database.

| File | Purpose |
|---|---|
| `github_star_comparison_raw.json` | Frozen GitHub responses used to reproduce the July 2026 repository-popularity comparison without network access. |
| `claude_repeated_content_audit_labels.csv` | Working labels used while developing the repeated-content classifier. They are not an independent human validation set. |
| `language_manual_audit_reviewed.csv` | Completed human review of the prose-language sample. |
| `github_claude_size_counts.json` | Cached GitHub file-count queries underlying the dataset-coverage figures in the main README. |

Do not replace these inputs when reproducing the paper. New GitHub counts or a
new manual review answer a different, later question.

Their SHA-256 digests are recorded in `SHA256SUMS`.
