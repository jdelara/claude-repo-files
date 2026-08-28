# Replication database

Place the frozen database here with this exact name:

```text
data/mined.db
```

The database is not stored in Git because it is about 1.20 GB. 
It can be downloaded from [here](https://miso.es/claude-repo-files/mined.db).

- SHA-256: `29bca85a0b5a9d9cf953461b0c0e2c90f7b8cf6855623ceb6e06eb9be2cb8442`
- Observation interval: 14 July 2026 18:09:05 UTC to 23 July 2026 17:50:14 UTC
- SQLite library used for the reported analyses: 3.49.1

## Licensing and third-party material

The database structure, selection, and arrangement are licensed under the
Open Data Commons Attribution License 1.0. Original annotations and derived
research results created by the dataset authors are licensed under Creative
Commons Attribution 4.0 International.

The database also contains text copied from public GitHub repositories,
including `CLAUDE.md` files. That text remains the property of its original
authors and is governed by the license of its source repository. It is not
relicensed under ODC-By or CC BY by this replication package. A repository's
public visibility on GitHub should not be interpreted as a new license from
the dataset authors.

Read [DATA_LICENSE.md](../DATA_LICENSE.md) before redistributing or reusing the
database. The software license in [LICENSE](../LICENSE) applies to the Python
code, not to the collected repository text.

After downloading the file, run this command from the repository root:

```console
python scripts/verify_replication_data.py --db data/mined.db
```

The command checks the byte size, SHA-256 digest, schema, and principal row
counts. It opens the database read-only.
