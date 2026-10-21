# Local exact-asset search

Historical local-search snapshot, superseded for current availability by
[Genkai recovery](GENKAI_RECOVERY_REPORT.md). All 948 expected assets were later
located and hash-verified on Genkai; the counts below describe the earlier local search.

Date: 2026-10-21. The search covered the current paper workspace, local analysis
folders under Documents/Projects, Downloads, Desktop, local attachment/storage
paths and temporary analysis archives. Historical private GitHub repositories and
their local checkouts were excluded. No remote analysis assets were fetched and
no absent file was reconstructed as a substitute.

The inventory contained 60,443 distinct file paths. Twenty-one relevant archives
were inspected without running their contents (3,365 member headers); 198 loose
files/archive-member candidates were hashed. These counts describe the searched
scopes, not proof that no other copy exists on an unmounted disk, server or backup.
An author-only candidate/path inventory is retained outside tracked release files.

| Asset | Verified exact local files | Status |
|---|---:|---|
| Curated full-expression profiles | 0 / 24 | Final identity hashes retained; original bytes missing |
| Top1000 ranked lists | 24 / 24 | SHA-256 matches retained production identity |
| Top100 ranked lists | 1 / 24 | Exact Cas_DHT replicate 1 file matches retained input identity |
| Top200/300/500 lists | 0 / 72 | Required hashes retained; original bytes missing |
| Prepared Top2000 lists | 0 / 24 | Required hashes retained; original bytes missing |
| Original Top3000 lists | 0 / 24 | Required hashes retained; original bytes missing |
| Final project checkpoints | 0 / 174 | No matching final bytes found |
| Formal predictions | 0 / 576 | Older candidates found but no required hash matched |
| External encoder/dictionaries | 0 / 6 | Dependencies only; not redistributed |

`INPUT_ASSET_IDENTITY.tsv` records all 192 exact input identities. The accepted
25 inputs are staged byte-for-byte, preserving original rank order and paired
Gene/Expression values. All 24 Top1000 lists already appear compactly in GitHub
as Figure1 input sources. The additional original Top100 file is staged only for
the prospective Zenodo package. It also matches the first 100 ranked rows of the
corresponding verified Top1000 list; that observation is not used to fabricate
other missing nominal input files.

## Rejected full-expression candidate

One local Cas_DHT replicate 1 full-expression CSV has 24,411 rows and 24,410 unique
strings, including a duplicated date-formatted gene identifier. Thirteen expected
March/Sept gene symbols have been converted into date strings; its gene set is not
the final curated 24,410-gene universe. Its SHA-256 is
`46b7f42fa2c0a9cb9e781046d8ea4fd1b7ea0e52c385532f3a88ccfaf07bd577`,
which differs from the required file hash in `INPUT_ASSET_IDENTITY.tsv`. It was
not repaired, adopted or staged. Older prediction candidates likewise do not match
the final production prediction hashes and were not substituted.

## Top2000 preparation contract

The retained final preparation code uses the existing original Top3000 CSV, reads
its header and first 2,000 data rows in order, and writes those rows as Top2000.
It checks row count, uniqueness, descending expression and agreement with the
corresponding full profile. The code contract agrees with the manuscript package's
existing description. Missing Top3000/Top2000 files prevent a new byte-level prefix
comparison; that limit is explicit. No alternative sorting/tie handling was added.

## Archive status

The proposed manifest has 1,020 records: 103 existing staged files (149,143,537 bytes)
and 917 missing planned project assets. It distinguishes physical availability,
recorded hash identity and redistribution review. Third-party resources are tracked
separately and excluded from deposit staging. The package remains incomplete for
model training/prediction and full reevaluation until missing files are supplied.
No archive has been uploaded and no DOI or release has been created.
