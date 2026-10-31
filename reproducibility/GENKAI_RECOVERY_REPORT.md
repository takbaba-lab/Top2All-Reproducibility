# Genkai production asset recovery

Date: 2026-10-21. All 948 expected identities from the pre-search external-asset
inventory were located and SHA-256 verified on Genkai. Expected hashes were never
changed. Historical private GitHub repositories and their checkouts were not used.
This is the historical recovery record. Current availability, DOI and publication
scope are defined by the v1.0.0 release documentation; original identity/hash
evidence in this report is retained.

| Asset | Located | Expected SHA-256 matched | Locally staged |
|---|---:|---:|---:|
| Curated full-expression | 24/24 | 24/24 | 24/24 |
| Top100 | 24/24 | 24/24 | 24/24 |
| Top200 | 24/24 | 24/24 | 24/24 |
| Top300 | 24/24 | 24/24 | 24/24 |
| Top500 | 24/24 | 24/24 | 24/24 |
| Top1000 | 24/24 | 24/24 | 24/24 |
| Production Top2000 | 24/24 | 24/24 | 24/24 |
| Original Top3000 | 24/24 | 24/24 | 24/24 |
| Final project checkpoints | 174/174 | 174/174 | 0/174 |
| Formal predictions | 576/576 | 576/576 | 576/576 |
| Third-party encoder/dictionaries | 6/6 | 6/6 | Excluded |

## Search and acceptance

The specified Genkai Top2All project tree was recursively inventoried: 53,189
file paths, 241,871,867,869 bytes including unrelated/older runs and environment
files. An additional focused output-tree inventory covered 6,873 paths. Candidate
selection used filenames, directory patterns and sizes; input CSV headers and row
counts were inspected. SHA-256 was calculated only for 1,008 input candidates and
1,296 production/dependency candidates, not for the entire tree. All expected
identities were resolved inside the requested project tree, so no wider search
was required. The complete candidate decisions, sizes, hashes and private paths
are retained only in ignored author-local records. Hash-mismatched candidates
were rejected without repair, substitution or regeneration.

`GENKAI_ASSET_VERIFICATION.tsv` records each expected/observed hash, observed
filename, size and availability without disclosing private filesystem paths.
`EXTERNAL_ASSETS_REQUIRED.tsv` distinguishes local staging from remote-only
verified checkpoints and nonredistributed third-party dependencies. Original
input, checkpoint and prediction hashes are unchanged.

## Input and prediction checks

All full profiles have 24,410 unique genes matching the fixed release gene
universe. All ranked lists have exactly N unique genes in descending expression
order, with original Gene/Expression strings preserved. Production Top2000 files
exist for all 24 samples and match the first 2,000 Top3000 data rows in header,
Gene/Expression strings and order. Their original CRLF serialization differs from
the original Top3000 LF serialization: the raw byte prefix differs, while the
prefix after line-ending normalization agrees in all 24 cases. Both originals
match their own expected provenance hashes and were retained byte-for-byte.
No derivative was generated.

All 576 formal prediction files match the expected production hashes and contain
24,410 unique gene rows and the `raw_pred_wo_overwrite` column used for formal
evaluation. Post-overwrite columns are retained as part of the exact original
files, but are not substituted for the formal column. No metric was recomputed
or changed as part of this recovery.

## Data volume and transfer

The 942 unique project-origin expected assets total 24,584,923,419 bytes. The six
external encoder/dictionary assets total 80,895,517 bytes. Together the 948 exact
assets total 24,665,818,936 bytes; duplicate candidate copies are not counted.

New transfer to ignored local staging: 743 project files, 699,356,978 bytes
(143 input files first, then 24 Top2000 files and 576 predictions). Every transferred
file was SHA-256 checked again locally against the Genkai hash and retained
expected identity. The 25 previously staged ranked files were also reverified.
No project data or checkpoint bytes were added to GitHub.

The 174 checkpoints total 23,885,006,370 bytes. They remain on Genkai because the
preparation machine had approximately 16.6 GB available before prediction transfer.
For v1.0.0, these 174 verified checkpoints are explicitly excluded from the core
archive and may be distributed separately if needed.
The six third-party files were not copied or staged. Their original upstream
revision and redistribution terms remain unresolved where previously unrecorded;
matching bytes do not establish a redistribution right.

`ZENODO_MANIFEST.tsv` retains 1,020 records: 846 locally staged files totaling
848,696,315 bytes and 174 checkpoints not staged. The checkpoint rows retain empty
local paths and unknown local size, with separate verified Genkai status/size
columns. No expected project asset remains unlocated. Staging metadata and its
checksum list were refreshed; external resources remain outside the deposit.

## Remaining limits and verification

Checkpoint exclusion is a release-scope decision, not an unlocated-asset condition.
The exact raw-read-to-curated-matrix workflow and original run-bound complete
software lockfile remain unavailable.
This recovery did not rerun training, inference or scientific evaluation.
Scientific source tables, registry values, figures and analysis logic are unchanged.
The only script change is the user-approved archive count check: counts now follow
manifest statuses instead of the obsolete 103/917 snapshot. All other audit and
scientific checks retain their previous logic.

Release numerical/link/figure/fold validation and physical archive SHA-256 audit
passed. Final release checksums and private path/username/credential scans are
verified before commit. Training and inference were not rerun.
