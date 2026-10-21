# Prepublication correction and reaudit

Date: 2026-10-21. Repository must remain private pending Takayoshi Yamashita's
confirmation. No visibility change, release, Zenodo upload or DOI issuance is
authorized. Historical private GitHub repositories/checkouts were not used.

| Check | Result |
|---|---|
| Display terminology | Figure1 Condition shift; Figure3 condition fingerprint; Figure4 measured expression/residual SD; FigureS1 measured SD quartiles |
| Metadata titles | Figures 1–5/S1–S3 use manuscript-facing titles; internal working-version descriptions removed |
| Figure regeneration | Eight figures, all PNG/PDF/SVG regenerated |
| Nontext geometry | All eight figures retain the exact pre-edit SVG drawing primitives/marker/path coordinates |
| Figure text audit | 381 manuscript text nodes checked; no inappropriate measured/condition wording or internal release history |
| Source tables | All 91 tracked source_data files remain byte-identical |
| Registry scientific fields | All 45,543 rows unchanged; only figure/script hash fields rebound |
| Source reductions / plot linkage | 45,543 exact reductions and 42,157 plotted numerical cell links verified |
| Regenerated numerical tables | 30 plot tables / 85,664 numeric cells agree with unchanged sources |
| Gene order, metadata and folds | Unchanged; three OOF/eight LOCO folds, no train/test overlap |
| Model and scientific evaluation code | Unchanged; original seeds, preprocessing, metric formulas and model specifications preserved |
| Release SHA-256 | Every tracked release file except the checksum file itself is covered and verified |
| Local archive staging | 846 actual files, 848,696,315 bytes, all sizes/hashes verified |
| Archive plan | 1,020 records; all project originals located; 174 checkpoints not staged |
| Private paths/usernames/credentials | Text, SVG and image/PDF metadata scanned; no detected private path, machine username or credential signature |

`figure_display_audit.json` preserves before/after hashes of nontext SVG geometry;
`audit_figure_display.py` recomputes the current fingerprint and checks displayed
wording. Source schema identifiers remain compatible. Technical true-variable and
upstream software Creator-version metadata are allowed; experiment groups use
condition terminology. Internal dictionary filenames are external-resource
identifiers, not alternative paper releases.

`validate_release.py` includes numerical source/link checks, source/figure/script
hashes, fold isolation, figure display geometry/terminology and archive inventory
validation. `audit_archive_assets.py --require-local-staging` additionally verifies
the actual ignored local archive bytes and archive checksum coverage. Neither
validator changes scientific outputs or runs model training.

The subsequent Genkai recovery located all 948 expected assets and verified each
against its unchanged expected SHA-256. All 192 inputs and 576 formal predictions
were locally reverified. All 24 production Top2000 files match Top3000 prefixes
in paired values/order; their original CRLF versus LF line endings differ.
The 174 checkpoints (23,885,006,370 bytes) remain verified on Genkai pending storage.
Six third-party files remain external and were not staged. See
`GENKAI_RECOVERY_REPORT.md` and `GENKAI_ASSET_VERIFICATION.tsv`.

Only the archive audit's obsolete fixed 103/917 count assertion was changed with
explicit user authorization; counts now derive from the manifest. Registry values,
source tables, figures and scientific analysis logic remain byte-identical to the
pre-recovery baseline. Original input/checkpoint/prediction identities are unchanged.
Full training/inference was not rerun. Checkpoint transfer, author review, original
software lockfile and raw-read-to-curated-matrix workflow remain unresolved.
No release, archive upload or DOI issuance was performed.
