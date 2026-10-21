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
| Local archive staging | 103 actual files, 149,143,537 bytes, all sizes/hashes verified |
| Archive plan | 1,020 records, including 917 missing planned project assets; no availability claim for missing files |
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

The expanded asset search confirmed 24 exact Top1000 files and one exact Top100
file. No exact curated full-expression profile, required final checkpoint or
formal prediction bytes were found. An older full-input candidate has date-formatted
gene symbols and a duplicate row; it was rejected without repair or substitution.
All 192 input identities are now recorded, including original Top3000 and prepared
Top2000 hashes. The first-2,000-row preparation contract was confirmed in retained
code; missing file bytes prevent a new prefix comparison.

The prospective archive includes verified ranked inputs, detailed evaluations,
metadata/folds/gene orders, identities and checksums. Third-party encoder weights,
dictionaries, software, Arial and raw reads remain external references only.
`THIRD_PARTY_RESOURCES.tsv` distinguishes known identity hashes and unresolved
version/terms information. Project-asset redistribution awaits author review.

Full training/prediction/reevaluation still requires original missing assets. A
complete raw-read-to-curated-matrix workflow and run-bound original software lockfile
are also unavailable. These limitations are not hidden by the successful figure/
registry audit. No manuscript or Zenodo DOI is invented. CITATION.cff is unchanged.
