# Paper-release audit

Date: 2026-10-21. Scope: a fresh release assembled from current local manuscript-facing
materials. No historical GitHub repository was used as a source or modified.

| Check | Result |
|---|---|
| Original final local validator | PASS; 27 read-only gates, all 45,543 source reductions |
| Release registry | PASS; 45,543 unique result IDs and exact numerical reductions |
| Figure/source links | PASS; 42,157 plotted cells resolve through release paths |
| Figure set | PASS; only Figures 1–5 and S1–S3, 24 PNG/PDF/SVG exports |
| Final export identity | PASS; committed exports match selected final local bytes |
| Supplement numbering | S1 target variability; S2 seed robustness; S3 input depth |
| Metadata/folds | PASS; 24 samples, eight conditions, three OOF/eight LOCO folds; no train/test overlap |
| Scientific code preservation | PASS; 18 engine/evaluation programs retain computation structure, operators and numeric constants; shared preprocessing function is exact |
| Numerical table preservation | PASS; 1,118,521 numeric cells checked before redundant unannotated intermediates were omitted |
| Plot execution | PASS; eight figures execute from release sources |
| Regenerated plot values | PASS; 30 tables and 85,664 numeric cells agree |
| Visual reproduction | Figures 2–5/S1–S3 pixel-exact; Figure1 has a minor schematic layout discrepancy, documented in ENVIRONMENT.md |
| Paths/secrets | PASS; text, SVG, PDF text/metadata and PNG metadata contain no detected private paths, machine usernames or credential signatures |
| Release checksums | All release files except the checksum file itself are listed in checksums.sha256; validator verifies exact coverage |
| Original scientific materials | Original selected figures and numerical source package remain unchanged |
| Raw RNA-seq | Referenced at DDBJ DRA015952; not redistributed |
| Git history | Fresh repository; no imported commits or private archive history |

The code-structure comparison excludes docstrings, string labels/paths and the
run-logging setup function; it retains executable numeric expressions and control
flow. Model dictionary filenames are independently checked unchanged. The additional
input-depth wrapper preserves the exact gene-metric computation while replacing
production job-receipt discovery with explicit portable manifest/hash checks.
It has not been executed on unavailable original predictions.

## Limits that remain explicit

Full model reproduction requires external assets: 24 curated expression profiles,
120 additional ranked sensitivity lists, four encoder files, two dictionaries,
174 final-epoch checkpoints and 576 prediction files. Sizes are unknown when bytes
were unavailable. Recorded hashes are labeled as provenance-only, not freshly
verified. The additional ranked lists have no hashes in the retained final manifest.
A complete original training environment lockfile and a raw-read-to-curated-matrix
workflow are missing. Model training/inference and complete prediction-based metric
reevaluation were not rerun during packaging.

The original final figures retain `true`/`True` labels for measured expression and
some `state` labels for transcriptomic/condition structure. These terms are explained
in the release documentation; biological-condition descriptions use manuscript
terminology. Source schema fields remain unchanged. Figure1's small layout
reproduction discrepancy does not affect numerical validation; the exact final
schematic export is included.

The future supplementary archive manifest covers 66 locally verified detailed
evaluation files, 147,916,292 bytes, excluded from GitHub. None has been uploaded to
Zenodo. Missing model/input assets must be resolved before claiming a complete
training/prediction archive, and third-party redistribution terms must be checked.

The repository remains private pending author review and a separate, explicit
visibility approval. No GitHub release, version tag or Zenodo deposit is created.
