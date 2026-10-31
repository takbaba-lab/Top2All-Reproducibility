# Top2All v1.0.0 release audit

Release date: 2026-11-01. Core archive DOI:
[10.5281/zenodo.23361275](https://doi.org/10.5281/zenodo.23361275), reserved for
publication by the deposit owner. This audit finalizes GitHub release metadata and
scope; it does not publish or replace Zenodo files.

| Check | Result |
|---|---|
| Release identity | v1.0.0; CFF version 1.0.0, DOI and 2026-11-01 release date |
| Registry/source reductions | PASS; 45,543 exact source reductions |
| Figure-registry linkage | PASS; 42,157 plotted numerical cell links |
| Figure set | Figures 1–5 and S1–S3, 24 exports; no regeneration or byte changes |
| Scientific file preservation | 160 protected files unchanged from the pre-release main commit |
| Source data/registry/model code | No source values, registry fields, analysis code or model specifications changed |
| Metadata/folds/seeds | Unchanged; three OOF/eight LOCO folds, no train/test overlap |
| Manuscript | Ten local manuscript DOCX files unchanged; no manuscript edit |
| Security/privacy | No detected local/HPC paths, machine usernames, credential signatures, internal instruction files or disallowed emails in release files; image/PDF metadata included |
| Core archive identity | SHA-256 matches 26446ed0b8bb8913ee745047855464cc53a112b468514160039fda69e280f4c3 |
| Core members | 846 files, 848,696,315 uncompressed bytes; every member hash matches the manifest and preserved local bytes |
| Checkpoint scope | 174 verified production checkpoints (~23.9 GB) explicitly excluded from core; optional separate distribution |
| Third-party scope | Pretrained weights/dictionaries, packages, fonts and raw reads excluded |
| Core privacy | No detected private filesystem paths, machine identities or credential signatures in archive member content |
| SHA-256 coverage | All release files except the checksum file itself listed and verified |

The immutable archive is `Top2All_Reproducibility_Core_v1.0.0.tar.gz`
(229,189,272 compressed bytes). Its included assets comprise 24 curated full
profiles, 168 ranked inputs, 576 formal predictions, 66 detailed evaluations and
12 metadata/provenance/documentation/checksum files. `CORE_ARCHIVE.json` fixes its
identity and scope. `ZENODO_MANIFEST.tsv` retains all 1,020 identities while
separating 846 included members from 174 excluded checkpoints; none is labeled
as an unlocated production asset. Expected input/checkpoint/prediction hashes are
unchanged. All 948 production/dependency identities were previously located and
hash-verified; six third-party encoder/dictionary resources remain external only.

Frozen metadata copies inside `archive_metadata/` are historical preparation
snapshots. Their bytes and identity hashes are retained; current v1.0.0 GitHub
documentation supersedes administrative availability/review/publication wording.
The companion `ZENODO_README.md` states the current scope and reserved DOI without
rewriting the core archive. Historical recovery/search reports retain acceptance
and rejection evidence with explicit time scope.

The finalization changes citation/documentation, availability/scope fields,
release identity and administrative validators only. Source hashes/values, final
figures, formal predictions, gene order, fold/seed definitions and scientific
metric/model logic are unchanged. No figure regeneration, model training,
inference or scientific reevaluation was performed. Historical figure regeneration
and source-preservation evidence remains in its existing records.

`validate_release.py` checks the scientific source reductions, figure links,
source/figure/script hashes, folds, display geometry and v1.0.0 archive inventory.
`audit_archive_assets.py --require-local-staging` also verifies all preserved local
core member bytes and internal checksum coverage. Ignored operating-system Finder
metadata is excluded from release coverage, consistent with `.gitignore`.

The original raw-read-to-curated-matrix workflow and original run-bound complete
software lockfile remain unavailable. Exact curated inputs/predictions support
analysis from those assets; the release is not a self-contained raw-read processing
or original training environment. External pretrained resources must be obtained
under their provider terms; identity matching does not establish redistribution
rights. The repository DOI is not a manuscript DOI.
