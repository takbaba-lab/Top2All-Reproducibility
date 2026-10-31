# Top2All: Benchmarking sparse-to-transcriptome reconstruction from high-expression profiles across sex- and sex-steroid-defined conditions

Authors: Yuki Nishio, Takayoshi Yamashita, Takashi Baba

Corresponding author: Takashi Baba

Version: **v1.0.0**

GitHub release date: **2026-11-01**

Zenodo DOI: [10.5281/zenodo.23361275](https://doi.org/10.5281/zenodo.23361275).
This is the reserved DOI for the core archive; resolution depends on publication
by the deposit owner. It is not a manuscript DOI.

Associated code: [Top2All-Reproducibility](https://github.com/takbaba-lab/Top2All-Reproducibility).
The manuscript-associated GitHub repository supplies training/prediction/evaluation/
plotting code, compact source tables, metadata, validation folds, figures, a
provenance-linked registry and reproduction instructions.

Raw/source RNA-seq: **DDBJ DRA015952**. Raw reads are not duplicated here.

## Core archive

`Top2All_Reproducibility_Core_v1.0.0.tar.gz` unpacks into
`Top2All_Reproducibility_Core_v1.0.0/`. Its SHA-256 is
`26446ed0b8bb8913ee745047855464cc53a112b468514160039fda69e280f4c3`.
It contains **846 files**, totaling **848,696,315 uncompressed file bytes**:

- `inputs/full/`: 24 exact curated full-expression profiles.
- `inputs/topN/`: 168 exact ranked inputs, 24 each for Top100/200/300/500/1000,
  production Top2000 and original Top3000.
- `predictions/` and `input_depth/`: 576 exact formal prediction files. Formal
  evaluation uses `raw_pred_wo_overwrite` before observed-gene replacement.
- `detailed_evaluations/`: 66 detailed numerical evaluations.
- `metadata/`: six sample metadata, validation fold and gene-order/gene-set files.
- `archive_metadata/`: four provenance/resource manifests, a historical preparation
  README and the original internal SHA-256 list.

All included project assets match their retained identity hashes. Original bytes,
rank order and line endings are preserved. Production Top2000 preserves the first
2,000 Top3000 data rows; its original CRLF serialization differs from Top3000 LF,
while paired values/order and line-ending-normalized prefixes agree.

The 174 project-trained checkpoints (~23.9 GB; 23,885,006,370 bytes) are not
included in the core archive and may be distributed separately if needed. Third-party
Mouse-Geneformer weights/dictionaries, software packages, Arial and DDBJ raw reads
are excluded. Provider identities/hashes and terms information are documented in
GitHub; no redistribution right is inferred from an identity match.

The immutable `archive_metadata/` files are historical preparation snapshots.
Administrative review/availability/publication wording in those snapshots does
not define the current release state. Use this companion README and GitHub v1.0.0
documentation for current scope; the snapshots' scientific identities/checksums
remain unchanged. Manifest staging paths describe the preparation machine; use
`intended_archive_path` relative to the unpacked archive root to locate a member.

## Verification and limits

Verify the compressed archive with its companion `.sha256` file. From the unpacked
archive root, run `shasum -a 256 -c archive_metadata/checksums.sha256` (or GNU
`sha256sum -c`). The internal list covers 845 members and excludes itself. All 846
member hashes, including the checksum file, were checked against the release
manifest during finalization.

The original raw-read-to-curated-matrix workflow and original run-bound complete
software lockfile are unavailable. The core archive supports analysis from exact
curated inputs and saved predictions; it is not a self-contained raw-read processing
or original training environment. Full retraining requires separately obtained
pretrained resources and a suitable environment. Release audits do not constitute
a new run of model training/inference or all scientific evaluations.

MIT applies to original project code and documentation. Data and third-party
resources retain their applicable terms; MIT does not relicense external resources.
