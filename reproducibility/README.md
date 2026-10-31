# Top2All v1.0.0 reproduction guide

## Included numerical evidence

`result_registry.tsv` contains 45,543 results. Each row identifies its validation
scheme, model, condition-input mode, seeds, metric, statistic, evaluation unit,
selected/valid counts, source table, JSON selector, source column, reduction,
plotting script and final figure. SHA-256 fields refer to release file bytes.
Scientific values, seeds and denominator fields were preserved from the selected
final local registry. Release result IDs are stable within this package.

The standard-library validator checks every source reduction and 42,157 plotted
cell links, figure identities, exact file checksums and sample partitions. It does
not rerun model training/inference or independently repeat scientific evaluation
of the recovered predictions.
Source tables contain undefined metrics where the original numerical definition
is undefined; they are not imputed. All scientific calculations retain their
original thresholds and formulas. Structural checks distinguish finite and
undefined values.

`model_provenance.tsv` identifies the 42 primary model fold/seed pairs (nine
condition-aware OOF, nine no-condition OOF and 24 no-condition LOCO) using recorded
checkpoint hashes and explicit training/test sample identities. Local packaging
retains model computation and numerical constants; path/import setup, logging
identity fields and release provenance labels are adapted. `script_preservation.tsv`
records source/release hashes, numeric-constant and dictionary-name checks for the
nine model engines. Original third-party dictionary filenames remain required.

## Model execution

Use a separate model environment and obtain the resources in
`EXTERNAL_ASSETS_REQUIRED.tsv`. Place curated sample files in `external_assets/full/`,
ranked nominal input lists in `external_assets/topN/`, encoder resources in
`external_assets/encoder/`, and dictionaries in `external_assets/dictionaries/`.
The `external_assets/` directory is ignored by Git.

The headerless `metadata/common_genes.txt` fixes the 24,410-gene order. Every full
expression profile must have this gene universe, with gene symbols and nonnegative
CNN expression. Each nominal input list must preserve the original ranked `Gene`
and `Expression` pairs, with exactly N unique genes, prior to token filtering.
Top1000 lists are also retained under `source_data/Figure1/input_gene_lists/`.
The final input-depth procedure used existing Top100/200/300/500/1000 files and the
first 2,000 rows of existing Top3000 lists. Regenerating tie order from a fresh sort
is not assumed equivalent to the original ranked lists.

The following prints the three commands for one OOF fold without running a model:

```bash
python scripts/training/run_training.py \
  --family condition_oof --heldout-rep 1 --seed 42 --topn 1000 \
  --model-dir external_assets/encoder \
  --dictionary-dir external_assets/dictionaries \
  --full-dir external_assets/full --topn-dir external_assets/topN \
  --output outputs/condition_oof/seed42_1234/fold1
```

Add `--execute` to train the reconstruction module, train its paired scale module,
and predict held-out samples. Output directories must be new. Use `nocond_oof`
for no-condition OOF, or `nocond_loco --heldout-condition Cas_DHT` for one LOCO
condition (omit `--heldout-rep` for LOCO). Repeat seeds 42, 43 and 44 with paired
scale seeds 1234, 1235 and 1236, all three OOF folds and all eight LOCO conditions.
Input-depth sensitivity uses `nocond_oof` at 100, 200, 300, 500, 1000 and 2000.

The published commands use reconstruction AdamW learning rate 1e-4, weight decay
0.01, batch size 4, 30 epochs; mean pooling, Softplus, expression loss weight 0.3,
positive-class weight 3 and nonzero-loss coefficient 1. The scale module uses
learning rate 1e-4, zero weight decay, batch size 4, 50 epochs and epsilon 1e-6.
Both stages use final-epoch checkpoints. The original sequence-cap arguments are
preserved: 1000 for condition-aware OOF and LOCO and 2048 for no-condition OOF/
input-depth sensitivity; nominal ranked inputs remain N and are filtered without
refill. Encoder weights stay frozen; encoder dropout remains active during
reconstruction training and is disabled during scale training and inference.

The runner preserves model output filenames expected by the evaluators. Some
checkpoint class names, third-party filenames and output schema fields retain
identifiers required for compatibility; these are not alternative analyses.

## Evaluation from predictions

The original evaluation scripts accept directory arguments. Use `--help` on each
program for its complete interface. Assemble OOF predictions under
`external_assets/prediction_view/seed_log1p42_scale1234/oof_cv/top1000/` or
`top1000_nocond/`, with `fold1_test_rep1/prediction/` through
`fold3_test_rep3/prediction/`. A nested model output directory is accepted by the
original recursive file discovery. LOCO uses
`external_assets/prediction_view/seed_log1p42_scale1234/loco/top1000_nocond/test_<condition>/prediction/`.
Repeat for the other two seed pairs.

```bash
python scripts/evaluation/evaluate_oof_genewise_fixed_nonTopN.py \
  --oof_base_dir external_assets/prediction_view/seed_log1p42_scale1234/oof_cv/top1000 \
  --metadata_csv metadata/sample_metadata.csv \
  --topn_dir external_assets/topN --topn 1000 \
  --out_dir outputs/evaluation/oof_condition_seed42
```

`evaluate_oof_nonTopN.py` provides sample-wise evaluation;
`compare_oof_top2all_vs_mean_baselines.py` adds mean baselines and measured-SD
quartiles; `evaluate_oof_within_condition_residuals.py` provides OOF residuals.
The LOCO evaluation, decomposition, centroid-shift and retrieval baseline scripts
use the corresponding LOCO prediction tree and the curated full-expression files.
Always select pre-overwrite predictions. The strict OOF evaluator receives its
paths through these environment variables (set them explicitly to existing assets):

```text
TOP2ALL_PROJECT       repository root
TOP2ALL_META          metadata/sample_metadata.csv
TOP2ALL_FULL          curated full-expression directory
TOP2ALL_TOP           nominal Top1000 directory
TOP2ALL_GENES         metadata/common_genes.txt
TOP2ALL_STRICT_OUT    new output directory
TOP2ALL_VIEW          external_assets/prediction_view
```

Then run `python scripts/evaluation/evaluate_oof_foldlocal_hidden_stress.py`.
These programs perform new evaluations into new output directories; they do not
modify the supplied final figures or registry. Per-gene reevaluation of input-depth
predictions also requires the fixed target identities in `metadata/`. Complete
summary/plot reproduction from the supplied final sources requires no predictions.

## Core archive and external dependencies

The v1.0.0 core archive is identified by Zenodo DOI
[10.5281/zenodo.23361275](https://doi.org/10.5281/zenodo.23361275), reserved for
publication by the deposit owner. The DOI may not resolve before owner publication.
The fixed archive is `Top2All_Reproducibility_Core_v1.0.0.tar.gz`:

- SHA-256: `26446ed0b8bb8913ee745047855464cc53a112b468514160039fda69e280f4c3`.
- Compressed bytes: 229,189,272.
- Included: 846 files, 848,696,315 uncompressed file bytes.
- Contents: 24 exact curated full profiles, 168 exact ranked inputs, 576 formal
  predictions, 66 detailed evaluations, six metadata files and six manifest,
  documentation and checksum files.
- Excluded: all 174 project-trained checkpoints (23,885,006,370 bytes, ~23.9 GB),
  third-party weights/dictionaries/software/fonts and raw reads.

The 174 project-trained checkpoints are not included in the core archive and may
be distributed separately if needed. Their exact originals were located and
SHA-256 verified; they are not missing identities. Retraining from the curated
inputs uses the released code and separately obtained third-party model resources.
Evaluation of saved formal predictions does not require those trained checkpoints.

`ZENODO_MANIFEST.tsv` retains 1,020 identity records: 846 included core members and
174 explicitly excluded project checkpoints. It records original hashes, verified
sizes, archive paths and release scope. `EXTERNAL_ASSETS_REQUIRED.tsv` records 948
production/dependency identities: all were located and hash-verified, comprising
192 inputs, 174 checkpoints, 576 predictions and six external encoder/dictionary
files. `GENKAI_ASSET_VERIFICATION.tsv` preserves portable expected/observed hash
records; the recovery report describes verification rather than a new analysis.

All 24 production Top2000 files preserve the first 2,000 Top3000 data rows in
header, paired Gene/Expression strings and rank order. Their original CRLF versus
LF serialization differs; line-ending-normalized prefixes agree. Both original
serializations match their independent provenance hashes. No input was regenerated.
All 576 formal prediction files contain 24,410 unique gene rows and the formal
`raw_pred_wo_overwrite` column. Other retained output columns must not replace it.

The manifest's `.local/zenodo_assets/` locators describe the preparation machine,
not files included in a fresh GitHub clone. After downloading and unpacking the
core archive, use `intended_archive_path` relative to
`Top2All_Reproducibility_Core_v1.0.0/` to locate a member. For example, pass its
`inputs/full/` and `inputs/topN/` directories to the CLI examples above, or copy
inputs to the ignored `external_assets/` layout. Prediction manifests should use
paths relative to the prediction manifest itself, as described below.

Verify the compressed archive with its companion `.sha256` file. Then, from the
unpacked archive root, verify `archive_metadata/checksums.sha256`; that internal
list covers 845 members and excludes itself. All 846 member hashes, including
the checksum file, were independently compared with the release manifest. The
[archive identity record](CORE_ARCHIVE.json) fixes the compressed hash and size.

The archive is immutable. Its `archive_metadata/README.md` and manifest copies
are historical preparation snapshots retained byte-for-byte for provenance.
Administrative availability, review or publication wording in those snapshots is
superseded by the current v1.0.0 GitHub documentation; source identities and
checksums remain unchanged. The [companion Zenodo README](ZENODO_README.md)
describes the current release scope without rewriting the frozen archive.

## Reproducibility limits

The original raw-read-to-curated-matrix workflow and original run-bound complete
software lockfile remain unavailable. DDBJ DRA015952 is the raw/source sequencing
reference; raw reads alone do not specify the exact curated CNN matrix. The core
archive supplies that exact matrix and the formal predictions, supporting
prediction-based evaluation. Full retraining additionally needs external pretrained
resources and an appropriate hardware/software environment. A byte-identity audit
is not a claim that training, inference or all scientific evaluations were rerun.

`THIRD_PARTY_RESOURCES.tsv` records provider identities, versions/hashes and terms
information. Those resources remain external only; matching hashes do not establish
redistribution rights. Original third-party versions/terms remain unresolved where
not recorded, and no third-party resource is included in the core archive.

For input-depth reevaluation, `evaluate_input_depth.py` uses a portable TSV with
`topn`, `seed_log1p`, `seed_scale`, `heldout_rep`, `sample_id`, `path` and `sha256`
columns (432 prediction files across 54 fold jobs). Paths resolve relative to the
TSV. The wrapper preserves the final numerical metric functions, fixed target
sets, native hidden-gene calculation, measured-profile tolerances and hidden-overwrite checks;
it replaces production job-receipt discovery with explicit manifest/hash checks.
The wrapper has not been rerun on the recovered predictions as part of release finalization:

```bash
python scripts/evaluation/evaluate_input_depth.py \
  --prediction-manifest external_assets/input_depth_predictions.tsv \
  --full-dir external_assets/full --topn-dir external_assets/topN \
  --out-dir outputs/evaluation/input_depth
```
