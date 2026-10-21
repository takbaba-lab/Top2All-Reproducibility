# Reproduction guide

## Included numerical evidence

`result_registry.tsv` contains 45,543 results. Each row identifies its validation
scheme, model, condition-input mode, seeds, metric, statistic, evaluation unit,
selected/valid counts, source table, JSON selector, source column, reduction,
plotting script and final figure. SHA-256 fields refer to release file bytes.
Scientific values, seeds and denominator fields were preserved from the selected
final local registry. Release result IDs are stable within this package.

The standard-library validator checks every source reduction and 42,157 plotted
cell links, figure identities, exact file checksums and sample partitions. It does
not claim to validate unavailable original predictions or rerun model training.
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

## External asset and archive status

`EXTERNAL_ASSETS_REQUIRED.tsv` records the final curated 24-sample expression files,
encoder files/dictionaries, final-epoch checkpoint identities and prediction files.
Its `sha256` values come from retained production provenance, not a new read of
missing remote files. Unknown sizes are explicitly marked `unknown`. This list
must be resolved against actual bytes before an archive is described as complete.
No server paths, usernames or execution receipts are published.

`ZENODO_MANIFEST.tsv` lists **66 locally verified final detailed evaluation files**
(total 147,916,292 bytes), with intended archive paths, sizes and SHA-256 hashes.
The preparation machine has ignored `.local/zenodo_assets/` files at each listed
`local_source_path`. Those relative paths disclose no private filesystem details.
The `.local/` files are administrative staging assets, not GitHub content; after a
fresh clone the original preparer must provide them or fetch the future archive.
No Zenodo upload or release has been made.

Large/raw/intermediate data, training logs, internal audits and development history
are excluded from GitHub. DDBJ DRA015952 remains the raw/source RNA-seq location.
DDBJ alone is not the exact curated 24,410-gene matrix used by these programs: the
complete read-processing/quantification and identifier-curation workflow is not
contained in the final local release sources. Reconstructing that exact matrix
requires author-supplied curated inputs or an independently documented preprocessing
workflow. Original ranked sensitivity lists, pretrained assets, original predictions
and checkpoints remain external dependencies. A complete original training lockfile
is also unavailable; see `ENVIRONMENT.md`.

## Archiving an approved release

Review the private repository, license, external asset status and all audit results.
After explicit author approval, make it public, enable its Zenodo integration, and
create the GitHub `v1.0.0` release once the archive contents and terms are settled.
Upload the approved supplementary assets using this manifest, rerun hashes, and add
the issued DOI to `CITATION.cff`, README and manuscript data/code availability.
No manuscript DOI should be inferred from the repository DOI.

For input-depth reevaluation, `evaluate_input_depth.py` uses a portable TSV with
`topn`, `seed_log1p`, `seed_scale`, `heldout_rep`, `sample_id`, `path` and `sha256`
columns (432 prediction files across 54 fold jobs). Paths resolve relative to the
TSV. The wrapper preserves the final numerical metric functions, fixed target
sets, native hidden-gene calculation, truth tolerances and hidden-overwrite checks;
it replaces production job-receipt discovery with explicit manifest/hash checks.
It has not been run on the unavailable full predictions during packaging:

```bash
python scripts/evaluation/evaluate_input_depth.py \
  --prediction-manifest external_assets/input_depth_predictions.tsv \
  --full-dir external_assets/full --topn-dir external_assets/topN \
  --out-dir outputs/evaluation/input_depth
```
