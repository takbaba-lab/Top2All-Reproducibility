# Top2All reproducibility repository

**Top2All: Benchmarking sparse-to-transcriptome reconstruction from high-expression profiles across sex- and sex-steroid-defined conditions**

Yuki Nishio, Takayoshi Yamashita, Takashi Baba  
Corresponding author: Takashi Baba ([email](mailto:baba.takashi.547@m.kyushu-u.ac.jp)).

Top2All reconstructs transcriptome-wide expression from sample-specific sparse
high-expression inputs using a frozen Mouse-Geneformer encoder, an expression
reconstruction module and a sample-wise scale module. This repository accompanies
the manuscript and provides final figures, numerical source tables, code, sample
partitions and checks needed to inspect the benchmark.

The study uses **24 mouse adrenal zona fasciculata cell-number-normalized (CNN)
RNA-seq samples across eight sex- and sex-steroid-defined conditions**, with three
biological replicates per condition. Raw/source RNA-seq data used in the paper are
available from **DDBJ accession [DRA015952](https://ddbj.nig.ac.jp/resource/sra-submission/DRA015952)**.
Raw reads are not duplicated here.

## Benchmark design

The conditions are sham-operated female and male, ovariectomized female with Oil,
DHT or E2, and castrated male with Oil, DHT or E2. `Control_female` and `Control_male`
are the corresponding sample identifiers for the sham groups. Oil denotes vehicle,
DHT dihydrotestosterone, and E2 17β-estradiol. The `cell_type` metadata column stores
these biological conditions for compatibility with the model programs; every sample
is from the same cell population.

- **Replicate-wise out-of-fold (OOF):** three folds, each with 16 training samples
  and eight held-out samples, one replicate per condition. Both condition-aware and
  no-condition models use identical partitions.
- **Leave-one-condition-out (LOCO):** eight folds, each with 21 training samples and
  the three replicates of one unseen condition held out. Models receive no explicit
  condition labels.
- **Reference baselines:** GlobalMean and SameConditionMean use only the appropriate
  OOF training profiles. kNN1, kNN3 and kNN5 select neighbors from the training fold
  using standardized sparse nominal Top1000 expression and average their measured
  full log1p profiles. LOCO also includes a zero-condition-shift control.
- **Evaluation:** sample-wise and gene-wise metrics in log1p expression space,
  replicate residuals, within-condition LOCO residuals and LOCO centroid shifts.
  High sample-wise correlation alone does not establish recovery of gene-specific
  biological variation. All formal evaluations use `raw_pred_wo_overwrite`, before
  replacing observed input genes with their measured values.
- **Targets:** the universe contains 24,410 genes. The nominal Top1000 union contains
  1,312 genes; the fixed complement contains 23,098 genes, including 6,951 genes with
  positive measured log1p expression SD across the 24 samples. Strict fold-local OOF
  additionally excludes the training-fold input union and the test sample's input
  genes (23,111–23,136 eligible genes per sample); its retained gene-wise set here is
  the same 6,951 genes, each observed in all 24 held-out samples.
- **Robustness:** reconstruction/scale seed pairs are 42/1234, 43/1235 and 44/1236.
  Target-variability quartiles use measured sample SD (`ddof=1`) exclusively.
  Input-depth sensitivity evaluates nominal TopN = 100, 200, 300, 500, 1000 and 2000
  on 5,605 fixed true-variable non-Top2000 genes. A bridge analysis uses the main
  6,951-gene target at depths through Top1000; Top2000 is intentionally excluded
  from this bridge to avoid input/target overlap.

Nominal TopN lists are filtered by tokenizability without replacement. Retained
paired gene/expression values preserve rank order. Within-sample log1p values use
population SD for normalization; the scale feature uses the sum of retained linear
expression values and a natural logarithm with epsilon 1e-6. Computational seeds,
biological replicates and validation folds are distinct concepts.

## Contents

```text
Top2All-Reproducibility/
├── README.md, LICENSE, CITATION.cff
├── requirements.txt, requirements-models.txt
├── metadata/                 # 24 samples, OOF/LOCO folds, ordered gene identities
├── scripts/
│   ├── preprocessing/        # shared token/expression preprocessing
│   ├── training/             # reconstruction/scale models and fold command runner
│   ├── prediction/           # condition-aware and no-condition prediction
│   ├── evaluation/           # metrics, baselines, residuals, shifts, validator
│   └── plotting/             # all eight manuscript figures
├── source_data/
│   ├── Figure1/ … Figure5/   # plot values and numerical sources
│   └── FigureS1/ … FigureS3/
├── figures/                  # final Figures 1–5 and S1–S3 in PNG/PDF/SVG
├── docs/FIGURE_LEGENDS.md     # manuscript-facing figure legends
└── reproducibility/          # registry, hashes, asset inventories, instructions
```

[Figure legends](docs/FIGURE_LEGENDS.md) describe the panels and numerical definitions.

Supplementary Figure S1 presents target-gene variability, S2 seed robustness and
S3 input-depth sensitivity. Each figure directory includes a `registry_links.tsv`
that connects plotted numerical cells to the result registry. Internal `true` field names in numerical tables are retained for schema compatibility
and denote measured expression. Visible figure labels and metadata titles use
measured/condition terminology, with technical true-variable terminology retained.

## Reproduce from GitHub alone

From the repository root, Python 3.12 can check all release files, folds and numerical
source reductions without installing third-party packages:

```bash
python scripts/evaluation/validate_release.py
```

To regenerate the eight figures from the included tables, install the figure
environment and make Arial available to Matplotlib:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/plotting/reproduce_figures.py --out-dir reproduced_figures
```

The output directory must be new. Committed figures and numerical sources are not
modified. Newly generated files may differ in timestamps or SVG identifiers; see
[environment details](reproducibility/ENVIRONMENT.md).

GitHub alone supports figure regeneration, exact registry/source-table checks,
fold inspection and re-summarizing the included numerical tables. It does not
provide the complete curated expression matrix, pretrained encoder/dictionaries,
trained checkpoints or per-sample prediction matrices. All 24 nominal Top1000
ranked inputs are included and match retained production hashes. All 24 full profiles,
168 ranked inputs and 576 formal predictions have now been recovered and verified
in ignored local staging. Full training, prediction,
metric evaluation from model outputs and raw-read processing need external assets.

## Reproduce with external assets

See [the reproduction guide](reproducibility/README.md) for model parameters,
fold commands, evaluation layout and asset requirements.
[EXTERNAL_ASSETS_REQUIRED.tsv](reproducibility/EXTERNAL_ASSETS_REQUIRED.tsv) records
required asset identities, verified local inputs and unresolved dependencies,
distinguishing recorded provenance hashes from files reopened locally. It is not a download manifest or a claim that those assets
have already been archived. The missing exact raw-read-to-curated-matrix workflow
and run-bound software lockfile are documented there and in the guide.

[ZENODO_MANIFEST.tsv](reproducibility/ZENODO_MANIFEST.tsv) retains 1,020 records:
**846 locally staged files (848,696,315 bytes)** and **174 checkpoints not staged**.
All expected production assets were located and SHA-256 verified on Genkai.
Checkpoint transfer (23,885,006,370 bytes) awaits a storage decision. The staging
contains all 24 curated full profiles, 168 ranked inputs, 576 formal predictions,
66 detailed evaluations and metadata/manifests/checksums. Files have not been
uploaded to Zenodo. See [the recovery report](reproducibility/GENKAI_RECOVERY_REPORT.md)
and [verification records](reproducibility/GENKAI_ASSET_VERIFICATION.tsv).

Third-party weights, dictionaries, software packages, Arial and DDBJ raw reads
are external references only. Provider URLs, identities/hashes and unresolved terms
are recorded in [THIRD_PARTY_RESOURCES.tsv](reproducibility/THIRD_PARTY_RESOURCES.tsv).
The repository must remain private pending Takayoshi Yamashita's confirmation;
no public visibility change, release, upload or DOI issuance is authorized.

## Citation and release status

Use `CITATION.cff` for the author order and manuscript title. No manuscript DOI or
Zenodo DOI has been assigned in this repository. A Zenodo DOI can be added after
archiving the approved release. The original project code and accompanying release
documentation use the MIT license; external resources retain their own terms.
