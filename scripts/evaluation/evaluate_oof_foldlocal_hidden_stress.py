#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd


import os
ROOT = Path(os.environ['TOP2ALL_PROJECT'])
META_FILE = Path(os.environ['TOP2ALL_META'])
FULL_DIR = Path(os.environ['TOP2ALL_FULL'])
TOPN_DIR = Path(os.environ['TOP2ALL_TOP'])
GENE_FILE = Path(os.environ['TOP2ALL_GENES'])
OUT_DIR = Path(os.environ['TOP2ALL_STRICT_OUT'])
OUT_DIR.mkdir(parents=True, exist_ok=False)
TOPN = 1000
EPS = 1e-12
MODEL_ROOTS = {f'seed{a}_{b}':Path(os.environ['TOP2ALL_VIEW'])/f'seed_log1p{a}_scale{b}'/'oof_cv/top1000_nocond' for a,b in [(42,1234),(43,1235),(44,1236)]}

def get_rep(sid):
    m = re.search(r"_rep([123])$", str(sid))
    if not m:
        raise RuntimeError(f"Cannot parse replicate: {sid}")
    return int(m.group(1))


def metric(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    rmse = float(np.sqrt(np.mean((p-y)**2)))
    mae = float(np.mean(np.abs(p-y)))

    if np.std(y) > EPS and np.std(p) > EPS:
        pearson = float(np.corrcoef(y, p)[0, 1])

        ry = pd.Series(y).rank().to_numpy()
        rp = pd.Series(p).rank().to_numpy()

        if np.std(ry) > EPS and np.std(rp) > EPS:
            spearman = float(np.corrcoef(ry, rp)[0, 1])
        else:
            spearman = np.nan
    else:
        pearson = np.nan
        spearman = np.nan

    sst = np.sum((y-y.mean())**2)

    if sst > EPS:
        r2 = float(
            1.0
            - np.sum((p-y)**2) / sst
        )
    else:
        r2 = np.nan

    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def infer_prediction(df, path):

    # Prefer final raw-scale prediction.
    exact_raw = [
        "raw_pred_wo_overwrite",
        "raw_pred_final",
        "pred_raw_final",
        "raw_prediction_final",
        "prediction_raw_final",
        "predicted_raw_final",
        "raw_pred",
        "pred_raw",
        "predicted_raw",
    ]

    for c in exact_raw:
        if c in df.columns:
            x = pd.to_numeric(
                df[c],
                errors="raise",
            ).to_numpy(float)

            if x.min() < -1e-5:
                raise RuntimeError(
                    f"{path}: materially negative raw prediction "
                    f"in {c}: min={x.min()}"
                )

            return np.log1p(
                np.maximum(x, 0.0)
            ), c, "raw"

    # General raw + pred search
    candidates = [
        c for c in df.columns
        if "pred" in c.lower()
        and "raw" in c.lower()
    ]

    if candidates:
        candidates = sorted(
            candidates,
            key=lambda x: (
                "final" not in x.lower(),
                x,
            ),
        )

        c = candidates[0]

        x = pd.to_numeric(
            df[c],
            errors="raise",
        ).to_numpy(float)

        if x.min() < -1e-5:
            raise RuntimeError(
                f"{path}: materially negative raw prediction "
                f"in {c}: min={x.min()}"
            )

        return np.log1p(
            np.maximum(x, 0.0)
        ), c, "raw"

    # Fall back to a log1p prediction column
    candidates = [
        c for c in df.columns
        if "pred" in c.lower()
        and (
            "log1p" in c.lower()
            or "log" in c.lower()
        )
    ]

    if candidates:
        c = sorted(candidates)[0]
        return (
            pd.to_numeric(
                df[c],
                errors="raise",
            ).to_numpy(float),
            c,
            "log1p",
        )

    raise RuntimeError(
        f"Cannot infer prediction column in {path}\n"
        f"columns={list(df.columns)}"
    )


def prediction_file(model_root, rep, sid):

    pred_dir = (
        model_root
        / f"fold{rep}_test_rep{rep}"
        / "prediction"
    )

    candidates = sorted(
        p for p in pred_dir.rglob("*.csv")
        if sid in p.name
        and "pred_vs_true" in p.name
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"{sid}: expected one prediction file under "
            f"{pred_dir}; found {len(candidates)}\n"
            + "\n".join(str(x) for x in candidates)
        )

    return candidates[0]


# ============================================================
# Load reference data
# ============================================================

meta = pd.read_csv(META_FILE)
meta["sample_id"] = meta["sample_id"].astype(str)
meta["rep"] = meta["sample_id"].map(get_rep)

genes = [
    x.strip()
    for x in GENE_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

if len(genes) != 24410:
    raise RuntimeError(
        f"Expected 24410 genes; got {len(genes)}"
    )

if len(set(genes)) != 24410:
    raise RuntimeError("Duplicate reference genes")

gene2idx = {
    g: i for i, g in enumerate(genes)
}

top_lists = {}
top_maps = {}
truth_log = {}
condition = {}

for _, row in meta.iterrows():

    sid = row["sample_id"]
    condition[sid] = str(row["cell_type"])

    t = pd.read_csv(
        TOPN_DIR / row["top1000_file"]
    ).iloc[:TOPN].copy()

    t["Gene"] = t["Gene"].astype(str)

    top_lists[sid] = t["Gene"].tolist()

    top_maps[sid] = dict(
        zip(
            t["Gene"],
            np.log1p(
                pd.to_numeric(
                    t["Expression"],
                    errors="raise",
                ).to_numpy(float)
            ),
        )
    )

    f = pd.read_csv(
        FULL_DIR / row["full_file"]
    )

    fmap = dict(
        zip(
            f["Gene"].astype(str),
            pd.to_numeric(
                f["Expression"],
                errors="raise",
            ).to_numpy(float),
        )
    )

    truth_log[sid] = np.array(
        [
            np.log1p(fmap[g])
            for g in genes
        ],
        dtype=float,
    )


# ============================================================
# Load Top2All predictions
# ============================================================

top2all = {}

prediction_column_qc = []

for model_name, model_root in MODEL_ROOTS.items():

    top2all[model_name] = {}

    for _, row in meta.iterrows():

        sid = row["sample_id"]
        rep = int(row["rep"])

        pf = prediction_file(
            model_root,
            rep,
            sid,
        )

        df = pd.read_csv(pf)

        gene_col = (
            "gene"
            if "gene" in df.columns
            else "Gene"
        )

        pg = df[gene_col].astype(str).tolist()

        if len(pg) != 24410:
            raise RuntimeError(
                f"{pf}: expected 24410 rows"
            )

        pred_log, pred_col, scale = \
            infer_prediction(df, pf)

        pmap = dict(zip(pg, pred_log))

        top2all[model_name][sid] = np.array(
            [pmap[g] for g in genes],
            dtype=float,
        )

        prediction_column_qc.append({
            "model": model_name,
            "sample_id": sid,
            "prediction_file": str(pf),
            "prediction_column": pred_col,
            "prediction_scale": scale,
        })


# ============================================================
# Evaluate fold by fold
# ============================================================

sample_rows = []
neighbor_rows = []

# Store observations for masked gene-wise analysis.
# model -> gene index -> lists
gene_truth = {}
gene_pred = {}

method_names = (
    list(MODEL_ROOTS.keys())
    + ["knn1", "knn3", "knn5"]
)

for m in method_names:
    gene_truth[m] = [
        [] for _ in genes
    ]
    gene_pred[m] = [
        [] for _ in genes
    ]


for heldout_rep in [1, 2, 3]:

    train_ids = (
        meta.loc[
            meta["rep"] != heldout_rep,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    test_ids = (
        meta.loc[
            meta["rep"] == heldout_rep,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    if len(train_ids) != 16:
        raise RuntimeError("Expected 16 training samples")

    if len(test_ids) != 8:
        raise RuntimeError("Expected 8 test samples")

    # --------------------------------------------------------
    # Training-only Top1000 union
    # --------------------------------------------------------

    train_union = set()

    for sid in train_ids:
        train_union.update(
            top_lists[sid]
        )

    # --------------------------------------------------------
    # kNN feature space: training-only Top1000 union
    # --------------------------------------------------------

    features = sorted(train_union)

    f2i = {
        g: i for i, g in enumerate(features)
    }

    def sparse_vector(sid):

        x = np.zeros(
            len(features),
            dtype=float,
        )

        for g, value in top_maps[sid].items():
            j = f2i.get(g)

            if j is not None:
                x[j] = value

        return x

    Xtrain = np.stack(
        [
            sparse_vector(sid)
            for sid in train_ids
        ]
    )

    Xtest = np.stack(
        [
            sparse_vector(sid)
            for sid in test_ids
        ]
    )

    mu = Xtrain.mean(axis=0)
    sd = Xtrain.std(axis=0)

    sd[sd < EPS] = 1.0

    Xtrain = (
        Xtrain - mu
    ) / sd

    Xtest = (
        Xtest - mu
    ) / sd

    print(
        f"rep{heldout_rep}: "
        f"training Top1000 union={len(train_union)}"
    )

    for ti, sid in enumerate(test_ids):

        # ----------------------------------------------------
        # Strict evaluation mask
        #
        # 1. gene absent from training Top1000 union
        # 2. gene absent from THIS test sample Top1000
        #
        # No test expression values are used to define mask.
        # ----------------------------------------------------

        test_observed = set(
            top_lists[sid]
        )

        strict_genes = [
            g for g in genes
            if g not in train_union
            and g not in test_observed
        ]

        strict_idx = np.array(
            [
                gene2idx[g]
                for g in strict_genes
            ],
            dtype=int,
        )

        y = truth_log[sid][strict_idx]

        # ----------------------------------------------------
        # kNN ranking
        # ----------------------------------------------------

        dist = np.sqrt(
            np.mean(
                (
                    Xtrain
                    - Xtest[ti][None, :]
                ) ** 2,
                axis=1,
            )
        )

        ranking = sorted(
            zip(
                train_ids,
                dist.tolist(),
            ),
            key=lambda x: (
                x[1],
                x[0],
            ),
        )

        for rank, (neighbor_sid, d) \
                in enumerate(ranking, start=1):

            neighbor_rows.append({
                "heldout_rep":
                    heldout_rep,
                "test_sample":
                    sid,
                "test_condition":
                    condition[sid],
                "rank":
                    rank,
                "neighbor_sample":
                    neighbor_sid,
                "neighbor_condition":
                    condition[neighbor_sid],
                "same_condition":
                    condition[neighbor_sid]
                    == condition[sid],
                "distance":
                    d,
            })

        predictions = {}

        # Top2All 3 seeds
        for model_name in MODEL_ROOTS:
            predictions[model_name] = (
                top2all[model_name][sid][
                    strict_idx
                ]
            )

        # kNN
        for k in [1, 3, 5]:

            nn = [
                x[0]
                for x in ranking[:k]
            ]

            full_pred = np.mean(
                np.stack(
                    [
                        truth_log[n]
                        for n in nn
                    ]
                ),
                axis=0,
            )

            predictions[f"knn{k}"] = (
                full_pred[strict_idx]
            )

        # ----------------------------------------------------
        # Sample-wise and masked gene-wise storage
        # ----------------------------------------------------

        for method, p in predictions.items():

            m = metric(y, p)

            sample_rows.append({
                "heldout_rep":
                    heldout_rep,
                "sample_id":
                    sid,
                "condition":
                    condition[sid],
                "method":
                    method,
                "n_strict_hidden_genes":
                    len(strict_genes),
                **m,
            })

            for local_j, global_j \
                    in enumerate(strict_idx):

                gene_truth[method][global_j].append(
                    y[local_j]
                )

                gene_pred[method][global_j].append(
                    p[local_j]
                )


# ============================================================
# Sample-wise outputs
# ============================================================

sample_df = pd.DataFrame(sample_rows)

sample_df.to_csv(
    OUT_DIR / "strict_samplewise_metrics.csv",
    index=False,
)

sample_summary_rows = []

for method, x in sample_df.groupby(
    "method",
    sort=False,
):

    row = {
        "method": method,
        "n_samples": len(x),
        "strict_hidden_genes_median":
            x["n_strict_hidden_genes"].median(),
        "strict_hidden_genes_min":
            x["n_strict_hidden_genes"].min(),
        "strict_hidden_genes_max":
            x["n_strict_hidden_genes"].max(),
    }

    for col in [
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
    ]:
        row[f"{col}_mean"] = x[col].mean()
        row[f"{col}_median"] = x[col].median()

    sample_summary_rows.append(row)

sample_summary = pd.DataFrame(
    sample_summary_rows
)

sample_summary.to_csv(
    OUT_DIR / "strict_samplewise_summary.csv",
    index=False,
)


# ============================================================
# Gene-wise metrics with variable availability
#
# Require:
#   >=12 eligible held-out observations
#   true variance > 0
# ============================================================

gene_rows = []

for method in method_names:

    for j, gene in enumerate(genes):

        y = np.asarray(
            gene_truth[method][j],
            dtype=float,
        )

        p = np.asarray(
            gene_pred[method][j],
            dtype=float,
        )

        n = len(y)

        if n < 12:
            continue

        if np.std(y) <= EPS:
            continue

        m = metric(y, p)

        gene_rows.append({
            "method": method,
            "Gene": gene,
            "n_eligible_samples": n,
            **m,
        })

gene_df = pd.DataFrame(gene_rows)

gene_df.to_csv(
    OUT_DIR / "strict_genewise_metrics_min12.csv",
    index=False,
)

gene_summary_rows = []

for method, x in gene_df.groupby(
    "method",
    sort=False,
):

    row = {
        "method": method,
        "n_genes": len(x),
        "eligible_samples_median":
            x["n_eligible_samples"].median(),
    }

    for col in [
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
    ]:
        row[f"{col}_median"] = (
            x[col].median()
        )
        row[f"{col}_mean"] = (
            x[col].mean()
        )

    gene_summary_rows.append(row)

gene_summary = pd.DataFrame(
    gene_summary_rows
)

gene_summary.to_csv(
    OUT_DIR / "strict_genewise_summary_min12.csv",
    index=False,
)


# ============================================================
# Paired sample-wise kNN vs Top2All
# ============================================================

paired_rows = []

for seed in MODEL_ROOTS:

    a = (
        sample_df[
            sample_df["method"] == seed
        ]
        .set_index("sample_id")
    )

    for k in [1, 3, 5]:

        b = (
            sample_df[
                sample_df["method"] == f"knn{k}"
            ]
            .set_index("sample_id")
        )

        common = a.index.intersection(
            b.index
        )

        row = {
            "top2all_seed": seed,
            "knn": f"knn{k}",
            "n_samples": len(common),
        }

        # Higher is better
        for metric_name in [
            "pearson",
            "spearman",
            "r2",
        ]:
            row[
                f"knn_win_fraction_{metric_name}"
            ] = float(
                (
                    b.loc[common, metric_name]
                    >
                    a.loc[common, metric_name]
                ).mean()
            )

        # Lower is better
        for metric_name in [
            "rmse",
            "mae",
        ]:
            row[
                f"knn_win_fraction_{metric_name}"
            ] = float(
                (
                    b.loc[common, metric_name]
                    <
                    a.loc[common, metric_name]
                ).mean()
            )

        paired_rows.append(row)

paired_df = pd.DataFrame(
    paired_rows
)

paired_df.to_csv(
    OUT_DIR
    / "strict_knn_vs_top2all_samplewise_wins.csv",
    index=False,
)


# ============================================================
# Neighbor output
# ============================================================

neighbor_df = pd.DataFrame(
    neighbor_rows
)

neighbor_df.to_csv(
    OUT_DIR / "strict_knn_neighbors.csv",
    index=False,
)


# ============================================================
# QC
# ============================================================

pd.DataFrame(
    prediction_column_qc
).to_csv(
    OUT_DIR / "prediction_column_qc.csv",
    index=False,
)

qc = pd.DataFrame(
    [
        ("reference_gene_universe", 24410),
        ("n_samples", 24),
        ("n_oof_folds", 3),
        ("train_samples_per_fold", 16),
        ("test_samples_per_fold", 8),
        (
            "evaluation_definition",
            "gene absent from training Top1000 union "
            "AND absent from held-out sample Top1000"
        ),
        (
            "test_expression_used_for_target_definition",
            "NO"
        ),
        (
            "heldout_sample_top1000_identity_used",
            "YES, only to exclude observed input genes"
        ),
        (
            "gene_wise_min_eligible_samples",
            12
        ),
    ],
    columns=["item", "value"],
)

qc.to_csv(
    OUT_DIR / "strict_stress_test_qc.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# Print concise report
# ============================================================

print()
print("=" * 80)
print("FOLD-LOCAL STRICT HIDDEN-GENE STRESS TEST")
print("=" * 80)

counts = (
    sample_df[
        [
            "sample_id",
            "n_strict_hidden_genes",
        ]
    ]
    .drop_duplicates()
)

print(
    "\nStrict hidden genes/sample:"
)
print(
    "median =",
    int(counts["n_strict_hidden_genes"].median()),
    "range =",
    (
        int(counts["n_strict_hidden_genes"].min()),
        int(counts["n_strict_hidden_genes"].max()),
    ),
)

print(
    "\nSample-wise summary:"
)
print(
    sample_summary.to_string(
        index=False
    )
)

print(
    "\nGene-wise summary "
    "(>=12 eligible samples, true-variable):"
)
print(
    gene_summary.to_string(
        index=False
    )
)

print(
    "\nPaired kNN win fractions:"
)
print(
    paired_df.to_string(
        index=False
    )
)

print()
print("[WRITE]", OUT_DIR)
print("\nDONE")
