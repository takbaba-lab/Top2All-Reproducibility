#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Top2All LOCO centroid-shift reconstruction against simple baselines.

Baselines
---------
1) ZeroShift
   Predict no deviation from the 21-sample training centroid.

2) kNN-1 / kNN-3 / kNN-5
   Use ONLY sparse TopN input profiles to find nearest training samples.
   - Each LOCO fold builds its feature universe from TRAINING TopN genes only.
   - Each sample is represented by sparse log1p(TopN expression), with genes
     absent from that sample's TopN set filled with zero.
   - Euclidean distance is computed after feature-wise standardization using
     training samples only.
   - Hidden fixed-nonTopN profile is predicted as the mean full log1p profile
     of the k nearest training samples.
   - The 3 held-out replicate predictions are averaged to a condition state.
   - Shift is relative to the same 21-sample training centroid.

3) TrainingVariability ranking baseline (magnitude only)
   Rank hidden genes by SD across the 21 training full profiles.
   Compare Top-K overlap with true |held-out shift| Top-K.

Primary target genes
--------------------
Global fixed non-TopN:
prediction gene universe MINUS union of TopN genes across all 24 samples.

This matches the main Top2All evaluation target set.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TOP_KS = [100, 500, 1000]
KNN_KS = [1, 3, 5]


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt(np.sum(x0*x0) * np.sum(y0*y0))
    return np.nan if den == 0 else float(np.sum(x0*y0)/den)


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(float)
    yr = pd.Series(y).rank(method="average").to_numpy(float)
    return pearson(xr, yr)


def regression(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.var(x) == 0:
        return np.nan, np.nan
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(coef[0])


def metrics(t, p):
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    err = p - t
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    sst = float(np.sum((t - t.mean())**2))
    r2 = np.nan if sst == 0 else float(1 - np.sum(err**2)/sst)
    slope, intercept = regression(t, p)
    tsd = float(np.std(t, ddof=1))
    psd = float(np.std(p, ddof=1))
    nz = np.abs(t) > 1e-8
    signc = (
        float(np.mean(np.sign(t[nz]) == np.sign(p[nz])))
        if nz.any() else np.nan
    )
    return {
        "pearson": pearson(t, p),
        "spearman": spearman(t, p),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
        "true_sd": tsd,
        "pred_sd": psd,
        "pred_true_sd_ratio": np.nan if tsd == 0 else psd/tsd,
        "sign_concordance": signc,
    }


def parse_sid(filename):
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(filename)
    return filename[:-len(suffix)]


def parse_rep(sid):
    m = re.search(r"_rep(\d+)$", str(sid))
    if not m:
        raise ValueError(sid)
    return int(m.group(1))


def resolve_file(value, base):
    p = Path(str(value))
    if p.is_file():
        return p
    q = Path(base) / str(value)
    if q.is_file():
        return q
    raise FileNotFoundError(f"{value} under {base}")


def load_full_map(path):
    df = pd.read_csv(path)
    if not {"Gene", "Expression"}.issubset(df.columns):
        raise ValueError(f"{path}: need Gene, Expression")
    genes = df["Gene"].astype(str)
    expr = pd.to_numeric(df["Expression"], errors="coerce").fillna(0.0)
    return dict(zip(genes, expr))


def load_topn_map(path, topn):
    df = pd.read_csv(path)
    if "Gene" not in df.columns:
        raise ValueError(f"{path}: missing Gene")
    if "Expression" not in df.columns:
        raise ValueError(
            f"{path}: missing Expression; columns={df.columns.tolist()}"
        )
    if len(df) < topn:
        raise ValueError(f"{path}: fewer than Top{topn} rows")
    df = df.iloc[:topn].copy()
    genes = df["Gene"].astype(str)
    expr = pd.to_numeric(df["Expression"], errors="coerce").fillna(0.0)
    return dict(zip(genes, expr))


def vector_from_map(mapping, genes):
    return np.array(
        [float(mapping.get(g, 0.0)) for g in genes],
        dtype=float,
    )


def topk_overlap(true_abs, pred_abs, k):
    tr = np.argsort(true_abs)[::-1][:k]
    pr = np.argsort(pred_abs)[::-1][:k]
    T, P = set(tr.tolist()), set(pr.tolist())
    O = T & P
    U = T | P
    return len(O), len(O)/k, len(O)/len(U) if U else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco_base_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    loco = Path(args.loco_base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv).copy()
    topcol = f"top{args.topn}_file"
    need = {"sample_id", "cell_type", "full_file", topcol}
    missing = need - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing {sorted(missing)}")

    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["cell_type"] = meta["cell_type"].astype(str)
    meta["rep"] = meta["sample_id"].map(parse_rep)
    midx = meta.set_index("sample_id", drop=False)
    conditions = sorted(meta["cell_type"].unique().tolist())

    # Locate Top2All prediction files and gene universe.
    pred_files = {}
    gene_order = None
    for cond in conditions:
        pdir = loco / f"test_{cond}" / "prediction"
        files = sorted(pdir.rglob("*_pred_vs_true_v6_raw_final.csv"))
        if len(files) != 3:
            raise RuntimeError(f"{cond}: expected 3 predictions, got {len(files)}")
        for p in files:
            sid = parse_sid(p.name)
            pdf = pd.read_csv(
                p, usecols=["gene", "raw_pred_wo_overwrite"]
            )
            genes = pdf["gene"].astype(str).tolist()
            if gene_order is None:
                gene_order = genes
            elif genes != gene_order:
                raise RuntimeError(f"gene order mismatch: {p}")
            pred_files[sid] = p

    # Load TopN maps and global union.
    top_maps = {}
    top_union = set()
    for _, r in meta.iterrows():
        sid = r["sample_id"]
        tm = load_topn_map(
            resolve_file(r[topcol], args.topn_dir), args.topn
        )
        top_maps[sid] = tm
        top_union.update(tm.keys())

    fixed_genes = [g for g in gene_order if g not in top_union]
    fixed_idx = np.array(
        [i for i, g in enumerate(gene_order) if g not in top_union],
        dtype=int,
    )
    N = len(fixed_genes)

    # Load full hidden truth and Top2All hidden prediction in log1p.
    full_log = {}
    top2all_log = {}
    for _, r in meta.iterrows():
        sid = r["sample_id"]
        fmap = load_full_map(resolve_file(r["full_file"], args.full_dir))
        raw = vector_from_map(fmap, fixed_genes)
        full_log[sid] = np.log1p(np.clip(raw, 0, None))

        pdf = pd.read_csv(
            pred_files[sid], usecols=["gene", "raw_pred_wo_overwrite"]
        )
        pr = pd.to_numeric(
            pdf["raw_pred_wo_overwrite"], errors="coerce"
        ).to_numpy(float)[fixed_idx]
        top2all_log[sid] = np.log1p(np.clip(pr, 0, None))

    metric_rows = []
    topk_rows = []
    nn_rows = []

    for heldout in conditions:
        test_sids = meta.loc[
            meta["cell_type"] == heldout, "sample_id"
        ].tolist()
        train_sids = meta.loc[
            meta["cell_type"] != heldout, "sample_id"
        ].tolist()

        # Hidden-expression truth objects.
        train_hidden = np.vstack([full_log[s] for s in train_sids])
        train_centroid = train_hidden.mean(axis=0)
        true_state = np.vstack([full_log[s] for s in test_sids]).mean(axis=0)
        true_shift = true_state - train_centroid

        # Top2All.
        top2all_state = np.vstack(
            [top2all_log[s] for s in test_sids]
        ).mean(axis=0)
        top2all_shift = top2all_state - train_centroid

        method_shifts = {
            "Top2All": top2all_shift,
            "ZeroShift": np.zeros_like(true_shift),
        }

        # Build strict training-only TopN feature universe for kNN.
        train_feat_genes = sorted(
            set().union(*(set(top_maps[s].keys()) for s in train_sids))
        )

        Xtr = np.vstack([
            np.log1p(
                np.clip(
                    vector_from_map(top_maps[s], train_feat_genes),
                    0, None
                )
            )
            for s in train_sids
        ])

        Xte = np.vstack([
            np.log1p(
                np.clip(
                    vector_from_map(top_maps[s], train_feat_genes),
                    0, None
                )
            )
            for s in test_sids
        ])

        # Standardize by training only.
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0, ddof=0)
        keep = sd > 1e-12
        if keep.sum() == 0:
            raise RuntimeError(f"{heldout}: no variable TopN features")

        Xtrz = (Xtr[:, keep] - mu[keep]) / sd[keep]
        Xtez = (Xte[:, keep] - mu[keep]) / sd[keep]

        # Distances for each test replicate.
        dist = np.sqrt(
            ((Xtez[:, None, :] - Xtrz[None, :, :]) ** 2).mean(axis=2)
        )

        for ti, sid in enumerate(test_sids):
            order = np.argsort(dist[ti])
            for k in KNN_KS:
                use = order[:k]
                nn_rows.append({
                    "heldout_condition": heldout,
                    "test_sample": sid,
                    "k": k,
                    "nearest_training_samples": ";".join(
                        train_sids[i] for i in use
                    ),
                    "nearest_distances": ";".join(
                        f"{dist[ti, i]:.6g}" for i in use
                    ),
                })

        for k in KNN_KS:
            pred_test = []
            for ti in range(len(test_sids)):
                order = np.argsort(dist[ti])[:k]
                pred_test.append(train_hidden[order].mean(axis=0))
            knn_state = np.vstack(pred_test).mean(axis=0)
            method_shifts[f"kNN{k}"] = knn_state - train_centroid

        # Metrics for each method.
        zrmse = float(np.sqrt(np.mean(true_shift**2)))
        zmae = float(np.mean(np.abs(true_shift)))

        for method, pred_shift in method_shifts.items():
            mm = metrics(true_shift, pred_shift)
            metric_rows.append({
                "heldout_condition": heldout,
                "method": method,
                "n_fixed_nonTopN_genes": N,
                **mm,
                "zero_shift_rmse": zrmse,
                "zero_shift_mae": zmae,
                "rmse_improvement_vs_zero": zrmse - mm["rmse"],
                "mae_improvement_vs_zero": zmae - mm["mae"],
            })

            # Top-K overlap for methods that define a magnitude ranking.
            for k0 in TOP_KS:
                k = min(k0, N)
                nov, frac, jac = topk_overlap(
                    np.abs(true_shift), np.abs(pred_shift), k
                )
                topk_rows.append({
                    "heldout_condition": heldout,
                    "method": method,
                    "K": k,
                    "n_overlap": nov,
                    "overlap_fraction_of_K": frac,
                    "jaccard": jac,
                    "uniform_random_expected_fraction": k / N,
                    "fold_enrichment_vs_uniform_random": (
                        frac / (k / N) if k > 0 else np.nan
                    ),
                })

        # Training-variability magnitude baseline.
        train_sd = train_hidden.std(axis=0, ddof=1)
        for k0 in TOP_KS:
            k = min(k0, N)
            nov, frac, jac = topk_overlap(
                np.abs(true_shift), train_sd, k
            )
            topk_rows.append({
                "heldout_condition": heldout,
                "method": "TrainingSD_rank",
                "K": k,
                "n_overlap": nov,
                "overlap_fraction_of_K": frac,
                "jaccard": jac,
                "uniform_random_expected_fraction": k / N,
                "fold_enrichment_vs_uniform_random": (
                    frac / (k / N) if k > 0 else np.nan
                ),
            })

    mdf = pd.DataFrame(metric_rows)
    kdf = pd.DataFrame(topk_rows)
    nndf = pd.DataFrame(nn_rows)

    mdf.to_csv(out / "centroid_shift_baseline_metrics_by_condition.csv", index=False)
    kdf.to_csv(out / "centroid_shift_topk_overlap_baselines.csv", index=False)
    nndf.to_csv(out / "knn_neighbors_by_test_sample.csv", index=False)

    summary_rows = []
    for method, sub in mdf.groupby("method", sort=False):
        for metric in [
            "pearson", "spearman", "rmse", "mae", "r2", "slope",
            "pred_true_sd_ratio", "sign_concordance",
            "rmse_improvement_vs_zero", "mae_improvement_vs_zero",
        ]:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            summary_rows.append({
                "method": method,
                "metric": metric,
                "n_conditions": len(vals),
                "mean": vals.mean() if len(vals) else np.nan,
                "median": vals.median() if len(vals) else np.nan,
                "q25": vals.quantile(0.25) if len(vals) else np.nan,
                "q75": vals.quantile(0.75) if len(vals) else np.nan,
            })

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out / "centroid_shift_baseline_metrics_summary.csv", index=False)

    topk_summary = (
        kdf.groupby(["method", "K"], as_index=False)
        .agg(
            n_conditions=("heldout_condition", "nunique"),
            overlap_fraction_mean=("overlap_fraction_of_K", "mean"),
            overlap_fraction_median=("overlap_fraction_of_K", "median"),
            fold_enrichment_mean=("fold_enrichment_vs_uniform_random", "mean"),
            fold_enrichment_median=("fold_enrichment_vs_uniform_random", "median"),
        )
    )
    topk_summary.to_csv(
        out / "centroid_shift_topk_overlap_baselines_summary.csv",
        index=False
    )

    print("[DONE] centroid-shift baseline comparison completed")
    print(f"[INFO] fixed non-TopN genes: {N}")
    print()
    print("=== SHIFT METRICS: MEDIAN ACROSS 8 CONDITIONS ===")
    pivot = sdf.pivot(index="method", columns="metric", values="median")
    cols = [
        "pearson", "spearman", "rmse", "mae", "r2", "slope",
        "pred_true_sd_ratio", "sign_concordance",
        "rmse_improvement_vs_zero", "mae_improvement_vs_zero",
    ]
    print(pivot.reindex(columns=cols).to_string())
    print()
    print("=== TOP-|SHIFT| OVERLAP BASELINES ===")
    print(topk_summary.to_string(index=False))
    print()
    print("=== PER-CONDITION CORE METRICS ===")
    print(mdf[[
        "heldout_condition","method","pearson","spearman","rmse","mae","r2",
        "slope","pred_true_sd_ratio","sign_concordance",
        "rmse_improvement_vs_zero","mae_improvement_vs_zero"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
