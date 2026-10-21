#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate whether Top2All adds sample-specific information beyond the
same-condition mean baseline.

For each held-out OOF sample:
  baseline = mean log1p(raw) of the other 2 replicates from the same condition

Define:
  true_residual = true - baseline
  pred_residual = Top2All - baseline

If Top2All uses sample-specific TopN information beyond the condition label,
pred_residual should track true_residual.

Primary outputs:
1) sample-wise residual correlations
2) gene-wise residual correlations across 24 OOF samples
3) gene-sample-pair error improvement:
       |true - baseline| - |true - Top2All|
   Positive => Top2All beats same-condition mean
4) improvement stratified by magnitude of the true residual (Q1-Q4)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return np.nan if den == 0 else float(np.sum(x*y) / den)


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


def parse_rep(sample_id):
    m = re.search(r"_rep(\d+)$", sample_id)
    if not m:
        raise ValueError(sample_id)
    return int(m.group(1))


def parse_sample_id(filename):
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(filename)
    return filename[:-len(suffix)]


def resolve_file(value, base):
    p = Path(str(value))
    if p.is_file():
        return p
    q = Path(base) / str(value)
    if q.is_file():
        return q
    raise FileNotFoundError(f"Cannot find {value}")


def load_full_exact(path, target_genes):
    df = pd.read_csv(path)
    if not {"Gene", "Expression"}.issubset(df.columns):
        raise ValueError(f"{path}: need Gene, Expression")
    # Match Top2All dict(zip(...)) semantics: last duplicate wins.
    mapping = dict(zip(df["Gene"].astype(str),
                       pd.to_numeric(df["Expression"], errors="coerce").fillna(0.0)))
    raw = np.array([float(mapping.get(g, 0.0)) for g in target_genes], dtype=float)
    if np.min(raw) < -1e-6:
        raise RuntimeError(f"Negative expression in {path}")
    return np.log1p(np.clip(raw, 0, None))


def load_top_genes(path, topn):
    df = pd.read_csv(path)
    if "Gene" not in df.columns:
        raise ValueError(f"{path}: need Gene")
    if len(df) < topn:
        raise RuntimeError(f"{path}: only {len(df)} rows")
    return df["Gene"].astype(str).iloc[:topn].tolist()


def qcut_labels(values, q=4):
    s = pd.Series(values, dtype=float)

    # Exact zero residuals are biologically uninformative for
    # "residual magnitude" stratification, so keep them separate.
    out = pd.Series(index=s.index, dtype="object")
    zero_mask = np.isclose(s, 0.0)
    out.loc[zero_mask] = "Zero"

    pos_mask = (~zero_mask) & s.notna()

    if pos_mask.sum() == 0:
        return out

    # Quartiles among non-zero residual magnitudes only.
    cats = pd.qcut(
        s.loc[pos_mask],
        q=q,
        duplicates="drop"
    )

    n_bins = len(cats.cat.categories)
    labels = [f"Q{i+1}" for i in range(n_bins)]

    out.loc[pos_mask] = pd.qcut(
        s.loc[pos_mask],
        q=q,
        labels=labels,
        duplicates="drop"
    ).astype(str)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_base_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    oof = Path(args.oof_base_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["rep"] = meta["sample_id"].map(parse_rep)

    topn_col = f"top{args.topn}_file"
    needed = {"sample_id", "cell_type", "full_file", topn_col}
    missing = needed - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing {sorted(missing)}")

    meta_idx = meta.set_index("sample_id", drop=False)

    pred_files = sorted(oof.rglob("*_pred_vs_true_v6_raw_final.csv"))
    if len(pred_files) != 24:
        raise RuntimeError(f"Expected 24 OOF predictions, found {len(pred_files)}")

    # Establish prediction gene universe.
    pred_by_sample = {}
    gene_order = None
    for p in pred_files:
        sid = parse_sample_id(p.name)
        df = pd.read_csv(p, usecols=["gene", "raw_pred_wo_overwrite"])
        genes = df["gene"].astype(str).tolist()
        if gene_order is None:
            gene_order = genes
        elif genes != gene_order:
            raise RuntimeError(f"Gene order mismatch: {p}")
        pred_by_sample[sid] = p

    if set(pred_by_sample) != set(meta["sample_id"]):
        raise RuntimeError("Prediction samples do not match metadata samples")

    # Fixed non-TopN genes.
    top_union = set()
    for _, r in meta.iterrows():
        top_path = resolve_file(r[topn_col], args.topn_dir)
        top_union.update(load_top_genes(top_path, args.topn))

    fixed_genes = [g for g in gene_order if g not in top_union]
    fixed_idx = np.array([i for i, g in enumerate(gene_order)
                          if g not in top_union], dtype=int)

    # Load truth and Top2All predictions.
    truth = {}
    pred = {}
    for _, r in meta.iterrows():
        sid = r["sample_id"]
        full_path = resolve_file(r["full_file"], args.full_dir)
        truth[sid] = load_full_exact(full_path, fixed_genes)

        pdf = pd.read_csv(pred_by_sample[sid],
                          usecols=["gene", "raw_pred_wo_overwrite"])
        raw_pred = pd.to_numeric(pdf["raw_pred_wo_overwrite"],
                                 errors="coerce").to_numpy(float)[fixed_idx]
        if not np.isfinite(raw_pred).all():
            raise RuntimeError(f"Nonfinite prediction: {sid}")
        pred[sid] = np.log1p(np.clip(raw_pred, 0, None))

    sample_rows = []
    pair_rows = []

    # Per-sample residual evaluation.
    for _, r in meta.sort_values(["cell_type", "rep"]).iterrows():
        sid = r["sample_id"]
        cond = r["cell_type"]
        rep = r["rep"]

        same_train = meta[(meta["cell_type"] == cond) & (meta["rep"] != rep)]
        if len(same_train) != 2:
            raise RuntimeError(f"{sid}: same-condition training n={len(same_train)}")

        baseline = np.vstack([truth[x] for x in same_train["sample_id"]]).mean(axis=0)

        t = truth[sid]
        p = pred[sid]
        true_resid = t - baseline
        pred_resid = p - baseline

        abs_err_baseline = np.abs(t - baseline)
        abs_err_top2all = np.abs(t - p)
        improvement = abs_err_baseline - abs_err_top2all

        sample_rows.append({
            "sample_id": sid,
            "cell_type": cond,
            "rep": rep,
            "n_genes": len(fixed_genes),
            "resid_pearson": pearson(true_resid, pred_resid),
            "resid_spearman": spearman(true_resid, pred_resid),
            "baseline_mae": float(abs_err_baseline.mean()),
            "top2all_mae": float(abs_err_top2all.mean()),
            "mean_mae_improvement": float(improvement.mean()),
            "median_mae_improvement": float(np.median(improvement)),
            "top2all_win_fraction": float(np.mean(improvement > 0)),
            "top2all_tie_fraction": float(np.mean(np.isclose(improvement, 0))),
        })

        for j, gene in enumerate(fixed_genes):
            pair_rows.append({
                "sample_id": sid,
                "cell_type": cond,
                "rep": rep,
                "gene": gene,
                "true_residual": true_resid[j],
                "pred_residual": pred_resid[j],
                "abs_true_residual": abs(true_resid[j]),
                "baseline_abs_error": abs_err_baseline[j],
                "top2all_abs_error": abs_err_top2all[j],
                "mae_improvement": improvement[j],
                "top2all_better": bool(improvement[j] > 0),
            })

    sdf = pd.DataFrame(sample_rows)
    pdf = pd.DataFrame(pair_rows)

    # Residual magnitude quartiles across all gene-sample pairs.
    pdf["residual_magnitude_quartile"] = qcut_labels(
        pdf["abs_true_residual"].to_numpy(), q=4
    )

    quartile_rows = []
    for q, sub in pdf.groupby("residual_magnitude_quartile", sort=True):
        quartile_rows.append({
            "quartile": q,
            "n_gene_sample_pairs": len(sub),
            "abs_true_residual_median": float(sub["abs_true_residual"].median()),
            "abs_true_residual_mean": float(sub["abs_true_residual"].mean()),
            "baseline_mae": float(sub["baseline_abs_error"].mean()),
            "top2all_mae": float(sub["top2all_abs_error"].mean()),
            "mean_mae_improvement": float(sub["mae_improvement"].mean()),
            "median_mae_improvement": float(sub["mae_improvement"].median()),
            "top2all_win_fraction": float(sub["top2all_better"].mean()),
            "resid_pearson_pooled": pearson(
                sub["true_residual"], sub["pred_residual"]
            ),
            "resid_spearman_pooled": spearman(
                sub["true_residual"], sub["pred_residual"]
            ),
        })
    qdf = pd.DataFrame(quartile_rows)

    # Gene-wise residual correlation across 24 OOF samples.
    gene_rows = []
    for gene, sub in pdf.groupby("gene", sort=False):
        tr = sub["true_residual"].to_numpy(float)
        pr = sub["pred_residual"].to_numpy(float)
        gene_rows.append({
            "gene": gene,
            "n_samples": len(sub),
            "true_residual_sd": float(np.std(tr, ddof=1)),
            "resid_pearson": pearson(tr, pr),
            "resid_spearman": spearman(tr, pr),
            "baseline_mae": float(sub["baseline_abs_error"].mean()),
            "top2all_mae": float(sub["top2all_abs_error"].mean()),
            "mean_mae_improvement": float(sub["mae_improvement"].mean()),
            "top2all_win_fraction": float(sub["top2all_better"].mean()),
        })
    gdf = pd.DataFrame(gene_rows)

    # Summary of variable genes.
    variable = gdf[gdf["true_residual_sd"] > 0].copy()

    summary = pd.DataFrame([{
        "fixed_nonTopN_genes": len(fixed_genes),
        "gene_sample_pairs": len(pdf),
        "sample_resid_pearson_median": float(sdf["resid_pearson"].median()),
        "sample_resid_spearman_median": float(sdf["resid_spearman"].median()),
        "overall_baseline_mae": float(pdf["baseline_abs_error"].mean()),
        "overall_top2all_mae": float(pdf["top2all_abs_error"].mean()),
        "overall_mean_mae_improvement": float(pdf["mae_improvement"].mean()),
        "overall_median_mae_improvement": float(pdf["mae_improvement"].median()),
        "overall_top2all_win_fraction": float(pdf["top2all_better"].mean()),
        "gene_resid_pearson_median": float(variable["resid_pearson"].median()),
        "gene_resid_spearman_median": float(variable["resid_spearman"].median()),
        "gene_mean_mae_improvement_median": float(
            variable["mean_mae_improvement"].median()
        ),
    }])

    sdf.to_csv(out / "samplewise_within_condition_residuals.csv", index=False)
    pdf.to_csv(out / "gene_sample_pair_within_condition_residuals.csv", index=False)
    qdf.to_csv(out / "residual_magnitude_quartile_summary.csv", index=False)
    gdf.to_csv(out / "genewise_within_condition_residuals.csv", index=False)
    summary.to_csv(out / "within_condition_residual_summary.csv", index=False)

    print("[DONE] within-condition residual evaluation")
    print()
    print("=== OVERALL SUMMARY ===")
    print(summary.to_string(index=False))
    print()
    print("=== SAMPLE-WISE RESIDUAL CORRELATION ===")
    print(sdf[[
        "sample_id","cell_type","rep","resid_pearson","resid_spearman",
        "baseline_mae","top2all_mae","mean_mae_improvement",
        "top2all_win_fraction"
    ]].to_string(index=False))
    print()
    print("=== BY TRUE RESIDUAL MAGNITUDE QUARTILE ===")
    print(qdf.to_string(index=False))
    print()
    print("=== GENE-WISE RESIDUAL CORRELATION SUMMARY ===")
    print(pd.DataFrame({
        "metric": ["pearson", "spearman", "mean_mae_improvement", "win_fraction"],
        "median": [
            variable["resid_pearson"].median(),
            variable["resid_spearman"].median(),
            variable["mean_mae_improvement"].median(),
            variable["top2all_win_fraction"].median(),
        ],
        "mean": [
            variable["resid_pearson"].mean(),
            variable["resid_spearman"].mean(),
            variable["mean_mae_improvement"].mean(),
            variable["top2all_win_fraction"].mean(),
        ],
    }).to_string(index=False))


if __name__ == "__main__":
    main()
