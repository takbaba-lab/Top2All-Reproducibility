#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gene-wise OOF evaluation for Top2All.

Primary purpose
---------------
Test whether Top2All reproduces sample-to-sample biological variation,
rather than merely the average zF transcriptome.

For each TopN setting:
  1) collect all 24 OOF prediction files,
  2) define a FIXED evaluation gene set:
       genes that are NEVER included in TopN in ANY of the 24 samples,
  3) for each fixed non-TopN gene, evaluate across all 24 OOF samples:
       Pearson, Spearman, RMSE, MAE, R2
     using:
       log1p(raw_true) vs log1p(raw_pred_wo_overwrite)

Why fixed non-TopN?
-------------------
A gene that is TopN in one sample but non-TopN in another has inconsistent
observability across samples. Excluding the UNION of TopN genes across all
24 samples gives a clean gene-wise benchmark in which every evaluated gene
was unobserved in every sample.

Outputs
-------
- oof_genewise_metrics_fixed_nonTopN_log1p.csv
- oof_genewise_summary_fixed_nonTopN_log1p.csv
- oof_genewise_summary_by_trueSD_quantile.csv
- oof_genewise_summary_by_trueMean_quantile.csv
- evaluation_run_info.txt
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["pearson", "spearman", "rmse", "mae", "r2"]


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom == 0:
        return np.nan
    return float(np.sum(x0 * y0) / denom)


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_corr(xr, yr)


def calc_metrics(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return {k: np.nan for k in METRICS}

    err = y - x
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    sse = float(np.sum((y - x) ** 2))
    sst = float(np.sum((x - x.mean()) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - (sse / sst)

    return {
        "pearson": pearson_corr(x, y),
        "spearman": spearman_corr(x, y),
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }


def choose_gene_column(df, path):
    for c in ["Gene", "gene", "GENE"]:
        if c in df.columns:
            return c
    raise ValueError(f"No gene column found in {path}; columns={df.columns.tolist()}")


def parse_sample_id(filename):
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {filename}")
    return filename[:-len(suffix)]


def parse_fold_dir(fold_dir):
    m = re.fullmatch(r"fold(\d+)_test_rep(\d+)", fold_dir.name)
    if not m:
        raise ValueError(f"Unexpected fold dir: {fold_dir.name}")
    return int(m.group(1)), int(m.group(2))


def parse_rep(sample_id):
    m = re.search(r"_rep(\d+)$", sample_id)
    if not m:
        raise ValueError(f"Cannot parse replicate from: {sample_id}")
    return int(m.group(1))


def summarize_metric_distribution(df, label):
    rows = []
    for metric in METRICS:
        v = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "set": label,
            "metric": metric,
            "n_genes_evaluable": int(v.size),
            "mean": float(np.mean(v)) if v.size else np.nan,
            "median": float(np.median(v)) if v.size else np.nan,
            "q25": float(np.quantile(v, 0.25)) if v.size else np.nan,
            "q75": float(np.quantile(v, 0.75)) if v.size else np.nan,
            "sd": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
            "min": float(np.min(v)) if v.size else np.nan,
            "max": float(np.max(v)) if v.size else np.nan,
        })
    return rows


def safe_qcut(series, q=4):
    """
    Quantile bins with duplicate-edge protection.
    Returns labels Q1...Qk where Q1 is lowest.
    """
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype="object")
    valid = s.notna()
    if valid.sum() == 0:
        return out
    try:
        cats = pd.qcut(s[valid], q=q, duplicates="drop")
        ncat = len(cats.cat.categories)
        labels = [f"Q{i+1}" for i in range(ncat)]
        out.loc[valid] = pd.qcut(
            s[valid], q=q, labels=labels, duplicates="drop"
        ).astype(str)
    except ValueError:
        out.loc[valid] = "Q1"
    return out


def make_binned_summary(df, bin_col, bin_value_col):
    rows = []
    for bin_name, sub in df.dropna(subset=[bin_col]).groupby(bin_col, sort=True):
        row = {
            bin_col: bin_name,
            "n_genes": int(len(sub)),
            f"{bin_value_col}_median": float(sub[bin_value_col].median()),
            f"{bin_value_col}_mean": float(sub[bin_value_col].mean()),
        }
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(vals))
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_base_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_n_samples", type=int, default=24)
    ap.add_argument("--negative_tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    oof_base = Path(args.oof_base_dir)
    metadata_csv = Path(args.metadata_csv)
    topn_dir = Path(args.topn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    required = {"sample_id", "cell_type"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    topn_col = f"top{args.topn}_file"
    if topn_col not in meta.columns:
        raise ValueError(
            f"metadata lacks column '{topn_col}'. "
            f"Available columns={meta.columns.tolist()}"
        )

    if meta["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in metadata")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta_idx = meta.set_index("sample_id", drop=False)

    fold_dirs = sorted(
        [p for p in oof_base.iterdir()
         if p.is_dir() and re.fullmatch(r"fold\d+_test_rep\d+", p.name)],
        key=lambda p: parse_fold_dir(p)[0],
    )
    if len(fold_dirs) != 3:
        raise RuntimeError(f"Expected 3 fold dirs, found {len(fold_dirs)}")

    sample_records = []
    seen = set()
    all_topn_union = set()
    reference_gene_order = None

    # First pass: validate files, collect topN union, and store prediction paths.
    for fold_dir in fold_dirs:
        fold, test_rep = parse_fold_dir(fold_dir)
        pred_files = sorted(
            (fold_dir / "prediction").rglob("*_pred_vs_true_v6_raw_final.csv")
        )
        if len(pred_files) != 8:
            raise RuntimeError(
                f"{fold_dir.name}: expected 8 prediction files, got {len(pred_files)}"
            )

        for pred_file in pred_files:
            sample_id = parse_sample_id(pred_file.name)
            if sample_id in seen:
                raise RuntimeError(f"Duplicate OOF prediction: {sample_id}")
            seen.add(sample_id)

            if sample_id not in meta_idx.index:
                raise KeyError(f"{sample_id} not found in metadata")

            if parse_rep(sample_id) != test_rep:
                raise RuntimeError(
                    f"OOF mismatch: {sample_id} under {fold_dir.name}"
                )

            mrow = meta_idx.loc[sample_id]
            topn_val = str(mrow[topn_col])
            topn_path = Path(topn_val)
            if not topn_path.is_file():
                topn_path = topn_dir / topn_val
            if not topn_path.is_file():
                raise FileNotFoundError(
                    f"TopN file not found for {sample_id}: {topn_val}"
                )

            top_df = pd.read_csv(topn_path)
            gene_col = choose_gene_column(top_df, topn_path)
            if len(top_df) < args.topn:
                raise RuntimeError(
                    f"{topn_path} has {len(top_df)} rows, less than Top{args.topn}"
                )

            top_genes = (
                top_df[gene_col].astype(str).iloc[:args.topn].tolist()
            )
            if len(set(top_genes)) != args.topn:
                raise RuntimeError(
                    f"TopN genes are not unique for {sample_id}"
                )
            all_topn_union.update(top_genes)

            pred_df = pd.read_csv(
                pred_file,
                usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
            )
            genes = pred_df["gene"].astype(str).tolist()
            if len(set(genes)) != len(genes):
                raise RuntimeError(f"Duplicate genes in {pred_file}")

            if reference_gene_order is None:
                reference_gene_order = genes
            elif genes != reference_gene_order:
                raise RuntimeError(
                    f"Gene order/universe differs across prediction files: {pred_file}"
                )

            sample_records.append({
                "fold": fold,
                "test_rep": test_rep,
                "sample_id": sample_id,
                "cell_type": str(mrow["cell_type"]),
                "prediction_file": pred_file,
                "topN_file": topn_path,
            })

    if len(sample_records) != args.expected_n_samples:
        raise RuntimeError(
            f"Expected {args.expected_n_samples} OOF samples, got {len(sample_records)}"
        )

    expected = set(meta["sample_id"])
    if seen != expected:
        raise RuntimeError(
            f"OOF sample set != metadata. "
            f"missing={sorted(expected-seen)}, unexpected={sorted(seen-expected)}"
        )

    gene_universe = set(reference_gene_order)
    topn_union_in_universe = all_topn_union & gene_universe
    fixed_genes = [g for g in reference_gene_order if g not in all_topn_union]

    if len(fixed_genes) == 0:
        raise RuntimeError("No fixed non-TopN genes remain")

    fixed_idx = np.array(
        [i for i, g in enumerate(reference_gene_order) if g not in all_topn_union],
        dtype=int,
    )

    # Stable sample order: condition then replicate is useful for matrix inspection.
    sample_records = sorted(
        sample_records, key=lambda r: (r["cell_type"], r["test_rep"], r["sample_id"])
    )

    n_samples = len(sample_records)
    n_genes = len(fixed_genes)
    true_mat = np.empty((n_samples, n_genes), dtype=np.float64)
    pred_mat = np.empty((n_samples, n_genes), dtype=np.float64)

    sample_info_rows = []

    for i, rec in enumerate(sample_records):
        df = pd.read_csv(
            rec["prediction_file"],
            usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
        )
        raw_true = pd.to_numeric(df["raw_true"], errors="coerce").to_numpy(float)[fixed_idx]
        raw_pred = pd.to_numeric(
            df["raw_pred_wo_overwrite"], errors="coerce"
        ).to_numpy(float)[fixed_idx]

        finite = np.isfinite(raw_true) & np.isfinite(raw_pred)
        if not finite.all():
            raise RuntimeError(
                f"Non-finite values found in fixed gene set for {rec['sample_id']}"
            )

        if (
            raw_true.min() < -args.negative_tolerance
            or raw_pred.min() < -args.negative_tolerance
        ):
            raise RuntimeError(
                f"Substantive negative raw values in {rec['sample_id']}"
            )

        raw_true = np.clip(raw_true, 0.0, None)
        raw_pred = np.clip(raw_pred, 0.0, None)

        true_mat[i, :] = np.log1p(raw_true)
        pred_mat[i, :] = np.log1p(raw_pred)

        sample_info_rows.append({
            "matrix_row": i,
            "sample_id": rec["sample_id"],
            "cell_type": rec["cell_type"],
            "fold": rec["fold"],
            "test_rep": rec["test_rep"],
        })

    # Gene-wise metrics across all 24 OOF samples.
    rows = []
    for j, gene in enumerate(fixed_genes):
        x = true_mat[:, j]
        y = pred_mat[:, j]
        metrics = calc_metrics(x, y)

        true_mean = float(np.mean(x))
        true_sd = float(np.std(x, ddof=1))
        pred_mean = float(np.mean(y))
        pred_sd = float(np.std(y, ddof=1))
        true_nonzero_fraction = float(np.mean(x > 0))
        pred_nonzero_fraction = float(np.mean(y > 0))

        rows.append({
            "gene": gene,
            "n_samples": n_samples,
            "true_mean_log1p": true_mean,
            "true_sd_log1p": true_sd,
            "pred_mean_log1p": pred_mean,
            "pred_sd_log1p": pred_sd,
            "true_nonzero_fraction": true_nonzero_fraction,
            "pred_nonzero_fraction": pred_nonzero_fraction,
            **metrics,
        })

    gdf = pd.DataFrame(rows)

    # Useful diagnostics: correlations are undefined for invariant truth genes.
    gdf["true_variable"] = gdf["true_sd_log1p"] > 0

    # Quantile bins among variable genes only.
    gdf["trueSD_quantile"] = np.nan
    var_mask = gdf["true_variable"]
    if var_mask.any():
        gdf.loc[var_mask, "trueSD_quantile"] = safe_qcut(
            gdf.loc[var_mask, "true_sd_log1p"], q=4
        ).values

    gdf["trueMean_quantile"] = safe_qcut(gdf["true_mean_log1p"], q=4).values

    metrics_out = out_dir / "oof_genewise_metrics_fixed_nonTopN_log1p.csv"
    gdf.to_csv(metrics_out, index=False)

    # Overall summaries:
    summary_rows = []
    summary_rows.extend(summarize_metric_distribution(gdf, "all_fixed_nonTopN"))
    summary_rows.extend(
        summarize_metric_distribution(gdf[gdf["true_variable"]], "true_variable_only")
    )
    summary = pd.DataFrame(summary_rows)
    summary_out = out_dir / "oof_genewise_summary_fixed_nonTopN_log1p.csv"
    summary.to_csv(summary_out, index=False)

    sd_summary = make_binned_summary(
        gdf[gdf["true_variable"]].copy(),
        "trueSD_quantile",
        "true_sd_log1p",
    )
    sd_out = out_dir / "oof_genewise_summary_by_trueSD_quantile.csv"
    sd_summary.to_csv(sd_out, index=False)

    mean_summary = make_binned_summary(
        gdf.copy(),
        "trueMean_quantile",
        "true_mean_log1p",
    )
    mean_out = out_dir / "oof_genewise_summary_by_trueMean_quantile.csv"
    mean_summary.to_csv(mean_out, index=False)

    sample_info = pd.DataFrame(sample_info_rows)
    sample_info_out = out_dir / "oof_matrix_sample_order.tsv"
    sample_info.to_csv(sample_info_out, sep="\t", index=False)

    info = []
    info.append(f"topN: {args.topn}")
    info.append(f"n_OOF_samples: {n_samples}")
    info.append(f"prediction_gene_universe: {len(reference_gene_order)}")
    info.append(f"TopN_union_total: {len(all_topn_union)}")
    info.append(f"TopN_union_in_prediction_universe: {len(topn_union_in_universe)}")
    info.append(f"fixed_nonTopN_genes: {len(fixed_genes)}")
    info.append(f"true_variable_fixed_genes: {int(gdf['true_variable'].sum())}")
    info.append(
        f"invariant_true_fixed_genes: {int((~gdf['true_variable']).sum())}"
    )
    info.append("")
    info.append("Primary comparison:")
    info.append("  log1p(raw_true) vs log1p(raw_pred_wo_overwrite)")
    info.append("  each gene evaluated across all 24 held-out OOF samples")
    info.append("  fixed gene set = prediction universe minus union of all TopN genes")
    info.append("")
    info.append("Outputs:")
    for p in [metrics_out, summary_out, sd_out, mean_out, sample_info_out]:
        info.append(f"  {p}")

    info_out = out_dir / "evaluation_run_info.txt"
    info_out.write_text("\n".join(info) + "\n", encoding="utf-8")

    print("[DONE] gene-wise OOF evaluation completed")
    print(f"[INFO] n OOF samples              : {n_samples}")
    print(f"[INFO] prediction gene universe   : {len(reference_gene_order)}")
    print(f"[INFO] TopN union in gene universe: {len(topn_union_in_universe)}")
    print(f"[INFO] fixed non-TopN genes       : {len(fixed_genes)}")
    print(f"[INFO] variable fixed genes       : {int(gdf['true_variable'].sum())}")
    print()
    print("Gene-wise metric summary:")
    print(summary.to_string(index=False))
    print()
    print("By true-expression SD quartile:")
    print(sd_summary.to_string(index=False))


if __name__ == "__main__":
    main()
