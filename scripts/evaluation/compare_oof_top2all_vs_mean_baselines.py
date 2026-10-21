#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Top2All OOF predictions against two strict OOF mean baselines.

Methods
-------
1) Top2All
   log1p(raw_pred_wo_overwrite)

2) GlobalMean
   For each held-out replicate/fold, mean log1p(raw expression) across ALL
   16 training samples (the other two replicates from each of 8 conditions).

3) SameConditionMean
   For each held-out sample, mean log1p(raw expression) across the TWO
   training samples from the SAME biological condition.

Evaluation gene set
-------------------
Fixed non-TopN genes:
prediction gene universe MINUS the UNION of TopN genes across all 24 samples.

Thus every evaluated gene was absent from TopN in every sample.

Metrics
-------
Sample-wise:
  Pearson, Spearman, RMSE, MAE, R2 across fixed non-TopN genes.

Gene-wise:
  Pearson, Spearman, RMSE, MAE, R2 across the 24 OOF samples for each gene.

Important fairness choices
--------------------------
- All baselines are strictly OOF: held-out replicate is never used to form
  its baseline prediction.
- Means are calculated in log1p(raw) space, because that is the primary
  evaluation space and gives the strongest MSE-optimal naive mean baseline.
- Top2All uses raw_pred_wo_overwrite, never raw_pred_final.

Expected metadata columns
-------------------------
sample_id, cell_type, full_file, top{N}_file

Expected full/topN CSV columns
------------------------------
Gene, Expression
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ["Top2All", "GlobalMean", "SameConditionMean"]
METRICS = ["pearson", "spearman", "rmse", "mae", "r2"]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
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
    den = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if den == 0:
        return np.nan
    return float(np.sum(x0 * y0) / den)


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


def calc_metrics(true, pred):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(true) & np.isfinite(pred)
    true = true[m]
    pred = pred[m]

    if true.size < 2:
        return {k: np.nan for k in METRICS}

    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    sse = float(np.sum(err ** 2))
    sst = float(np.sum((true - true.mean()) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - sse / sst

    return {
        "pearson": pearson_corr(true, pred),
        "spearman": spearman_corr(true, pred),
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }


# ---------------------------------------------------------------------
# Parsing / loading
# ---------------------------------------------------------------------
def parse_rep(sample_id):
    m = re.search(r"_rep(\d+)$", str(sample_id))
    if not m:
        raise ValueError(f"Cannot parse replicate from sample_id: {sample_id}")
    return int(m.group(1))


def parse_fold_dir(path):
    m = re.fullmatch(r"fold(\d+)_test_rep(\d+)", path.name)
    if not m:
        raise ValueError(f"Unexpected fold directory: {path.name}")
    return int(m.group(1)), int(m.group(2))


def prediction_sample_id(filename):
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {filename}")
    return filename[:-len(suffix)]


def require_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns {sorted(missing)}; "
            f"columns={df.columns.tolist()}"
        )


def resolve_file(value, base_dir):
    p = Path(str(value))
    if p.is_file():
        return p
    q = Path(base_dir) / str(value)
    if q.is_file():
        return q
    raise FileNotFoundError(f"Cannot find file: {value} (base={base_dir})")


def load_gene_expression(path):
    df = pd.read_csv(path)
    require_columns(df, ["Gene", "Expression"], path)

    df = df[["Gene", "Expression"]].copy()
    df["Gene"] = df["Gene"].astype(str)
    df["Expression"] = pd.to_numeric(
        df["Expression"], errors="coerce"
    ).fillna(0.0)

    if (df["Expression"] < -1e-6).any():
        raise RuntimeError(f"Negative raw Expression values in {path}")

    df["Expression"] = df["Expression"].clip(lower=0.0)

    # IMPORTANT:
    # Match the preprocessing used in train_log1p.py:
    # exp_map = dict(zip(Gene, Expression))
    # Therefore, for duplicated Gene symbols, keep the LAST occurrence.
    n_dup = int(df["Gene"].duplicated(keep="last").sum())
    if n_dup > 0:
        print(
            f"[INFO] {path}: {n_dup} duplicated Gene rows; "
            "keeping last occurrence to match Top2All training preprocessing."
        )

    df = df.drop_duplicates(subset="Gene", keep="last")

    genes = df["Gene"].tolist()
    expr = df["Expression"].to_numpy(dtype=np.float64)

    return genes, expr


def load_top_genes(path, topn):
    df = pd.read_csv(path)
    require_columns(df, ["Gene"], path)
    if len(df) < topn:
        raise RuntimeError(f"{path} has only {len(df)} rows; Top{topn} requested")
    genes = df["Gene"].astype(str).iloc[:topn].tolist()
    if len(set(genes)) != topn:
        raise RuntimeError(f"First Top{topn} genes are not unique in {path}")
    return genes


def vector_aligned_to_gene_order(path, target_genes):
    genes, expr = load_gene_expression(path)
    mapping = dict(zip(genes, expr))
    missing = [g for g in target_genes if g not in mapping]
    if missing:
        raise RuntimeError(
            f"{path}: {len(missing)} target genes missing; first={missing[:10]}"
        )
    return np.array([mapping[g] for g in target_genes], dtype=np.float64)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------
def summarize_samplewise(df):
    rows = []
    for method, sub in df.groupby("method", sort=False):
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
            rows.append({
                "method": method,
                "metric": metric,
                "n_samples": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "q25": float(np.quantile(vals, 0.25)) if vals.size else np.nan,
                "q75": float(np.quantile(vals, 0.75)) if vals.size else np.nan,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_genewise(df, set_label):
    rows = []
    for method, sub in df.groupby("method", sort=False):
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
            rows.append({
                "set": set_label,
                "method": method,
                "metric": metric,
                "n_genes_evaluable": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "q25": float(np.quantile(vals, 0.25)) if vals.size else np.nan,
                "q75": float(np.quantile(vals, 0.75)) if vals.size else np.nan,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            })
    return rows


def safe_qcut(series, q=4):
    out = pd.Series(index=series.index, dtype="object")
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna()
    if valid.sum() == 0:
        return out
    cats = pd.qcut(s[valid], q=q, duplicates="drop")
    ncat = len(cats.cat.categories)
    labels = [f"Q{i+1}" for i in range(ncat)]
    out.loc[valid] = pd.qcut(
        s[valid], q=q, labels=labels, duplicates="drop"
    ).astype(str)
    return out


def summarize_genewise_by_sd(df):
    rows = []
    for (method, q), sub in df.dropna(subset=["trueSD_quantile"]).groupby(
        ["method", "trueSD_quantile"], sort=True
    ):
        row = {
            "method": method,
            "trueSD_quantile": q,
            "n_genes": int(len(sub)),
            "true_sd_log1p_median": float(sub["true_sd_log1p"].median()),
            "true_sd_log1p_mean": float(sub["true_sd_log1p"].mean()),
        }
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(vals))
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_improvement_table(sample_df):
    """
    Positive means Top2All is better.
    For correlation/R2: Top2All - baseline.
    For RMSE/MAE: baseline - Top2All.
    """
    key = ["fold", "test_rep", "sample_id", "cell_type"]
    wide = sample_df.pivot_table(
        index=key, columns="method", values=METRICS, aggfunc="first"
    )

    rows = []
    for idx in wide.index:
        base_info = dict(zip(key, idx if isinstance(idx, tuple) else [idx]))
        for baseline in ["GlobalMean", "SameConditionMean"]:
            row = {**base_info, "baseline": baseline}
            for metric in METRICS:
                top = wide.loc[idx, (metric, "Top2All")]
                base = wide.loc[idx, (metric, baseline)]
                if metric in ["rmse", "mae"]:
                    improvement = base - top
                else:
                    improvement = top - base
                row[f"{metric}_improvement"] = float(improvement)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_base_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_n_samples", type=int, default=24)
    args = ap.parse_args()

    oof_base = Path(args.oof_base_dir)
    metadata_csv = Path(args.metadata_csv)
    full_dir = Path(args.full_dir)
    topn_dir = Path(args.topn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    needed_meta = {"sample_id", "cell_type", "full_file", f"top{args.topn}_file"}
    missing = needed_meta - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["rep"] = meta["sample_id"].map(parse_rep)

    if len(meta) != args.expected_n_samples:
        raise RuntimeError(
            f"Expected {args.expected_n_samples} metadata rows, got {len(meta)}"
        )
    if meta["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id in metadata")

    meta_idx = meta.set_index("sample_id", drop=False)
    topn_col = f"top{args.topn}_file"

    # -------------------------------------------------------------
    # Locate all OOF prediction files and establish gene universe.
    # -------------------------------------------------------------
    fold_dirs = sorted(
        [
            p for p in oof_base.iterdir()
            if p.is_dir() and re.fullmatch(r"fold\d+_test_rep\d+", p.name)
        ],
        key=lambda p: parse_fold_dir(p)[0],
    )
    if len(fold_dirs) != 3:
        raise RuntimeError(f"Expected 3 fold directories, found {len(fold_dirs)}")

    pred_records = {}
    reference_gene_order = None

    for fold_dir in fold_dirs:
        fold, test_rep = parse_fold_dir(fold_dir)
        files = sorted(
            (fold_dir / "prediction").rglob("*_pred_vs_true_v6_raw_final.csv")
        )
        if len(files) != 8:
            raise RuntimeError(
                f"{fold_dir.name}: expected 8 prediction files, got {len(files)}"
            )

        for p in files:
            sample_id = prediction_sample_id(p.name)
            if sample_id in pred_records:
                raise RuntimeError(f"Duplicate OOF prediction: {sample_id}")
            if sample_id not in meta_idx.index:
                raise KeyError(f"Prediction sample not in metadata: {sample_id}")
            if parse_rep(sample_id) != test_rep:
                raise RuntimeError(
                    f"{sample_id} is not rep{test_rep} but is under {fold_dir.name}"
                )

            pdf = pd.read_csv(
                p, usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
            )
            genes = pdf["gene"].astype(str).tolist()
            if len(genes) != len(set(genes)):
                raise RuntimeError(f"Duplicate genes in {p}")

            if reference_gene_order is None:
                reference_gene_order = genes
            elif genes != reference_gene_order:
                raise RuntimeError(f"Gene universe/order differs: {p}")

            pred_records[sample_id] = {
                "fold": fold,
                "test_rep": test_rep,
                "path": p,
            }

    if set(pred_records) != set(meta["sample_id"]):
        missing = sorted(set(meta["sample_id"]) - set(pred_records))
        extra = sorted(set(pred_records) - set(meta["sample_id"]))
        raise RuntimeError(f"OOF sample mismatch: missing={missing}, extra={extra}")

    # -------------------------------------------------------------
    # Fixed non-TopN gene set = gene universe - union TopN across all 24.
    # -------------------------------------------------------------
    topn_union = set()
    for _, row in meta.iterrows():
        topn_path = resolve_file(row[topn_col], topn_dir)
        topn_union.update(load_top_genes(topn_path, args.topn))

    fixed_genes = [g for g in reference_gene_order if g not in topn_union]
    fixed_index = np.array(
        [i for i, g in enumerate(reference_gene_order) if g not in topn_union],
        dtype=int,
    )
    if len(fixed_genes) == 0:
        raise RuntimeError("No fixed non-TopN genes remain")

    # -------------------------------------------------------------
    # Load full truth for ALL 24 samples, aligned to fixed genes.
    # Baseline means are computed in log1p space.
    # -------------------------------------------------------------
    truth = {}
    for _, row in meta.iterrows():
        full_path = resolve_file(row["full_file"], full_dir)
        raw = vector_aligned_to_gene_order(full_path, fixed_genes)
        truth[row["sample_id"]] = np.log1p(raw)

    # -------------------------------------------------------------
    # Build 24 OOF predictions per method.
    # -------------------------------------------------------------
    sample_rows = []
    method_pred = {
        method: {} for method in METHODS
    }

    for _, row in meta.sort_values(["rep", "cell_type", "sample_id"]).iterrows():
        sample_id = row["sample_id"]
        condition = row["cell_type"]
        test_rep = int(row["rep"])
        fold = int(pred_records[sample_id]["fold"])

        train_meta = meta[meta["rep"] != test_rep].copy()
        if len(train_meta) != 16:
            raise RuntimeError(
                f"{sample_id}: expected 16 training samples, got {len(train_meta)}"
            )

        same_meta = train_meta[train_meta["cell_type"] == condition].copy()
        if len(same_meta) != 2:
            raise RuntimeError(
                f"{sample_id}: expected 2 same-condition training samples, "
                f"got {len(same_meta)}"
            )

        true_vec = truth[sample_id]

        global_mat = np.vstack(
            [truth[sid] for sid in train_meta["sample_id"]]
        )
        same_mat = np.vstack(
            [truth[sid] for sid in same_meta["sample_id"]]
        )

        global_pred = global_mat.mean(axis=0)
        same_pred = same_mat.mean(axis=0)

        pred_path = pred_records[sample_id]["path"]
        pdf = pd.read_csv(
            pred_path, usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
        )

        genes = pdf["gene"].astype(str).tolist()
        if genes != reference_gene_order:
            raise RuntimeError(f"Gene order changed unexpectedly: {pred_path}")

        pred_raw = pd.to_numeric(
            pdf["raw_pred_wo_overwrite"], errors="coerce"
        ).to_numpy(np.float64)[fixed_index]
        pred_true_raw = pd.to_numeric(
            pdf["raw_true"], errors="coerce"
        ).to_numpy(np.float64)[fixed_index]

        if not np.isfinite(pred_raw).all() or not np.isfinite(pred_true_raw).all():
            raise RuntimeError(f"Non-finite values in prediction CSV: {pred_path}")
        if pred_raw.min() < -1e-6 or pred_true_raw.min() < -1e-6:
            raise RuntimeError(f"Negative raw values in prediction CSV: {pred_path}")

        top2all_pred = np.log1p(np.clip(pred_raw, 0.0, None))
        pred_csv_truth = np.log1p(np.clip(pred_true_raw, 0.0, None))

        # Verify prediction-file truth matches the full-file truth.
        max_abs_truth_diff = float(np.max(np.abs(pred_csv_truth - true_vec)))
        if max_abs_truth_diff > 1e-5:
            raise RuntimeError(
                f"{sample_id}: raw_true in prediction file disagrees with full_file "
                f"(max abs log1p diff={max_abs_truth_diff:.6g})"
            )

        predictions = {
            "Top2All": top2all_pred,
            "GlobalMean": global_pred,
            "SameConditionMean": same_pred,
        }

        for method in METHODS:
            vec = predictions[method]
            method_pred[method][sample_id] = vec
            met = calc_metrics(true_vec, vec)
            sample_rows.append({
                "fold": fold,
                "test_rep": test_rep,
                "sample_id": sample_id,
                "cell_type": condition,
                "method": method,
                "n_genes_fixed_nonTopN": len(fixed_genes),
                **met,
            })

    sample_df = pd.DataFrame(sample_rows)
    sample_out = out_dir / "baseline_comparison_samplewise_fixed_nonTopN_log1p.csv"
    sample_df.to_csv(sample_out, index=False)

    sample_summary = summarize_samplewise(sample_df)
    sample_summary_out = (
        out_dir / "baseline_comparison_samplewise_summary_fixed_nonTopN_log1p.csv"
    )
    sample_summary.to_csv(sample_summary_out, index=False)

    paired = paired_improvement_table(sample_df)
    paired_out = out_dir / "top2all_paired_improvement_vs_baselines_samplewise.csv"
    paired.to_csv(paired_out, index=False)

    # -------------------------------------------------------------
    # Gene-wise metrics across all 24 OOF samples.
    # -------------------------------------------------------------
    sample_order = meta.sort_values(
        ["cell_type", "rep", "sample_id"]
    )["sample_id"].tolist()

    true_mat = np.vstack([truth[sid] for sid in sample_order])
    method_mats = {
        method: np.vstack([method_pred[method][sid] for sid in sample_order])
        for method in METHODS
    }

    true_mean = np.mean(true_mat, axis=0)
    true_sd = np.std(true_mat, axis=0, ddof=1)
    true_variable = true_sd > 0

    sd_quantile = pd.Series(index=np.arange(len(fixed_genes)), dtype="object")
    if np.any(true_variable):
        tmp = pd.Series(true_sd[true_variable])
        qlabels = safe_qcut(tmp, q=4).to_numpy()
        sd_quantile.loc[np.where(true_variable)[0]] = qlabels

    gene_rows = []
    for method in METHODS:
        pred_mat = method_mats[method]
        for j, gene in enumerate(fixed_genes):
            met = calc_metrics(true_mat[:, j], pred_mat[:, j])
            gene_rows.append({
                "gene": gene,
                "method": method,
                "n_samples": len(sample_order),
                "true_mean_log1p": float(true_mean[j]),
                "true_sd_log1p": float(true_sd[j]),
                "true_variable": bool(true_variable[j]),
                "trueSD_quantile": sd_quantile.iloc[j],
                **met,
            })

    gene_df = pd.DataFrame(gene_rows)
    gene_out = out_dir / "baseline_comparison_genewise_fixed_nonTopN_log1p.csv"
    gene_df.to_csv(gene_out, index=False)

    gene_summary_rows = []
    gene_summary_rows.extend(
        summarize_genewise(gene_df, "all_fixed_nonTopN")
    )
    gene_summary_rows.extend(
        summarize_genewise(
            gene_df[gene_df["true_variable"]].copy(),
            "true_variable_only",
        )
    )
    gene_summary = pd.DataFrame(gene_summary_rows)
    gene_summary_out = (
        out_dir / "baseline_comparison_genewise_summary_fixed_nonTopN_log1p.csv"
    )
    gene_summary.to_csv(gene_summary_out, index=False)

    gene_sd_summary = summarize_genewise_by_sd(
        gene_df[gene_df["true_variable"]].copy()
    )
    gene_sd_out = (
        out_dir / "baseline_comparison_genewise_by_trueSD_quartile.csv"
    )
    gene_sd_summary.to_csv(gene_sd_out, index=False)

    # -------------------------------------------------------------
    # Run info
    # -------------------------------------------------------------
    info = []
    info.append(f"topN: {args.topn}")
    info.append(f"OOF samples: {len(meta)}")
    info.append(f"prediction gene universe: {len(reference_gene_order)}")
    info.append(f"TopN union in prediction universe: {len(set(reference_gene_order) & topn_union)}")
    info.append(f"fixed non-TopN genes: {len(fixed_genes)}")
    info.append(f"variable fixed genes: {int(np.sum(true_variable))}")
    info.append("")
    info.append("Methods:")
    info.append("  Top2All = log1p(raw_pred_wo_overwrite)")
    info.append("  GlobalMean = mean training truth in log1p(raw) space (16 samples)")
    info.append("  SameConditionMean = mean same-condition training truth in log1p(raw) space (2 samples)")
    info.append("")
    info.append("Held-out replicate is excluded from ALL baseline construction.")
    info.append("")
    info.append("Outputs:")
    for p in [
        sample_out,
        sample_summary_out,
        paired_out,
        gene_out,
        gene_summary_out,
        gene_sd_out,
    ]:
        info.append(f"  {p}")

    info_out = out_dir / "evaluation_run_info.txt"
    info_out.write_text("\n".join(info) + "\n", encoding="utf-8")

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print("[DONE] OOF baseline comparison completed")
    print(f"[INFO] fixed non-TopN genes : {len(fixed_genes)}")
    print(f"[INFO] variable fixed genes : {int(np.sum(true_variable))}")
    print()
    print("=== SAMPLE-WISE SUMMARY ===")
    print(sample_summary.to_string(index=False))
    print()
    print("=== GENE-WISE SUMMARY: TRUE-VARIABLE GENES ONLY ===")
    print(
        gene_summary[
            gene_summary["set"] == "true_variable_only"
        ].to_string(index=False)
    )
    print()
    print("=== GENE-WISE BY TRUE-SD QUARTILE ===")
    print(gene_sd_summary.to_string(index=False))
    print()
    print("=== MEDIAN PAIRED IMPROVEMENT OF TOP2ALL VS BASELINES ===")
    imp_cols = [c for c in paired.columns if c.endswith("_improvement")]
    print(
        paired.groupby("baseline")[imp_cols]
        .median()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
