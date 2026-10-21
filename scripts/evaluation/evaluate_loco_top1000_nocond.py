#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate condition-free Top2All under Leave-One-Condition-Out (LOCO).

Expected layout
---------------
<loco_base_dir>/
  test_Control_female/prediction/*_pred_vs_true_v6_raw_final.csv
  test_Control_male/prediction/*
  ...
  test_Cas_E2/prediction/*

Each test_<condition> directory must contain exactly the 3 biological
replicates of that held-out condition.

Primary evaluation
------------------
1) Truth QC:
   prediction CSV raw_true vs the sample's current full.csv.

2) Fixed non-TopN gene set:
   prediction gene universe MINUS the UNION of TopN genes across ALL 24 samples.
   Thus each evaluated gene is unobserved in TopN for every sample.

3) Sample-wise metrics:
   log1p(raw_true) vs log1p(raw_pred_wo_overwrite)
   over fixed non-TopN genes.

4) Gene-wise metrics:
   for each fixed non-TopN gene, evaluate across all 24 LOCO predictions.

5) Condition-wise summaries:
   summarize the 3 held-out replicates for each unseen biological condition.

6) Fair LOCO baseline:
   GlobalTrainMean = mean log1p(raw) over the 21 training samples from the
   seven OTHER conditions. The held-out condition is never used.

Important:
- SameConditionMean is intentionally NOT used as a LOCO baseline because
  no sample from the held-out condition is available to the model/training set.
- Top2All uses raw_pred_wo_overwrite, never raw_pred_final.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["pearson", "spearman", "rmse", "mae", "r2"]
METHODS = ["Top2All_LOCO", "GlobalTrainMean_LOCO"]


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
# Loading helpers
# ---------------------------------------------------------------------
def parse_sample_id(filename):
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {filename}")
    return filename[:-len(suffix)]


def parse_rep(sample_id):
    m = re.search(r"_rep(\d+)$", str(sample_id))
    if not m:
        raise ValueError(f"Cannot parse replicate: {sample_id}")
    return int(m.group(1))


def resolve_file(value, base_dir):
    p = Path(str(value))
    if p.is_file():
        return p
    q = Path(base_dir) / str(value)
    if q.is_file():
        return q
    raise FileNotFoundError(f"Cannot find file: {value} (base={base_dir})")


def require_columns(df, cols, path):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns {sorted(missing)}; "
            f"available={df.columns.tolist()}"
        )


def load_full_exact(path, target_genes):
    """
    Match Top2All's dict(zip(Gene, Expression)) semantics:
    if a Gene appears more than once, the LAST occurrence wins.
    """
    df = pd.read_csv(path)
    require_columns(df, ["Gene", "Expression"], path)

    genes = df["Gene"].astype(str)
    expr = pd.to_numeric(df["Expression"], errors="coerce").fillna(0.0)
    mapping = dict(zip(genes, expr))

    raw = np.array([float(mapping.get(g, 0.0)) for g in target_genes], dtype=float)
    if not np.isfinite(raw).all():
        raise RuntimeError(f"Non-finite full expression in {path}")
    if raw.min() < -1e-6:
        raise RuntimeError(f"Negative full expression in {path}")

    return np.clip(raw, 0.0, None)


def load_top_genes(path, topn):
    df = pd.read_csv(path)
    require_columns(df, ["Gene"], path)
    if len(df) < topn:
        raise RuntimeError(f"{path}: only {len(df)} rows, requested Top{topn}")
    genes = df["Gene"].astype(str).iloc[:topn].tolist()
    return genes


def safe_qcut(series, q=4):
    """
    Return object-dtype Q labels, avoiding the pandas FutureWarning and
    handling duplicate quantile edges.
    """
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=series.index, dtype="object")
    valid = s.notna()
    if valid.sum() == 0:
        return out

    cats = pd.qcut(s.loc[valid], q=q, duplicates="drop")
    ncat = len(cats.cat.categories)
    labels = [f"Q{i+1}" for i in range(ncat)]

    out.loc[valid] = pd.qcut(
        s.loc[valid], q=q, labels=labels, duplicates="drop"
    ).astype(str)

    return out


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------
def summarize_metrics(df, group_cols, n_label):
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, sort=False)

    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))

        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(float)
            rows.append({
                **base,
                "metric": metric,
                n_label: int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "q25": float(np.quantile(vals, 0.25)) if vals.size else np.nan,
                "q75": float(np.quantile(vals, 0.75)) if vals.size else np.nan,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            })
    return pd.DataFrame(rows)


def summarize_genewise_by_sd(gdf):
    rows = []
    variable = gdf[gdf["true_variable"]].copy()

    for (method, q), sub in variable.dropna(
        subset=["trueSD_quantile"]
    ).groupby(["method", "trueSD_quantile"], sort=True):

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco_base_dir", required=True)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--truth_log1p_tolerance", type=float, default=1e-5)
    args = ap.parse_args()

    loco_base = Path(args.loco_base_dir)
    full_dir = Path(args.full_dir)
    topn_dir = Path(args.topn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    needed = {"sample_id", "cell_type", "full_file", f"top{args.topn}_file"}
    missing = needed - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["cell_type"] = meta["cell_type"].astype(str)
    meta["rep"] = meta["sample_id"].map(parse_rep)

    if len(meta) != 24:
        raise RuntimeError(f"Expected 24 metadata samples, found {len(meta)}")
    if meta["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id in metadata")

    conditions = sorted(meta["cell_type"].unique().tolist())
    if len(conditions) != 8:
        raise RuntimeError(
            f"Expected 8 conditions, found {len(conditions)}: {conditions}"
        )

    meta_idx = meta.set_index("sample_id", drop=False)
    topn_col = f"top{args.topn}_file"

    # -------------------------------------------------------------
    # Discover the 8 LOCO prediction sets.
    # -------------------------------------------------------------
    pred_records = {}
    gene_order = None

    for cond in conditions:
        test_dir = loco_base / f"test_{cond}"
        pred_dir = test_dir / "prediction"
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing LOCO prediction dir: {pred_dir}")

        files = sorted(pred_dir.rglob("*_pred_vs_true_v6_raw_final.csv"))
        if len(files) != 3:
            raise RuntimeError(
                f"{cond}: expected 3 prediction files, found {len(files)}"
            )

        for p in files:
            sid = parse_sample_id(p.name)

            if sid not in meta_idx.index:
                raise KeyError(f"Prediction sample not in metadata: {sid}")

            true_cond = str(meta_idx.loc[sid, "cell_type"])
            if true_cond != cond:
                raise RuntimeError(
                    f"{p}: directory says heldout={cond}, "
                    f"but metadata says cell_type={true_cond}"
                )

            if sid in pred_records:
                raise RuntimeError(f"Duplicate LOCO prediction for {sid}")

            pdf = pd.read_csv(
                p, usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
            )
            genes = pdf["gene"].astype(str).tolist()

            if len(genes) != len(set(genes)):
                raise RuntimeError(f"Duplicate genes in prediction file {p}")

            if gene_order is None:
                gene_order = genes
            elif genes != gene_order:
                raise RuntimeError(f"Prediction gene order differs: {p}")

            pred_records[sid] = {
                "heldout_condition": cond,
                "path": p,
            }

    if set(pred_records) != set(meta["sample_id"]):
        missing = sorted(set(meta["sample_id"]) - set(pred_records))
        extra = sorted(set(pred_records) - set(meta["sample_id"]))
        raise RuntimeError(
            f"LOCO prediction sample mismatch: missing={missing}, extra={extra}"
        )

    # -------------------------------------------------------------
    # Fixed non-TopN gene set.
    # -------------------------------------------------------------
    topn_union = set()

    for _, row in meta.iterrows():
        path = resolve_file(row[topn_col], topn_dir)
        topn_union.update(load_top_genes(path, args.topn))

    fixed_genes = [g for g in gene_order if g not in topn_union]
    fixed_idx = np.array(
        [i for i, g in enumerate(gene_order) if g not in topn_union],
        dtype=int,
    )

    if len(fixed_genes) == 0:
        raise RuntimeError("No fixed non-TopN genes remain")

    # -------------------------------------------------------------
    # Load current full truth for all 24 samples.
    # -------------------------------------------------------------
    truth_raw = {}
    truth_log = {}

    for _, row in meta.iterrows():
        sid = row["sample_id"]
        full_path = resolve_file(row["full_file"], full_dir)
        raw = load_full_exact(full_path, fixed_genes)
        truth_raw[sid] = raw
        truth_log[sid] = np.log1p(raw)

    # -------------------------------------------------------------
    # Truth QC + Top2All predictions + LOCO global-mean baseline.
    # -------------------------------------------------------------
    qc_rows = []
    sample_rows = []
    prediction_vectors = {m: {} for m in METHODS}

    for _, row in meta.sort_values(
        ["cell_type", "rep", "sample_id"]
    ).iterrows():

        sid = row["sample_id"]
        cond = row["cell_type"]
        rep = int(row["rep"])
        pred_path = pred_records[sid]["path"]

        pdf = pd.read_csv(
            pred_path, usecols=["gene", "raw_true", "raw_pred_wo_overwrite"]
        )

        genes = pdf["gene"].astype(str).tolist()
        if genes != gene_order:
            raise RuntimeError(f"Gene order changed unexpectedly: {pred_path}")

        raw_true_csv = pd.to_numeric(
            pdf["raw_true"], errors="coerce"
        ).to_numpy(float)[fixed_idx]
        raw_pred = pd.to_numeric(
            pdf["raw_pred_wo_overwrite"], errors="coerce"
        ).to_numpy(float)[fixed_idx]

        if not np.isfinite(raw_true_csv).all():
            raise RuntimeError(f"Non-finite raw_true in {pred_path}")
        if not np.isfinite(raw_pred).all():
            raise RuntimeError(f"Non-finite raw_pred_wo_overwrite in {pred_path}")

        raw_true_csv = np.clip(raw_true_csv, 0.0, None)
        raw_pred = np.clip(raw_pred, 0.0, None)

        true_vec = truth_log[sid]
        csv_true_log = np.log1p(raw_true_csv)
        top2all_pred = np.log1p(raw_pred)

        max_log_diff = float(np.max(np.abs(csv_true_log - true_vec)))
        rmse_truth_qc = float(
            np.sqrt(np.mean((csv_true_log - true_vec) ** 2))
        )

        qc_rows.append({
            "sample_id": sid,
            "heldout_condition": cond,
            "rep": rep,
            "prediction_file": str(pred_path),
            "n_fixed_nonTopN_genes": len(fixed_genes),
            "max_abs_log1p_truth_diff": max_log_diff,
            "rmse_log1p_truth_diff": rmse_truth_qc,
            "truth_qc_pass": bool(max_log_diff <= args.truth_log1p_tolerance),
        })

        # Fair LOCO baseline: all 21 samples from OTHER conditions.
        train_meta = meta[meta["cell_type"] != cond]
        if len(train_meta) != 21:
            raise RuntimeError(
                f"{sid}: expected 21 LOCO training samples, got {len(train_meta)}"
            )

        global_train_mean = np.vstack(
            [truth_log[x] for x in train_meta["sample_id"]]
        ).mean(axis=0)

        method_vectors = {
            "Top2All_LOCO": top2all_pred,
            "GlobalTrainMean_LOCO": global_train_mean,
        }

        for method, pred_vec in method_vectors.items():
            prediction_vectors[method][sid] = pred_vec
            met = calc_metrics(true_vec, pred_vec)

            sample_rows.append({
                "sample_id": sid,
                "heldout_condition": cond,
                "rep": rep,
                "method": method,
                "n_genes_fixed_nonTopN": len(fixed_genes),
                **met,
            })

    qc_df = pd.DataFrame(qc_rows)
    if not qc_df["truth_qc_pass"].all():
        bad = qc_df[~qc_df["truth_qc_pass"]]
        raise RuntimeError(
            "Truth QC failed for one or more samples:\n"
            + bad.to_string(index=False)
        )

    qc_out = out_dir / "loco_truth_qc.csv"
    qc_df.to_csv(qc_out, index=False)

    sample_df = pd.DataFrame(sample_rows)
    sample_out = out_dir / "loco_samplewise_fixed_nonTopN_log1p.csv"
    sample_df.to_csv(sample_out, index=False)

    # Overall sample-wise summary by method.
    sample_summary = summarize_metrics(
        sample_df, ["method"], "n_samples"
    )
    sample_summary_out = (
        out_dir / "loco_samplewise_summary_fixed_nonTopN_log1p.csv"
    )
    sample_summary.to_csv(sample_summary_out, index=False)

    # Condition-specific summary across the 3 held-out replicates.
    condition_summary = summarize_metrics(
        sample_df, ["method", "heldout_condition"], "n_samples"
    )
    condition_summary_out = (
        out_dir / "loco_conditionwise_summary_fixed_nonTopN_log1p.csv"
    )
    condition_summary.to_csv(condition_summary_out, index=False)

    # Paired Top2All-vs-baseline improvements per sample.
    wide = sample_df.pivot_table(
        index=["sample_id", "heldout_condition", "rep"],
        columns="method",
        values=METRICS,
        aggfunc="first",
    )

    improvement_rows = []
    for idx in wide.index:
        sid, cond, rep = idx
        row = {
            "sample_id": sid,
            "heldout_condition": cond,
            "rep": rep,
        }
        for metric in METRICS:
            top = wide.loc[idx, (metric, "Top2All_LOCO")]
            base = wide.loc[idx, (metric, "GlobalTrainMean_LOCO")]

            # Positive = Top2All better.
            if metric in ["rmse", "mae"]:
                imp = base - top
            else:
                imp = top - base

            row[f"{metric}_improvement_vs_global_train_mean"] = float(imp)

        improvement_rows.append(row)

    improvement_df = pd.DataFrame(improvement_rows)
    improvement_out = (
        out_dir / "loco_top2all_improvement_vs_global_train_mean_samplewise.csv"
    )
    improvement_df.to_csv(improvement_out, index=False)

    # -------------------------------------------------------------
    # Gene-wise evaluation across all 24 LOCO predictions.
    # -------------------------------------------------------------
    sample_order = meta.sort_values(
        ["cell_type", "rep", "sample_id"]
    )["sample_id"].tolist()

    true_mat = np.vstack([truth_log[sid] for sid in sample_order])

    method_mats = {
        method: np.vstack(
            [prediction_vectors[method][sid] for sid in sample_order]
        )
        for method in METHODS
    }

    true_mean = np.mean(true_mat, axis=0)
    true_sd = np.std(true_mat, axis=0, ddof=1)
    true_variable = true_sd > 0

    sd_quantile = pd.Series(
        index=np.arange(len(fixed_genes)), dtype="object"
    )

    var_idx = np.where(true_variable)[0]
    if len(var_idx):
        tmp = pd.Series(true_sd[var_idx], index=var_idx)
        sd_quantile.loc[var_idx] = safe_qcut(tmp, q=4).values

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
    gene_out = out_dir / "loco_genewise_fixed_nonTopN_log1p.csv"
    gene_df.to_csv(gene_out, index=False)

    gene_summary_all = summarize_metrics(
        gene_df, ["method"], "n_genes_evaluable"
    )
    gene_summary_all.insert(0, "set", "all_fixed_nonTopN")

    gene_summary_var = summarize_metrics(
        gene_df[gene_df["true_variable"]].copy(),
        ["method"],
        "n_genes_evaluable",
    )
    gene_summary_var.insert(0, "set", "true_variable_only")

    gene_summary = pd.concat(
        [gene_summary_all, gene_summary_var], ignore_index=True
    )

    gene_summary_out = (
        out_dir / "loco_genewise_summary_fixed_nonTopN_log1p.csv"
    )
    gene_summary.to_csv(gene_summary_out, index=False)

    sd_summary = summarize_genewise_by_sd(gene_df)
    sd_summary_out = (
        out_dir / "loco_genewise_by_trueSD_quartile.csv"
    )
    sd_summary.to_csv(sd_summary_out, index=False)

    # -------------------------------------------------------------
    # Run info
    # -------------------------------------------------------------
    run_info = [
        f"topN: {args.topn}",
        f"n samples: {len(meta)}",
        f"n conditions: {len(conditions)}",
        f"conditions: {', '.join(conditions)}",
        f"prediction gene universe: {len(gene_order)}",
        f"TopN union in gene universe: {len(set(gene_order) & topn_union)}",
        f"fixed non-TopN genes: {len(fixed_genes)}",
        f"variable fixed genes: {int(np.sum(true_variable))}",
        "",
        "LOCO design:",
        "  For each condition, all 3 samples from that condition are test.",
        "  Training baseline uses 21 samples from the other 7 conditions.",
        "  SameConditionMean is not used because it would leak the held-out condition.",
        "",
        "Methods:",
        "  Top2All_LOCO = log1p(raw_pred_wo_overwrite)",
        "  GlobalTrainMean_LOCO = mean training truth in log1p(raw) space",
        "",
        "Outputs:",
        f"  {qc_out}",
        f"  {sample_out}",
        f"  {sample_summary_out}",
        f"  {condition_summary_out}",
        f"  {improvement_out}",
        f"  {gene_out}",
        f"  {gene_summary_out}",
        f"  {sd_summary_out}",
    ]

    (out_dir / "evaluation_run_info.txt").write_text(
        "\n".join(run_info) + "\n", encoding="utf-8"
    )

    # -------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------
    print("[DONE] LOCO evaluation completed")
    print(f"[INFO] n LOCO samples             : {len(meta)}")
    print(f"[INFO] n held-out conditions      : {len(conditions)}")
    print(f"[INFO] prediction gene universe   : {len(gene_order)}")
    print(f"[INFO] TopN union in gene universe: {len(set(gene_order) & topn_union)}")
    print(f"[INFO] fixed non-TopN genes       : {len(fixed_genes)}")
    print(f"[INFO] variable fixed genes       : {int(np.sum(true_variable))}")
    print(f"[INFO] truth QC max log1p diff    : {qc_df['max_abs_log1p_truth_diff'].max():.6g}")
    print()

    print("=== SAMPLE-WISE LOCO SUMMARY ===")
    print(sample_summary.to_string(index=False))
    print()

    print("=== CONDITION-WISE TOP2ALL LOCO SUMMARY ===")
    top_cond = condition_summary[
        condition_summary["method"] == "Top2All_LOCO"
    ].copy()
    print(top_cond.to_string(index=False))
    print()

    print("=== GENE-WISE LOCO SUMMARY: TRUE-VARIABLE GENES ONLY ===")
    print(
        gene_summary[
            gene_summary["set"] == "true_variable_only"
        ].to_string(index=False)
    )
    print()

    print("=== GENE-WISE LOCO BY TRUE-SD QUARTILE ===")
    print(sd_summary.to_string(index=False))
    print()

    print("=== MEDIAN SAMPLE-WISE IMPROVEMENT VS GLOBAL TRAIN MEAN ===")
    imp_cols = [
        c for c in improvement_df.columns
        if c.endswith("_improvement_vs_global_train_mean")
    ]
    med = improvement_df[imp_cols].median().to_frame("median_improvement").reset_index()
    med = med.rename(columns={"index": "metric"})
    print(med.to_string(index=False))


if __name__ == "__main__":
    main()
