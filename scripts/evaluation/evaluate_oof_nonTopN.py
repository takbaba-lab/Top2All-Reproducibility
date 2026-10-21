#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate Top2All 3-fold OOF predictions on non-TopN genes only.

Primary evaluation scale:
    log1p(raw_true) vs log1p(raw_pred_wo_overwrite)

Outputs:
    1) oof_sample_metrics_nonTopN_log1p.csv
       - one row per held-out sample (expected: 24 rows)
    2) oof_fold_summary_nonTopN_log1p.csv
       - fold-wise summary of the 5 metrics
    3) oof_overall_summary_nonTopN_log1p.csv
       - overall summary across 24 sample-level metrics
    4) oof_pooled_metrics_nonTopN_log1p.csv
       - pooled gene-sample-pair metrics (secondary)
    5) evaluation_run_info.txt
       - input/output validation summary

Important:
    - Uses raw_pred_wo_overwrite, NOT raw_pred_final.
    - Excludes every gene listed in the corresponding TopN CSV
      (first N rows, matching the prediction pipeline).
    - Stops with an error if predictions/TopN files are missing.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["pearson", "spearman", "rmse", "mae", "r2"]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
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


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
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


def calc_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 2:
        return {k: np.nan for k in METRICS}

    err = y - x
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))

    sse = np.sum((y - x) ** 2)
    sst = np.sum((x - x.mean()) ** 2)
    r2 = np.nan if sst == 0 else 1.0 - (sse / sst)

    return {
        "pearson": pearson_corr(x, y),
        "spearman": spearman_corr(x, y),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
    }


def parse_sample_id(filename: str) -> str:
    suffix = "_pred_vs_true_v6_raw_final.csv"
    if not filename.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {filename}")
    return filename[:-len(suffix)]


def parse_rep_from_sample_id(sample_id: str) -> int:
    m = re.search(r"_rep(\d+)$", sample_id)
    if not m:
        raise ValueError(f"Cannot parse rep from sample_id: {sample_id}")
    return int(m.group(1))


def parse_fold_dir(fold_dir: Path):
    m = re.fullmatch(r"fold(\d+)_test_rep(\d+)", fold_dir.name)
    if not m:
        raise ValueError(f"Unexpected fold directory name: {fold_dir.name}")
    return int(m.group(1)), int(m.group(2))


def choose_gene_column(df: pd.DataFrame, path: Path) -> str:
    for c in ["Gene", "gene", "GENE"]:
        if c in df.columns:
            return c
    raise ValueError(f"No gene column found in {path}; columns={df.columns.tolist()}")


def summarize_metrics(df: pd.DataFrame, group_col=None) -> pd.DataFrame:
    rows = []
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=True))

    for group_value, sub in groups:
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row = {
                "metric": metric,
                "n_samples": int(vals.size),
                "mean": float(np.mean(vals)) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
                "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "min": float(np.min(vals)) if vals.size else np.nan,
                "max": float(np.max(vals)) if vals.size else np.nan,
            }
            if group_col is not None:
                row[group_col] = group_value
            rows.append(row)

    out = pd.DataFrame(rows)
    if group_col is not None:
        cols = [group_col, "metric", "n_samples", "mean", "median", "sd", "min", "max"]
        out = out[cols]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--oof_base_dir",
        required=True,
        help="Directory containing fold1_test_rep1, fold2_test_rep2, fold3_test_rep3",
    )
    ap.add_argument(
        "--metadata_csv",
        required=True,
        help="sample_metadata_zF_gen1.csv",
    )
    ap.add_argument(
        "--topn_dir",
        required=True,
        help="Directory containing zF_*_topN.csv files",
    )
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--expected_n_samples", type=int, default=24)
    ap.add_argument(
        "--negative_tolerance",
        type=float,
        default=1e-6,
        help="Values below -tolerance cause an error; tiny negatives are clipped to 0 before log1p.",
    )
    args = ap.parse_args()

    oof_base = Path(args.oof_base_dir)
    metadata_csv = Path(args.metadata_csv)
    topn_dir = Path(args.topn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not oof_base.is_dir():
        raise FileNotFoundError(f"oof_base_dir not found: {oof_base}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")
    if not topn_dir.is_dir():
        raise FileNotFoundError(f"topn_dir not found: {topn_dir}")

    meta = pd.read_csv(metadata_csv)
    required_meta = {"sample_id", "cell_type"}
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise ValueError(f"metadata missing columns: {sorted(missing_meta)}")

    topn_meta_col = f"top{args.topn}_file"
    if topn_meta_col not in meta.columns:
        raise ValueError(
            f"metadata lacks '{topn_meta_col}'. columns={meta.columns.tolist()}"
        )

    if meta["sample_id"].duplicated().any():
        dup = meta.loc[meta["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id in metadata: {dup}")

    meta_by_sample = meta.set_index("sample_id", drop=False)
    expected_samples = set(meta["sample_id"].astype(str))

    fold_dirs = sorted(
        [p for p in oof_base.iterdir() if p.is_dir() and re.fullmatch(r"fold\d+_test_rep\d+", p.name)],
        key=lambda p: parse_fold_dir(p)[0],
    )
    if len(fold_dirs) != 3:
        raise RuntimeError(
            f"Expected exactly 3 fold directories, found {len(fold_dirs)}: "
            f"{[p.name for p in fold_dirs]}"
        )

    rows = []
    seen_samples = set()
    pooled_true = []
    pooled_pred = []
    run_lines = []

    run_lines.append(f"oof_base_dir: {oof_base}")
    run_lines.append(f"metadata_csv: {metadata_csv}")
    run_lines.append(f"topn_dir: {topn_dir}")
    run_lines.append(f"topN: {args.topn}")
    run_lines.append("evaluation: non-TopN only")
    run_lines.append("scale: log1p(raw)")
    run_lines.append("prediction column: raw_pred_wo_overwrite")
    run_lines.append("")

    for fold_dir in fold_dirs:
        fold, test_rep = parse_fold_dir(fold_dir)
        pred_root = fold_dir / "prediction"
        if not pred_root.is_dir():
            raise FileNotFoundError(f"prediction directory not found: {pred_root}")

        pred_files = sorted(pred_root.rglob("*_pred_vs_true_v6_raw_final.csv"))
        if len(pred_files) != 8:
            raise RuntimeError(
                f"{fold_dir.name}: expected 8 prediction CSVs, found {len(pred_files)}"
            )

        run_lines.append(
            f"{fold_dir.name}: {len(pred_files)} prediction files; expected held-out rep={test_rep}"
        )

        for pred_file in pred_files:
            sample_id = parse_sample_id(pred_file.name)

            if sample_id in seen_samples:
                raise RuntimeError(f"Sample predicted more than once across folds: {sample_id}")
            seen_samples.add(sample_id)

            if sample_id not in meta_by_sample.index:
                raise KeyError(f"Prediction sample not found in metadata: {sample_id}")

            sample_rep = parse_rep_from_sample_id(sample_id)
            if sample_rep != test_rep:
                raise RuntimeError(
                    f"OOF mismatch: {sample_id} is rep{sample_rep}, "
                    f"but found under test_rep{test_rep}"
                )

            mrow = meta_by_sample.loc[sample_id]
            cell_type = str(mrow["cell_type"])

            topn_file_value = str(mrow[topn_meta_col])
            topn_path = Path(topn_file_value)
            if not topn_path.is_file():
                topn_path = topn_dir / topn_file_value
            if not topn_path.is_file():
                raise FileNotFoundError(
                    f"TopN file not found for {sample_id}: {topn_file_value}"
                )

            top_df = pd.read_csv(topn_path)
            top_gene_col = choose_gene_column(top_df, topn_path)
            if len(top_df) < args.topn:
                raise RuntimeError(
                    f"{topn_path} has only {len(top_df)} rows; requested Top{args.topn}"
                )

            # Match prediction pipeline: first N rows.
            top_genes_list = (
                top_df[top_gene_col]
                .astype(str)
                .iloc[: args.topn]
                .tolist()
            )
            top_genes = set(top_genes_list)

            pred_df = pd.read_csv(pred_file)
            needed = {"gene", "raw_true", "raw_pred_wo_overwrite"}
            missing = needed - set(pred_df.columns)
            if missing:
                raise ValueError(
                    f"{pred_file} missing required columns: {sorted(missing)}"
                )

            if pred_df["gene"].astype(str).duplicated().any():
                raise RuntimeError(f"Duplicate genes in prediction CSV: {pred_file}")

            genes = pred_df["gene"].astype(str)
            non_mask = ~genes.isin(top_genes).to_numpy()
            overlap = int((~non_mask).sum())

            if overlap == 0:
                raise RuntimeError(
                    f"No TopN genes overlap prediction gene universe for {sample_id}; "
                    "cannot define non-TopN evaluation reliably."
                )

            true_raw = pd.to_numeric(
                pred_df.loc[non_mask, "raw_true"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            pred_raw = pd.to_numeric(
                pred_df.loc[non_mask, "raw_pred_wo_overwrite"], errors="coerce"
            ).to_numpy(dtype=np.float64)

            finite = np.isfinite(true_raw) & np.isfinite(pred_raw)
            n_nonfinite = int((~finite).sum())
            true_raw = true_raw[finite]
            pred_raw = pred_raw[finite]

            if true_raw.size < 2:
                raise RuntimeError(f"Too few finite non-TopN genes for {sample_id}")

            min_true = float(np.min(true_raw))
            min_pred = float(np.min(pred_raw))
            if min_true < -args.negative_tolerance or min_pred < -args.negative_tolerance:
                raise RuntimeError(
                    f"Substantive negative raw value found for {sample_id}: "
                    f"min_true={min_true}, min_pred={min_pred}"
                )

            # Tiny floating-point negatives only.
            true_raw = np.clip(true_raw, 0.0, None)
            pred_raw = np.clip(pred_raw, 0.0, None)

            true_log1p = np.log1p(true_raw)
            pred_log1p = np.log1p(pred_raw)

            metrics = calc_metrics(true_log1p, pred_log1p)

            rows.append(
                {
                    "fold": fold,
                    "test_rep": test_rep,
                    "sample_id": sample_id,
                    "cell_type": cell_type,
                    "topN": args.topn,
                    "prediction_file": str(pred_file),
                    "topN_file": str(topn_path),
                    "n_genes_all": int(len(pred_df)),
                    "n_topN_rows_requested": int(args.topn),
                    "n_topN_unique": int(len(top_genes)),
                    "n_topN_overlap_in_prediction": overlap,
                    "n_genes_nonTopN": int(non_mask.sum()),
                    "n_nonfinite_removed": n_nonfinite,
                    **metrics,
                }
            )

            pooled_true.append(true_log1p)
            pooled_pred.append(pred_log1p)

    sample_df = pd.DataFrame(rows)

    if len(sample_df) != args.expected_n_samples:
        raise RuntimeError(
            f"Expected {args.expected_n_samples} OOF samples, got {len(sample_df)}"
        )

    if sample_df["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id found in final OOF metrics table")

    missing_expected = expected_samples - seen_samples
    unexpected = seen_samples - expected_samples
    if missing_expected or unexpected:
        raise RuntimeError(
            f"OOF sample set mismatch. missing={sorted(missing_expected)}, "
            f"unexpected={sorted(unexpected)}"
        )

    sample_df = sample_df.sort_values(
        ["test_rep", "cell_type", "sample_id"]
    ).reset_index(drop=True)

    sample_out = out_dir / "oof_sample_metrics_nonTopN_log1p.csv"
    sample_df.to_csv(sample_out, index=False)

    fold_summary = summarize_metrics(sample_df, group_col="fold")
    fold_out = out_dir / "oof_fold_summary_nonTopN_log1p.csv"
    fold_summary.to_csv(fold_out, index=False)

    overall_summary = summarize_metrics(sample_df, group_col=None)
    overall_out = out_dir / "oof_overall_summary_nonTopN_log1p.csv"
    overall_summary.to_csv(overall_out, index=False)

    pooled_true_arr = np.concatenate(pooled_true)
    pooled_pred_arr = np.concatenate(pooled_pred)
    pooled_metrics = calc_metrics(pooled_true_arr, pooled_pred_arr)
    pooled_df = pd.DataFrame(
        [
            {
                "n_gene_sample_pairs": int(pooled_true_arr.size),
                **pooled_metrics,
            }
        ]
    )
    pooled_out = out_dir / "oof_pooled_metrics_nonTopN_log1p.csv"
    pooled_df.to_csv(pooled_out, index=False)

    run_lines.append("")
    run_lines.append(f"validated OOF samples: {len(sample_df)}")
    run_lines.append(f"unique samples: {sample_df['sample_id'].nunique()}")
    run_lines.append(f"pooled non-TopN gene-sample pairs: {pooled_true_arr.size}")
    run_lines.append("")
    run_lines.append("Per-fold sample counts:")
    for fold, n in sample_df.groupby("fold").size().items():
        run_lines.append(f"  fold{fold}: {n}")
    run_lines.append("")
    run_lines.append("Outputs:")
    run_lines.append(f"  {sample_out}")
    run_lines.append(f"  {fold_out}")
    run_lines.append(f"  {overall_out}")
    run_lines.append(f"  {pooled_out}")

    info_out = out_dir / "evaluation_run_info.txt"
    info_out.write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    print("[DONE] OOF non-TopN evaluation completed.")
    print(f"[DONE] sample metrics : {sample_out}")
    print(f"[DONE] fold summary   : {fold_out}")
    print(f"[DONE] overall summary: {overall_out}")
    print(f"[DONE] pooled metrics : {pooled_out}")
    print(f"[DONE] run info       : {info_out}")
    print()
    print("Overall sample-level summary:")
    print(overall_summary.to_string(index=False))
    print()
    print("Pooled metrics (secondary):")
    print(pooled_df.to_string(index=False))


if __name__ == "__main__":
    main()
