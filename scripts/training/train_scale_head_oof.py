#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_scale_head.py (final)

Purpose:
  - Load a trained log1p model checkpoint (ZFReconstructionModelV5)
  - Freeze the log1p model (encoder + heads)
  - Train scale_head only using raw-aware input:
      x = concat_features + log(topN_raw_total)
  - Loss aligns per-sample total raw counts in common-genes space:
      loss = mean( (log(sum(raw_hat)+eps) - log(sum(raw_true)+eps))^2 )

Outputs:
  out_scale_dir/
    runs/scale_top{N}_YYYYmmdd_HHMMSS/
      train.log
      run_meta.json
      metrics.csv
      scale_head_last.pt  # fixed-epoch OOF checkpoint
"""

import argparse
import csv
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys as _s3_sys
from pathlib import Path as _S3Path
_s3_sys.path.insert(0, str(_S3Path(__file__).resolve().parents[1]))
from common import preprocess, PIPELINE, sha, BUNDLE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModel


# ----------------------------
# Utility: Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Utility: Logging + Run Dir
# ----------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def try_get_git_hash(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def setup_run_logging(out_scale_dir: Path, topn: int) -> Tuple[Path, logging.Logger, Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_scale_dir / "runs" / f"scale_top{topn}_{ts}"
    safe_mkdir(run_dir)

    log_path = run_dir / "train.log"
    meta_path = run_dir / "run_meta.json"
    metrics_path = run_dir / "metrics.csv"

    logger = logging.getLogger(f"train_scale_head_top{topn}_{ts}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear handlers (in case of interactive / re-run)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file      : {log_path}")
    logger.info(f"Meta JSON     : {meta_path}")

    return run_dir, logger, meta_path, metrics_path


# ----------------------------
# Token dict loader
# ----------------------------
def find_token_pkl_files(pkl_dir: Path) -> Tuple[Path, Path]:
    """
    Expected files (your environment):
      - MLM-re_token_dictionary_v1.pkl
      - MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl
    But we search robustly.
    """
    pkl_dir = Path(pkl_dir)

    cand_token = pkl_dir / "MLM-re_token_dictionary_v1.pkl"
    cand_sym2ens = pkl_dir / "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
    if cand_token.exists() and cand_sym2ens.exists():
        return cand_token, cand_sym2ens

    token_pkls = sorted(pkl_dir.glob("MLM-re_token_dictionary*.pkl"))
    sym2ens_pkls = sorted(pkl_dir.glob("MLM-re_token_dictionary*_GeneSymbol_to_EnsemblID*.pkl"))

    # Exclude sym2ens from token list
    token_pkls = [p for p in token_pkls if "GeneSymbol_to_EnsemblID" not in p.name]

    if not token_pkls:
        raise FileNotFoundError(f"Token dictionary pkl not found in: {pkl_dir}")
    if not sym2ens_pkls:
        raise FileNotFoundError(f"GeneSymbol->EnsemblID pkl not found in: {pkl_dir}")

    # take the latest-ish (last)
    return token_pkls[-1], sym2ens_pkls[-1]


def load_token_dicts(pkl_dir: str) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """
    Returns:
      symbol2ensembl: dict (GeneSymbol -> EnsemblID)
      ensembl2id: dict (EnsemblID or special token -> int)
      pad_id: int (for "<pad>" or "<PAD>" etc)
    """
    import pickle

    pkl_dir = Path(pkl_dir)
    token_pkl, sym2ens_pkl = find_token_pkl_files(pkl_dir)

    obj_token = pickle.load(open(token_pkl, "rb"))
    obj_sym2ens = pickle.load(open(sym2ens_pkl, "rb"))

    if not isinstance(obj_token, dict):
        raise TypeError(f"token dict pkl is not dict: {token_pkl}")
    if not isinstance(obj_sym2ens, dict):
        raise TypeError(f"symbol2ensembl pkl is not dict: {sym2ens_pkl}")

    # Normalize keys to str
    ensembl2id = {}
    for k, v in obj_token.items():
        ensembl2id[str(k)] = int(v)

    symbol2ensembl = {}
    for k, v in obj_sym2ens.items():
        # k: gene symbol, v: ensembl
        if v is None:
            continue
        symbol2ensembl[str(k)] = str(v)

    # pad token: prefer "<pad>"
    if "<pad>" in ensembl2id:
        pad_id = ensembl2id["<pad>"]
    elif "<PAD>" in ensembl2id:
        pad_id = ensembl2id["<PAD>"]
    else:
        raise KeyError("token dict lacks <pad>. Check token dictionary pkl.")

    return symbol2ensembl, ensembl2id, pad_id


# ----------------------------
# IO: gene list and CSV readers
# ----------------------------
def load_common_genes_from_txt(path: str) -> List[str]:
    genes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    return genes


def read_gene_expression_csv(path: Path) -> pd.DataFrame:
    """
    Reads CSV that contains gene and expression columns.
    Accepts column variants: Gene/gene, Expression/expression, etc.
    If not found, uses first two columns.
    Returns DataFrame with columns: gene(str), expr(float)
    """
    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns.tolist()]

    gene_col = None
    expr_col = None

    # gene column candidates
    for c in cols:
        if c.lower() in ("gene", "genes", "symbol", "id"):
            gene_col = c
            break
    # expr column candidates
    for c in cols:
        if c.lower() in ("expression", "expr", "count", "counts", "value"):
            expr_col = c
            break

    if gene_col is None or expr_col is None:
        if len(cols) < 2:
            raise ValueError(f"CSV has <2 columns: {path} columns={cols}")
        gene_col = cols[0]
        expr_col = cols[1]

    out = pd.DataFrame({
        "gene": df[gene_col].astype(str).values,
        "expr": pd.to_numeric(df[expr_col], errors="coerce").fillna(0.0).astype(float).values
    })
    return out


def find_topn_file(topn_dir: Path, sample_id: str, topn: int) -> Path:
    topn_dir = Path(topn_dir)
    # reference
    cand = topn_dir / f"{sample_id}_top{topn}.csv"
    if cand.exists():
        return cand
    # fallback glob
    hits = sorted(topn_dir.glob(f"{sample_id}*top{topn}*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"TopN file not found for sample_id={sample_id}, topn={topn} under: {topn_dir}")


def find_full_file(full_dir: Path, sample_id: str) -> Path:
    full_dir = Path(full_dir)
    # reference
    cand1 = full_dir / f"{sample_id}_full.csv"
    if cand1.exists():
        return cand1
    cand2 = full_dir / f"{sample_id}.csv"
    if cand2.exists():
        return cand2
    hits = sorted(full_dir.glob(f"{sample_id}*full*.csv"))
    if hits:
        return hits[0]
    hits = sorted(full_dir.glob(f"{sample_id}*.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Full file not found for sample_id={sample_id} under: {full_dir}")


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class SampleRecord:
    sample_id: str
    cell_type: str


class ZFBulkTopNDataset(Dataset):
    """
    Builds model inputs from TopN + Full files per sample.

    Returns dict keys (minimal):
      sample_id (str)
      cell_type (str)
      cell_type_ids (int)
      input_ids (LongTensor)           shape (L=topn,)
      attention_mask (LongTensor)      shape (L=topn,)
      expr_norm_seq (FloatTensor)      shape (L=topn,)
      target_log1p (FloatTensor)       shape (G=num_genes,)
      topn_raw_total (FloatTensor)     shape (1,)
      topn_raw_log (FloatTensor)       shape (1,)
    """

    def __init__(
        self,
        records: List[SampleRecord],
        topn_dir: Path,
        full_dir: Path,
        topn: int,
        common_genes: List[str],
        cell_type_to_id: Dict[str, int],
        symbol2ensembl: Dict[str, str],
        ensembl2id: Dict[str, int],
        pad_id: int,
        eps: float = 1e-6,
    ):
        self.records = records
        self.topn_dir = Path(topn_dir)
        self.full_dir = Path(full_dir)
        self.topn = int(topn)
        self.common_genes = list(common_genes)
        self.gene_index = {g: i for i, g in enumerate(self.common_genes)}
        self.num_genes = len(self.common_genes)
        self.cell_type_to_id = dict(cell_type_to_id)

        self.symbol2ensembl = dict(symbol2ensembl)
        self.ensembl2id = dict(ensembl2id)
        self.pad_id = int(pad_id)
        self.eps = float(eps)

    def __len__(self):
        return len(self.records)

    def _gene_to_ensembl(self, gene: str) -> str:
        # if already ensembl-like, keep
        if gene.startswith("ENSMUSG") or gene.startswith("ENSG"):
            return gene
        return self.symbol2ensembl.get(gene, gene)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        sample_id = rec.sample_id
        cell_type = rec.cell_type
        cell_type_id = self.cell_type_to_id[cell_type]

        # --- TopN file ---
        topn_path = find_topn_file(self.topn_dir, sample_id, self.topn)
        df_topn = read_gene_expression_csv(topn_path)

        prepared = preprocess(df_topn["gene"], df_topn["expr"],
                                self.symbol2ensembl, self.ensembl2id, self.topn)
        length = len(prepared["ids"])
        ids = np.full(self.topn, self.pad_id, dtype=np.int64)
        ids[:length] = prepared["ids"]
        mask = np.zeros(self.topn, dtype=np.int64)
        mask[:length] = 1
        norm = np.zeros(self.topn, dtype=np.float32)
        norm[:length] = prepared["normalized"]
        token_ids = torch.from_numpy(ids)
        attention_mask = torch.from_numpy(mask)
        expr_norm_seq = torch.from_numpy(norm)
        topn_raw_total_t = torch.tensor([prepared["total"]], dtype=torch.float32)
        topn_raw_log_t = torch.tensor([prepared["log_total"]], dtype=torch.float32)

        # --- Full file (target) ---
        full_path = find_full_file(self.full_dir, sample_id)
        df_full = read_gene_expression_csv(full_path)
        full_map = {g: float(x) for g, x in zip(df_full["gene"].astype(str), df_full["expr"].astype(float))}

        target_raw = np.zeros((self.num_genes,), dtype=np.float32)
        for g, i in self.gene_index.items():
            target_raw[i] = float(full_map.get(g, 0.0))

        target_log1p = torch.tensor(np.log1p(target_raw), dtype=torch.float32)

        return {
            "sample_id": sample_id,
            "cell_type": cell_type,
            "cell_type_ids": torch.tensor(cell_type_id, dtype=torch.long),
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "expr_norm_seq": expr_norm_seq,
            "target_log1p": target_log1p,
            "topn_raw_total": topn_raw_total_t,
            "topn_raw_log": topn_raw_log_t,
        }


def collate_fn_scale(batch: List[dict]) -> dict:
    """Run metadata and model setup."""
    MAX_LEN_ENCODER = 2048
    PAD_ID = 0  #

    # --- determine max_len in this batch (cap by 2048) ---
    seq_lens = [int(b["input_ids"].shape[0]) for b in batch]
    max_len = min(MAX_LEN_ENCODER, max(seq_lens))

    def _trunc_pad_1d(x: torch.Tensor, pad_value) -> torch.Tensor:
        # truncate first
        x = x[:max_len]
        # pad if needed
        if x.shape[0] < max_len:
            pad = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        return x

    input_ids_list = []
    attention_mask_list = []
    expr_norm_list = []

    for b in batch:
        input_ids_list.append(_trunc_pad_1d(b["input_ids"], PAD_ID))
        attention_mask_list.append(_trunc_pad_1d(b["attention_mask"], 0))
        expr_norm_list.append(_trunc_pad_1d(b["expr_norm_seq"], 0.0))

    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "cell_type": [b["cell_type"] for b in batch],

        # scalar / vector fields (no truncation)
        "cell_type_ids": torch.stack([b["cell_type_ids"] for b in batch], dim=0),
        "target_log1p": torch.stack([b["target_log1p"] for b in batch], dim=0),
        "topn_raw_total": torch.stack([b["topn_raw_total"] for b in batch], dim=0),
        "topn_raw_log": torch.stack([b["topn_raw_log"] for b in batch], dim=0),

        # encoder sequence fields (truncated/padded)
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
        "expr_norm_seq": torch.stack(expr_norm_list, dim=0),
    }
    return out


# ----------------------------
# Model: ZFReconstructionModelV5 (same design as notebook)
# ----------------------------
class ZFReconstructionModelV5(nn.Module):
    """
    v5 model:
      - Geneformer encoder (frozen externally)
      - concat hidden + expr_norm -> fuser
      - pooling + cell_type embedding
      - 2-layer-ish MLP
      - zero head + value head
      - log1p_hat = sigmoid(zero_logits) * softplus(value)
    """

    def __init__(
        self,
        encoder,
        hidden_size: int,
        num_cell_types: int,
        num_genes: int,
        celltype_emb_dim: int = 32,
        aggregation_mode: str = "mean",
        activation_type: str = "softplus",
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = int(hidden_size)
        self.num_cell_types = int(num_cell_types)
        self.num_genes = int(num_genes)
        self.aggregation_mode = aggregation_mode
        self.activation_type = activation_type

        self.celltype_emb = nn.Embedding(num_cell_types, celltype_emb_dim)

        self.expr_fuser = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
        )

        in_dim = hidden_size + celltype_emb_dim
        self.input_norm = nn.LayerNorm(in_dim)

        hidden_dim = 1024
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.head_zero = nn.Linear(hidden_dim, num_genes)
        self.head_value = nn.Linear(hidden_dim, num_genes)

        if activation_type == "softplus":
            self.out_act = nn.Softplus()
        elif activation_type == "relu":
            self.out_act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}")

    def forward(self, input_ids, attention_mask, cell_type_ids, expr_norm_seq):
        # encoder no-grad (frozen usage)
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, L, H)

        expr_norm_expand = expr_norm_seq.unsqueeze(-1)  # (B, L, 1)
        fused_input = torch.cat([hidden_states, expr_norm_expand], dim=-1)  # (B, L, H+1)
        fused_hidden = self.expr_fuser(fused_input)  # (B, L, H)

        if self.aggregation_mode == "mean":
            mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
            masked_hidden = fused_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
            lengths = mask.sum(dim=1).clamp(min=1)
            cell_emb = sum_hidden / lengths
        else:
            cell_emb = fused_hidden[:, 0, :]

        ct_emb = self.celltype_emb(cell_type_ids)  # (B, Cdim)
        concat = torch.cat([cell_emb, ct_emb], dim=-1)  # (B, H+Cdim)
        concat = self.input_norm(concat)

        z = self.mlp(concat)
        zero_logits = self.head_zero(z)
        y_raw = self.head_value(z)
        y_pos = self.out_act(y_raw)
        p_nonzero = torch.sigmoid(zero_logits)
        log1p_hat = p_nonzero * y_pos
        return zero_logits, y_pos, log1p_hat

    @torch.no_grad()
    def get_concat_features(self, input_ids, attention_mask, cell_type_ids, expr_norm_seq):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)

        expr_norm_expand = expr_norm_seq.unsqueeze(-1)
        fused_input = torch.cat([hidden_states, expr_norm_expand], dim=-1)
        fused_hidden = self.expr_fuser(fused_input)

        if self.aggregation_mode == "mean":
            mask = attention_mask.unsqueeze(-1)
            masked_hidden = fused_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            cell_emb = sum_hidden / lengths
        else:
            cell_emb = fused_hidden[:, 0, :]

        ct_emb = self.celltype_emb(cell_type_ids)
        concat = torch.cat([cell_emb, ct_emb], dim=-1)
        concat = self.input_norm(concat)
        return concat


# ----------------------------
# Scale head (raw-aware)
# ----------------------------
class ScaleHeadRawAware(nn.Module):
    """
    scale = Softplus( MLP([concat, topn_raw_log]) )
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Loss: total raw alignment (log-space)
# ----------------------------
def total_count_log_mse(pred_total: torch.Tensor, true_total: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_log = torch.log(pred_total + eps)
    true_log = torch.log(true_total + eps)
    return torch.mean((pred_log - true_log) ** 2)


# ----------------------------
# Metadata parsing
# ----------------------------
def load_sample_metadata(sample_metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(sample_metadata_csv)

    # normalize columns
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Must have sample_id + cell_type
    # Accept variants
    def pick_col(cands):
        for c in df.columns:
            if c.lower() in cands:
                return c
        return None

    sid_col = pick_col({"sample_id", "sampleid", "sample"})
    ct_col = pick_col({"cell_type", "celltype", "group", "condition"})

    if sid_col is None or ct_col is None:
        raise ValueError(
            f"sample_metadata_csv must include sample_id and cell_type columns. "
            f"found columns={list(df.columns)}"
        )

    df = df.rename(columns={sid_col: "sample_id", ct_col: "cell_type"})

    # If there is a rep column, keep; otherwise infer from sample_id
    rep_col = None
    for c in df.columns:
        if c.lower() in ("rep", "replicate"):
            rep_col = c
            break
    if rep_col is None:
        # infer rep: find "rep[0-9]+"
        reps = []
        for sid in df["sample_id"].astype(str).tolist():
            m = re.search(r"rep(\d+)", sid)
            reps.append(int(m.group(1)) if m else -1)
        df["rep"] = reps
    else:
        df["rep"] = pd.to_numeric(df[rep_col], errors="coerce").fillna(-1).astype(int)

    return df[["sample_id", "cell_type", "rep"]].copy()


def split_train_test_oof(df_meta: pd.DataFrame, heldout_rep: int) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """OOF split: held-out replicate is a true test set and is not used during training."""
    train = []
    test = []
    heldout_rep = int(heldout_rep)
    for _, r in df_meta.iterrows():
        rec = SampleRecord(sample_id=str(r["sample_id"]), cell_type=str(r["cell_type"]))
        if int(r["rep"]) == heldout_rep:
            test.append(rec)
        else:
            train.append(rec)
    return train, test


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--log1p_ckpt", required=True, help="Path to log1p model checkpoint (.pt)")
    ap.add_argument("--geneformer_model_dir", required=True, help="Local directory of Geneformer model (for AutoModel)")
    ap.add_argument("--pkl_dir", required=True, help="Directory containing token dict pkls")
    ap.add_argument("--sample_metadata_csv", required=True, help="CSV with sample_id and cell_type (and optionally rep)")
    ap.add_argument("--common_genes_csv", required=True, help="Text file with common genes (one per line). Used if ckpt lacks common_genes")
    ap.add_argument("--topn_dir", required=True, help="Directory containing TopN CSV files")
    ap.add_argument("--full_dir", required=True, help="Directory containing Full CSV files")
    ap.add_argument("--topn", type=int, required=True, help="TopN value (100/200/500/1000/3000 etc)")
    ap.add_argument("--out_scale_dir", required=True, help="Output directory for scale_head checkpoints (runs/ will be created)")
    ap.add_argument(
        "--heldout_rep", type=int, required=True, choices=[1, 2, 3],
        help="OOF test replicate. This replicate is excluded completely from scale-head training and is not evaluated during training."
    )

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force_topn_mismatch", action="store_true", help="Proceed even if ckpt topn_field differs from --topn")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_scale_dir = Path(args.out_scale_dir)
    safe_mkdir(out_scale_dir)

    run_dir, logger, meta_path, metrics_path = setup_run_logging(out_scale_dir, args.topn)

    # --- environment meta ---
    repo_root = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) >= 2 else Path(".").resolve()
    git_hash = try_get_git_hash(repo_root)

    # --- load log1p ckpt ---
    ckpt = torch.load(args.log1p_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"log1p_ckpt is not a dict: {args.log1p_ckpt}")

    ckpt_keys = list(ckpt.keys())
    logger.info(f"[CKPT] keys: {ckpt_keys}")

    # topn consistency check (optional)
    ckpt_topn_field = ckpt.get("topn_field", None)
    if ckpt_topn_field is not None:
        # might be "top1000" or actual int etc.
        ckpt_topn_str = str(ckpt_topn_field)
        if str(args.topn) not in ckpt_topn_str:
            msg = f"[WARN] ckpt topn_field={ckpt_topn_field} seems different from --topn={args.topn}"
            if args.force_topn_mismatch:
                logger.info(msg + " (force proceed)")
            else:
                raise ValueError(msg + " (use --force_topn_mismatch to override)")
    else:
        logger.info("[CKPT] topn_field not found (skip topn consistency check)")

    # common_genes / cell_type_to_id from ckpt preferred
    common_genes = ckpt.get("common_genes", None)
    if common_genes is None:
        common_genes = load_common_genes_from_txt(args.common_genes_csv)
        logger.info(f"[INFO] common_genes loaded from txt: n={len(common_genes)}")
    else:
        common_genes = list(common_genes)
        logger.info(f"[INFO] common_genes loaded from ckpt: n={len(common_genes)}")

    cell_type_to_id = ckpt.get("cell_type_to_id", None)
    id_to_cell_type = ckpt.get("id_to_cell_type", None)
    if cell_type_to_id is None:
        # fallback to metadata order
        df_meta_tmp = load_sample_metadata(args.sample_metadata_csv)
        uniq = sorted(df_meta_tmp["cell_type"].astype(str).unique().tolist())
        cell_type_to_id = {ct: i for i, ct in enumerate(uniq)}
        id_to_cell_type = {i: ct for ct, i in cell_type_to_id.items()}
        logger.info("[WARN] cell_type_to_id not in ckpt; created from metadata.")
    else:
        cell_type_to_id = dict(cell_type_to_id)
        if id_to_cell_type is None:
            id_to_cell_type = {i: ct for ct, i in cell_type_to_id.items()}
        logger.info(f"[INFO] cell_type_to_id loaded from ckpt: {cell_type_to_id}")

    hidden_size = int(ckpt.get("hidden_size"))
    celltype_emb_dim = int(ckpt.get("celltype_emb_dim", 32))
    aggregation_mode = ckpt.get("aggregation_mode", "mean")
    activation_type = ckpt.get("activation_type", "softplus")

    # --- token dicts ---
    symbol2ensembl, ensembl2id, pad_id = load_token_dicts(args.pkl_dir)
    logger.info(f"[INFO] pad token found in token dict: pad_id={pad_id}")

    # --- data split ---
    df_meta = load_sample_metadata(args.sample_metadata_csv)
    # optional: ensure metadata cell_types subset matches ckpt
    meta_cell_types = sorted(df_meta["cell_type"].astype(str).unique().tolist())
    ckpt_cell_types = sorted(list(cell_type_to_id.keys()))
    logger.info(f"[RUN META] cell_types (meta): {meta_cell_types}")
    logger.info(f"[RUN META] cell_types (ckpt): {ckpt_cell_types}")

    train_records, test_records = split_train_test_oof(df_meta, args.heldout_rep)
    if len(train_records) < 1 or len(test_records) < 1:
        raise ValueError(f"Could not make OOF split for heldout_rep={args.heldout_rep}")
    logger.info(f"[RUN META] heldout_rep: {args.heldout_rep}")
    logger.info(f"[RUN META] n_train / n_test: {len(train_records)} / {len(test_records)}")
    logger.info("[RUN META] held-out test set will NOT be evaluated during scale-head training.")

    # --- training dataset/loader only; held-out test data are not loaded here ---
    train_ds = ZFBulkTopNDataset(
        records=train_records,
        topn_dir=Path(args.topn_dir),
        full_dir=Path(args.full_dir),
        topn=args.topn,
        common_genes=common_genes,
        cell_type_to_id=cell_type_to_id,
        symbol2ensembl=symbol2ensembl,
        ensembl2id=ensembl2id,
        pad_id=pad_id,
        eps=args.eps,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_scale,
        num_workers=0,
    )
    logger.info(f"[INFO] train_loader batches: {len(train_loader)}")
    logger.info(f"[INFO] held-out test samples not loaded during training: {len(test_records)}")

    # --- device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"[INFO] device: {device}")

    # --- load encoder + log1p model ---
    config = AutoConfig.from_pretrained(args.geneformer_model_dir)
    encoder = AutoModel.from_pretrained(args.geneformer_model_dir, config=config)

    # freeze encoder params
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)
    encoder.eval()

    model_v5 = ZFReconstructionModelV5(
        encoder=encoder,
        hidden_size=hidden_size,
        num_cell_types=len(cell_type_to_id),
        num_genes=len(common_genes),
        celltype_emb_dim=celltype_emb_dim,
        aggregation_mode=aggregation_mode,
        activation_type=activation_type,
    )
    model_state = ckpt.get("model_state_dict", None)
    if model_state is None:
        raise KeyError("log1p_ckpt lacks model_state_dict.")
    model_v5.load_state_dict(model_state, strict=True)
    model_v5.to(device)
    model_v5.eval()
    for p in model_v5.parameters():
        p.requires_grad = False

    # --- determine concat_dim dynamically from one batch ---
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        concat0 = model_v5.get_concat_features(
            input_ids=first_batch["input_ids"].to(device),
            attention_mask=first_batch["attention_mask"].to(device),
            cell_type_ids=first_batch["cell_type_ids"].to(device),
            expr_norm_seq=first_batch["expr_norm_seq"].to(device),
        )
    concat_dim = int(concat0.shape[1])
    in_dim = concat_dim + 1  # + topn_raw_log
    logger.info(f"[INFO] concat_dim={concat_dim}, scale_head_in_dim={in_dim}")

    # --- scale head ---
    scale_head = ScaleHeadRawAware(in_dim=in_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(scale_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- metrics writer ---
    with open(metrics_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_pred_total_mean",
            "train_true_total_mean",
        ])

    # --- run meta ---
    run_meta = {
        "script": "train_scale_head.py",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": git_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "args": vars(args),
        "log1p_ckpt_path": args.log1p_ckpt,
        "log1p_ckpt_keys": ckpt_keys,
        "common_genes_n": len(common_genes),
        "cell_type_to_id": cell_type_to_id,
        "evaluation_design": "3-fold OOF by biological replicate; held-out replicate not used during training",
        "heldout_rep": int(args.heldout_rep),
        "checkpoint_selection": "fixed_epoch_last",
        "n_train": len(train_records),
        "n_test": len(test_records),
        "train_sample_ids": [r.sample_id for r in train_records],
        "test_sample_ids": [r.sample_id for r in test_records],
        "rawaware": True,
        "uses_topn_raw_log": True,
        "concat_dim": concat_dim,
        "scale_head_arch": "mlp128_softplus",
        "eps": args.eps,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # --- train/val epoch functions ---
    def run_epoch(loader, train: bool):
        if train:
            scale_head.train()
        else:
            scale_head.eval()

        total_loss = 0.0
        n_batches = 0

        pred_totals = []
        true_totals = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            expr_norm_seq = batch["expr_norm_seq"].to(device)
            cell_type_ids = batch["cell_type_ids"].to(device)
            target_log1p = batch["target_log1p"].to(device)
            topn_raw_log = batch["topn_raw_log"].to(device)  # (B,1)

            with torch.no_grad():
                _, _, log1p_hat = model_v5(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cell_type_ids=cell_type_ids,
                    expr_norm_seq=expr_norm_seq,
                )
                concat = model_v5.get_concat_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cell_type_ids=cell_type_ids,
                    expr_norm_seq=expr_norm_seq,
                )

            # raw-aware input: [concat, log(topn_raw_total)]
            x = torch.cat([concat, topn_raw_log], dim=-1)  # (B, concat_dim+1)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                scale = scale_head(x)  # (B,1) positive
                raw_hat = torch.expm1(log1p_hat) * scale  # (B,G)
                pred_total = raw_hat.sum(dim=1)  # (B,)

                raw_true = torch.expm1(target_log1p)  # (B,G)
                true_total = raw_true.sum(dim=1)      # (B,)

                loss = total_count_log_mse(pred_total, true_total, eps=args.eps)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(scale_head.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            pred_totals.append(pred_total.detach().cpu())
            true_totals.append(true_total.detach().cpu())

        if n_batches == 0:
            return np.nan, np.nan, np.nan

        pred_mean = torch.cat(pred_totals).mean().item()
        true_mean = torch.cat(true_totals).mean().item()
        avg_loss = total_loss / n_batches
        return avg_loss, pred_mean, true_mean

    # --- fixed-epoch OOF training loop ---
    last_path = run_dir / "scale_head_last.pt"

    logger.info("=== Scale-head training (raw-aware, OOF fixed-epoch) ===")
    logger.info(f"Held-out rep {args.heldout_rep} is not loaded/evaluated during training.")
    last_train_loss = None
    last_train_pred_mean = None
    last_train_true_mean = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred_mean, train_true_mean = run_epoch(train_loader, train=True)
        last_train_loss = train_loss
        last_train_pred_mean = train_pred_mean
        last_train_true_mean = train_true_mean

        logger.info(
            f"[Scale Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"train_pred_total(mean)={train_pred_mean:.2f} | train_true_total(mean)={train_true_mean:.2f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch, train_loss, train_pred_mean, train_true_mean])

    # save fixed-epoch last checkpoint; no held-out data used for selection
    torch.save({
        "pipeline_id": PIPELINE,
        "preprocess_sha256": sha(BUNDLE / "preprocessing.py"),
        "scale_head_state_dict": scale_head.state_dict(),
        "rawaware": True,
        "uses_topn_raw_log": True,
        "topn": args.topn,
        "concat_dim": concat_dim,
        "scale_head_arch": "mlp128_softplus",
        "eps": args.eps,
        "log1p_ckpt_path": args.log1p_ckpt,
        "heldout_rep": int(args.heldout_rep),
        "checkpoint_selection": "fixed_epoch_last",
        "train_meta": {
            "cell_type_to_id": cell_type_to_id,
            "n_train": len(train_records),
            "n_test": len(test_records),
            "train_sample_ids": [r.sample_id for r in train_records],
            "test_sample_ids": [r.sample_id for r in test_records],
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "final_train_loss": float(last_train_loss),
            "final_train_pred_total_mean": float(last_train_pred_mean),
            "final_train_true_total_mean": float(last_train_true_mean),
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "note": "OOF scale_head last; held-out replicate never used; raw-aware input = concat + log(topn_raw_total); loss=log(total) MSE",
    }, last_path)
    # Also save a stable copy directly under out_scale_dir for downstream prediction.
    stable_last_path = out_scale_dir / "scale_head_last.pt"
    torch.save(torch.load(last_path, map_location="cpu", weights_only=False), stable_last_path)
    logger.info(f"[DONE] Saved fixed-epoch OOF scale_head: {last_path}")
    logger.info(f"[DONE] Stable OOF scale_head copy: {stable_last_path}")
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
