#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_rawaware.py
- Load log1p (TopN -> log1p_hat) checkpoint + scale_head checkpoint
- Predict raw counts for each bulk sample, and output:
  raw_pred_wo_overwrite : scale-applied raw prediction (no overwrite)
  raw_pred              : overwrite-applied raw prediction (final; backwards compatible)
  raw_pred_final        : same as raw_pred (alias)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys as _s3_sys
from pathlib import Path as _S3Path
_s3_sys.path.insert(0, str(_S3Path(__file__).resolve().parents[1]))
from common import preprocess, PIPELINE, sha, BUNDLE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, BertModel


# ------------------------
# Logging helpers
# ------------------------

def _nowstamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.log_path, "w", encoding="utf-8")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def print(self, msg: str):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


# ------------------------
# Token dict loaders
# ------------------------

def _find_one(pkl_dir: Path, contains: str) -> Path:
    cands = sorted([p for p in pkl_dir.glob("*.pkl") if contains in p.name])
    if not cands:
        raise FileNotFoundError(f"No .pkl found in {pkl_dir} containing: {contains}")
    return cands[0]


def load_token_dicts(pkl_dir: Path, logger: TeeLogger) -> Tuple[Dict[str, str], Dict[str, int], int]:
    """
    Expect:
      - MLM-re_token_dictionary_v1.pkl (Ensembl -> token id, plus special tokens)
      - MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl (GeneSymbol -> Ensembl)
    """
    token_dict_path = _find_one(pkl_dir, "token_dictionary_v1.pkl")
    sym2ens_path = _find_one(pkl_dir, "GeneSymbol_to_EnsemblID.pkl")

    with open(token_dict_path, "rb") as f:
        tok = pickle.load(f)
    with open(sym2ens_path, "rb") as f:
        sym2ens = pickle.load(f)

    if not isinstance(tok, dict) or not isinstance(sym2ens, dict):
        raise TypeError("Loaded token dicts are not dicts. Check pkl files.")

    # pad id: Geneformer token dict uses '<pad>'
    pad_id = None
    for k in ("<pad>", "<PAD>", "[PAD]", "pad"):
        if k in tok:
            pad_id = int(tok[k])
            break
    if pad_id is None:
        pad_id = 0
        logger.print("[WARN] pad token not found in token dictionary; using pad_id=0.")

    # tok keys can be numpy.str_; normalize to str
    ens2id = {str(k): int(v) for k, v in tok.items()}

    logger.print(f"[INFO] pad_id           : {pad_id}")
    return {str(k): str(v) for k, v in sym2ens.items()}, ens2id, pad_id


# ------------------------
# Model: condition-free v5 log1p head
# ------------------------

class ZFReconstructionModelV5NoCondition(nn.Module):
    """Condition-free ablation matching train_log1p_oof_nocond.py."""

    def __init__(
        self,
        encoder: BertModel,
        n_genes: int,
        hidden_size: int,
        mlp_hidden: int,
        concat_dim: int,
        aggregation_mode: str = "mean",
        activation_type: str = "softplus",
        dropout: float = 0.10,
        use_expr_fuser: bool = True,
        expr_fuser_in: Optional[int] = None,
        expr_fuser_out: Optional[int] = None,
        use_mlp_layernorm: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_genes = int(n_genes)
        self.hidden_size = int(hidden_size)
        self.concat_dim = int(concat_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.aggregation_mode = aggregation_mode
        self.activation_type = activation_type

        self.use_expr_fuser = bool(use_expr_fuser)
        if self.use_expr_fuser:
            if expr_fuser_in is None or expr_fuser_out is None:
                expr_fuser_in = self.hidden_size + 1
                expr_fuser_out = self.hidden_size
            self.expr_fuser = nn.Sequential(
                nn.Linear(int(expr_fuser_in), int(expr_fuser_out)),
                nn.GELU(),
            )

        self.input_norm = nn.LayerNorm(self.concat_dim)

        mlp_layers: List[nn.Module] = [
            nn.Linear(self.concat_dim, self.mlp_hidden),
            nn.GELU(),
        ]
        if use_mlp_layernorm:
            mlp_layers.append(nn.LayerNorm(self.mlp_hidden))
        mlp_layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlp_layers)

        self.head_zero = nn.Linear(self.mlp_hidden, self.n_genes)
        self.head_value = nn.Linear(self.mlp_hidden, self.n_genes)

        if activation_type == "softplus":
            self.activation = nn.Softplus()
        elif activation_type == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}")

    @torch.no_grad()
    def get_concat_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expr_norm_seq: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.use_expr_fuser:
            fused_input = torch.cat([hidden_states, expr_norm_seq.unsqueeze(-1)], dim=-1)
            hidden_states = self.expr_fuser(fused_input)

        if self.aggregation_mode == "cls":
            cell_emb = hidden_states[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp_min(1e-6)
            cell_emb = (hidden_states * mask).sum(dim=1) / denom

        return self.input_norm(cell_emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        expr_norm_seq: torch.Tensor,
    ) -> torch.Tensor:
        concat = self.get_concat_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expr_norm_seq=expr_norm_seq,
        )
        h = self.mlp(concat)
        logits_zero = self.head_zero(h)
        logits_value = self.head_value(h)
        p_nonzero = torch.sigmoid(logits_zero)
        log1p_val = self.activation(logits_value)
        return p_nonzero * log1p_val


# ------------------------
# Dataset
# ------------------------

def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 1.0
    if sd < 1e-8:
        sd = 1.0
    return (x - mu) / sd


@dataclass
class SampleRow:
    sample_id: str
    cell_type: str
    rep: str
    topn_csv: str
    full_csv: str


class ZFBulkDatasetV6Infer(Dataset):
    def __init__(
        self,
        rows: List[SampleRow],
        common_genes: List[str],
        gene2idx: Dict[str, int],
        symbol2ensembl: Dict[str, str],
        ensembl2id: Dict[str, int],
        pad_id: int,
        topn: int,
        max_seq_len: int,
    ):
        self.rows = rows
        self.common_genes = list(common_genes)
        self.gene2idx = dict(gene2idx)
        self.sym2ens = dict(symbol2ensembl)
        self.ens2id = dict(ensembl2id)
        self.pad_id = int(pad_id)
        self.topn = int(topn)
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        topn_df = pd.read_csv(r.topn_csv)
        full_df = pd.read_csv(r.full_csv)

        prepared = preprocess(topn_df["Gene"], topn_df["Expression"],
                                self.sym2ens, self.ens2id, self.topn, self.max_seq_len)
        L = len(prepared["ids"])
        input_ids = np.full(self.max_seq_len, self.pad_id, dtype=np.int64)
        attention = np.zeros(self.max_seq_len, dtype=np.int64)
        expr_norm_seq = np.zeros(self.max_seq_len, dtype=np.float32)
        overwrite_idx = np.full(self.max_seq_len, -1, dtype=np.int64)
        overwrite_val = np.zeros(self.max_seq_len, dtype=np.float32)
        input_ids[:L] = prepared["ids"]
        attention[:L] = 1
        expr_norm_seq[:L] = prepared["normalized"]
        overwrite_idx[:L] = [self.gene2idx.get(g, -1) for g in prepared["genes"]]
        overwrite_val[:L] = prepared["raw"]

        # full raw vector aligned to common_genes
        full_gene_to_raw = dict(zip(full_df["Gene"].astype(str), full_df["Expression"].astype(float)))
        target_raw = np.array([full_gene_to_raw.get(g, 0.0) for g in self.common_genes], dtype=np.float32)

        topn_raw_total = prepared["total"]
        topn_raw_log = prepared["log_total"]

        return {
            "sample_id": r.sample_id,
            "cell_type": r.cell_type,
            "rep": r.rep,
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention),
            "expr_norm_seq": torch.from_numpy(expr_norm_seq),
            "target_raw": torch.from_numpy(target_raw),
            "topn_raw_total": torch.tensor([topn_raw_total], dtype=torch.float32),
            "topn_raw_log": torch.tensor([topn_raw_log], dtype=torch.float32),
            "overwrite_idx": torch.from_numpy(overwrite_idx),
            "overwrite_val": torch.from_numpy(overwrite_val),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "cell_type": [b["cell_type"] for b in batch],
        "rep": [b["rep"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "expr_norm_seq": torch.stack([b["expr_norm_seq"] for b in batch], dim=0),
        "target_raw": torch.stack([b["target_raw"] for b in batch], dim=0),
        "topn_raw_total": torch.stack([b["topn_raw_total"] for b in batch], dim=0),
        "topn_raw_log": torch.stack([b["topn_raw_log"] for b in batch], dim=0),
        "overwrite_idx": torch.stack([b["overwrite_idx"] for b in batch], dim=0),
        "overwrite_val": torch.stack([b["overwrite_val"] for b in batch], dim=0),
    }


# ------------------------
# Scale head loader/build  (PatchB-1 / PatchB-2)
# ------------------------

def _extract_state_dict(obj: object) -> dict:
    """Run metadata and model setup."""
    if isinstance(obj, dict):
        for k in ("scale_head_state_dict", "scale_state_dict", "state_dict", "model_state_dict", "scale_head"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # already state_dict-like?
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise TypeError("Unsupported checkpoint format for state_dict extraction.")


def _normalize_scale_keys(sd: dict) -> dict:
    """Run metadata and model setup."""
    out = {}
    for k, v in sd.items():
        kk = str(k)

        # remove common wrappers
        if kk.startswith("module."):
            kk = kk[len("module."):]

        # if 'scale_head.' appears in middle, drop everything up to it
        if "scale_head." in kk:
            kk = kk.split("scale_head.", 1)[1]

        # optionally drop leading 'net.' (common in nn.Sequential wrapped as self.net)
        if kk.startswith("net."):
            kk = kk[len("net."):]

        out[kk] = v
    return out


def build_scale_head_from_state_dict(sd: dict, logger: Optional[TeeLogger] = None) -> nn.Sequential:
    """Run metadata and model setup."""
    if not isinstance(sd, dict) or len(sd) == 0:
        raise ValueError("scale state_dict is empty or not a dict")

    # --- collect candidate Linear weights (2D tensors ending with '.weight')
    linear_weight_keys = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim != 2:
            continue
        if str(k).endswith(".weight"):
            linear_weight_keys.append(str(k))

    if len(linear_weight_keys) == 0:
        keys_sample = list(sd.keys())[:40]
        raise KeyError(
            "Could not find any 2D Linear-like weights ending with '.weight' in scale_ckpt state_dict. "
            f"sample keys: {keys_sample}"
        )

    # --- try numeric index scheme: '<idx>.weight'
    idx_pat = re.compile(r"(\d+)\.weight$")
    idx_keys = [k for k in linear_weight_keys if idx_pat.search(k)]
    use_numeric = (len(idx_keys) > 0)

    # helper: find key by suffix match
    def find_key_by_idx(idx: int, suffix: str) -> Optional[str]:
        # exact first
        exact = f"{idx}.{suffix}"
        if exact in sd:
            return exact
        # suffix search
        pat = re.compile(rf"(?:^|.*\.){idx}\.{suffix}$")
        for kk in sd.keys():
            if pat.match(str(kk)):
                return str(kk)
        return None

    modules: List[nn.Module] = []
    remap: Dict[str, torch.Tensor] = {}

    if use_numeric:
        idxs = sorted({int(idx_pat.search(k).group(1)) for k in idx_keys})
        # build Sequential
        for li, idx in enumerate(idxs):
            wkey = find_key_by_idx(idx, "weight")
            bkey = find_key_by_idx(idx, "bias")
            if wkey is None:
                raise KeyError(f"Missing weight for idx={idx} in scale state_dict.")
            W = sd[wkey]
            out_dim, in_dim = W.shape
            modules.append(nn.Linear(in_dim, out_dim, bias=True))
            if li < len(idxs) - 1:
                modules.append(nn.GELU())
            else:
                modules.append(nn.Softplus())

            tgt = 2 * li
            remap[f"{tgt}.weight"] = sd[wkey]
            if bkey is not None and bkey in sd:
                remap[f"{tgt}.bias"] = sd[bkey]

        if logger is not None:
            logger.print(f"[INFO] scale_head: numeric-index keys detected. idxs={idxs}")

    else:
        # Fallback: sort by key name; assume it's already in forward order
        sorted_w = sorted(linear_weight_keys)
        for li, wkey in enumerate(sorted_w):
            W = sd[wkey]
            out_dim, in_dim = W.shape
            modules.append(nn.Linear(in_dim, out_dim, bias=True))
            if li < len(sorted_w) - 1:
                modules.append(nn.GELU())
            else:
                modules.append(nn.Softplus())

            tgt = 2 * li
            remap[f"{tgt}.weight"] = sd[wkey]
            bkey = wkey[:-len("weight")] + "bias"
            if bkey in sd and torch.is_tensor(sd[bkey]) and sd[bkey].ndim == 1:
                remap[f"{tgt}.bias"] = sd[bkey]

        if logger is not None:
            logger.print("[INFO] scale_head: fallback mode (no numeric indices). "
                         f"linear_layers={len(sorted_w)}")

    scale_head = nn.Sequential(*modules)
    missing, unexpected = scale_head.load_state_dict(remap, strict=False)

    if logger is not None:
        logger.print(f"[INFO] scale_head built: len={len(scale_head)} (Linear at 0,2,4..., Softplus at end)")
        logger.print(f"[INFO] scale_head load_state_dict: missing={missing} unexpected={unexpected}")

    return scale_head


# ------------------------
# Main
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--geneformer_model_dir", required=True)
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--sample_metadata_csv", required=True)
    p.add_argument("--common_genes_csv", required=True)
    p.add_argument("--topn_dir", required=True)
    p.add_argument("--full_dir", required=True)
    p.add_argument("--topn", type=int, required=True)
    p.add_argument("--log1p_ckpt", required=True)
    p.add_argument("--scale_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rep_filter", default=None,
                   help="e.g., rep3 or 3; if provided, keep only rows whose 'rep' matches this.")
    p.add_argument("--condition_filter", default=None,
                   help="If provided, keep only rows whose cell_type exactly matches this condition.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = out_dir / f"runs/predict_rawaware_nocond_top{args.topn}_{_nowstamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(run_dir / "predict.log")

    logger.print(f"[INFO] run_dir          : {run_dir}")
    logger.print(f"[INFO] device           : {args.device}")

    # ---- load log1p checkpoint ----
    ckpt = torch.load(args.log1p_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("log1p_ckpt must be a dict containing 'model_state_dict'.")

    model_state = ckpt["model_state_dict"]
    ckpt_common_genes = ckpt.get("common_genes", None)

    # load common genes from file (sanity only)
    common_genes_file = [
        x.strip() for x in Path(args.common_genes_csv).read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    if ckpt_common_genes is None:
        common_genes = common_genes_file
    else:
        common_genes = list(map(str, ckpt_common_genes))
        if len(common_genes_file) != len(common_genes):
            logger.print(f"[WARN] common_genes count differs: file={len(common_genes_file)} vs ckpt={len(common_genes)}. Using ckpt.")

    logger.print(f"[INFO] common_genes      : {len(common_genes)}")
    gene2idx = {g: i for i, g in enumerate(common_genes)}

    # ---- encoder config ----
    geneformer_model_dir = Path(args.geneformer_model_dir)
    cfg = AutoConfig.from_pretrained(str(geneformer_model_dir))
    encoder = BertModel.from_pretrained(str(geneformer_model_dir), config=cfg)

    encoder_max_len = int(getattr(cfg, "max_position_embeddings", 2048))
    seq_len_used = min(int(args.topn), encoder_max_len)
    logger.print(f"[INFO] encoder_max_len   : {encoder_max_len} | topn={args.topn} | seq_len_used={seq_len_used}")

    # ---- token dicts ----
    sym2ens, ens2id, pad_id = load_token_dicts(Path(args.pkl_dir), logger)

    # ---- infer model hyperparams from checkpoint ----
    hidden_size = int(ckpt.get("hidden_size", cfg.hidden_size))
    if bool(ckpt.get("use_condition_embedding", True)):
        raise ValueError(
            "This predictor requires a condition-free checkpoint "
            "(use_condition_embedding=False)."
        )
    aggregation_mode = str(ckpt.get("aggregation_mode", "mean"))
    activation_type = str(ckpt.get("activation_type", "softplus"))

    # infer concat_dim from input_norm weight if exists
    if "input_norm.weight" in model_state:
        concat_dim = int(model_state["input_norm.weight"].numel())
    else:
        concat_dim = hidden_size

    if "mlp.0.weight" not in model_state:
        raise KeyError("log1p_ckpt model_state_dict lacks 'mlp.0.weight'.")

    mlp_hidden = int(model_state["mlp.0.weight"].shape[0])
    mlp_in = int(model_state["mlp.0.weight"].shape[1])
    if mlp_in != concat_dim:
        logger.print(f"[WARN] concat_dim mismatch: inferred={concat_dim} but mlp_in={mlp_in}. Using mlp_in.")
        concat_dim = mlp_in

    # infer n_genes from head
    if "head_zero.weight" in model_state:
        n_genes = int(model_state["head_zero.weight"].shape[0])
        head_in = int(model_state["head_zero.weight"].shape[1])
        if head_in != mlp_hidden:
            logger.print(f"[WARN] head_in ({head_in}) != mlp_hidden ({mlp_hidden}). Using head_in for mlp_hidden.")
            mlp_hidden = head_in
    else:
        raise KeyError("log1p_ckpt model_state_dict lacks 'head_zero.weight'.")

    use_expr_fuser = "expr_fuser.0.weight" in model_state
    expr_fuser_in = expr_fuser_out = None
    if use_expr_fuser:
        w = model_state["expr_fuser.0.weight"]
        expr_fuser_out = int(w.shape[0])
        expr_fuser_in = int(w.shape[1])

    use_mlp_ln = ("mlp.2.weight" in model_state and getattr(model_state["mlp.2.weight"], "ndim", 0) == 1)

    logger.print(f"[INFO] hidden_size       : {hidden_size}")
    logger.print("[INFO] condition embedding: DISABLED")
    logger.print(f"[INFO] concat_dim        : {concat_dim}")
    logger.print(f"[INFO] mlp_hidden        : {mlp_hidden}")
    logger.print(f"[INFO] aggregation_mode  : {aggregation_mode}")
    logger.print(f"[INFO] activation_type   : {activation_type}")

    # ---- sample metadata ----
    meta = pd.read_csv(args.sample_metadata_csv)
    meta.columns = [str(c).strip() for c in meta.columns]

    if "sample_id" not in meta.columns or "cell_type" not in meta.columns:
        raise ValueError(f"sample_metadata_csv must contain at least sample_id and cell_type. columns={list(meta.columns)}")

    if "rep" not in meta.columns:
        meta = meta.copy()
        meta["rep"] = meta["sample_id"].astype(str).str.extract(r"rep(\d+)")[0]

    # detect full/topN columns (flexible)
    full_col = None
    for cand in ("full_csv", "full_file", "full_path", "fullpath", "full"):
        if cand in meta.columns:
            full_col = cand
            break
    if full_col is None:
        for c in meta.columns:
            if str(c).lower().startswith("full"):
                full_col = c
                break

    topn_col = None
    for cand in (
        f"top{args.topn}_csv",
        f"top{args.topn}_file",
        f"top{args.topn}",
        "topn_csv",
        "topN_csv",
        "topn_file",
        "topN_file",
        "topn",
    ):
        if cand in meta.columns:
            topn_col = cand
            break
    if topn_col is None:
        for c in meta.columns:
            cl = str(c).lower()
            if cl.startswith("top") and str(args.topn) in cl:
                topn_col = c
                break

    if full_col is None or topn_col is None:
        raise ValueError(
            "sample_metadata_csv must contain full-file and topN-file columns. "
            f"found full_col={full_col}, topn_col={topn_col}. columns={list(meta.columns)}"
        )

    meta = meta.copy()
    meta["full_csv"] = meta[full_col].astype(str)
    meta["topn_csv"] = meta[topn_col].astype(str)
    meta["rep"] = meta["rep"].astype(str)

    if args.rep_filter:
        rf = str(args.rep_filter)
        rf_num = rf.replace("rep", "")
        n0 = len(meta)
        meta = meta[meta["rep"].str.replace("rep", "", regex=False) == rf_num].copy()
        logger.print(f"[INFO] rep_filter={args.rep_filter} : {n0} -> {len(meta)}")

    if args.condition_filter:
        cf = str(args.condition_filter)
        n0 = len(meta)
        meta = meta[meta["cell_type"].astype(str) == cf].copy()
        logger.print(f"[INFO] condition_filter={cf} : {n0} -> {len(meta)}")
        if len(meta) == 0:
            raise ValueError(f"No metadata rows matched condition_filter={cf}")

    # resolve paths
    topn_dir = Path(args.topn_dir)
    full_dir = Path(args.full_dir)

    rows: List[SampleRow] = []
    for _, row in meta.iterrows():
        topn_csv = str(row["topn_csv"])
        full_csv = str(row["full_csv"])
        topn_path = Path(topn_csv)
        full_path = Path(full_csv)
        if not topn_path.is_file():
            topn_path = topn_dir / topn_csv
        if not full_path.is_file():
            full_path = full_dir / full_csv
        if not topn_path.is_file():
            raise FileNotFoundError(f"TopN CSV not found: {topn_path}")
        if not full_path.is_file():
            raise FileNotFoundError(f"Full CSV not found: {full_path}")
        rows.append(SampleRow(
            sample_id=str(row["sample_id"]),
            cell_type=str(row["cell_type"]),
            rep=str(row["rep"]),
            topn_csv=str(topn_path),
            full_csv=str(full_path),
        ))

    # cell_type remains metadata only and is NOT supplied to the model.

    # dataset/dataloader
    ds = ZFBulkDatasetV6Infer(
        rows=rows,
        common_genes=common_genes,
        gene2idx=gene2idx,
        symbol2ensembl=sym2ens,
        ensembl2id=ens2id,
        pad_id=pad_id,
        topn=args.topn,
        max_seq_len=seq_len_used,
    )

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # build log1p model
    model_v5 = ZFReconstructionModelV5NoCondition(
        encoder=encoder,
        n_genes=n_genes,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        concat_dim=concat_dim,
        aggregation_mode=aggregation_mode,
        activation_type=activation_type,
        dropout=0.10,
        use_expr_fuser=use_expr_fuser,
        expr_fuser_in=expr_fuser_in,
        expr_fuser_out=expr_fuser_out,
        use_mlp_layernorm=use_mlp_ln,
    )
    model_v5.load_state_dict(model_state, strict=True)

    # ---- load scale head ----
    scale_obj = torch.load(args.scale_ckpt, map_location="cpu", weights_only=False)
    scale_sd = _extract_state_dict(scale_obj)
    scale_sd = _normalize_scale_keys(scale_sd)

    scale_head = build_scale_head_from_state_dict(scale_sd, logger=logger)

    # device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model_v5.to(device).eval()
    scale_head.to(device).eval()

    # output layout
    out_base = out_dir / f"zF_gen1_v6_top{args.topn}"
    out_base.mkdir(parents=True, exist_ok=True)
    logger.print(f"[INFO] out_base         : {out_base}")

    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    with torch.no_grad():
        for batch in loader:
            B = batch["input_ids"].shape[0]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            expr_norm_seq = batch["expr_norm_seq"].to(device)

            target_raw = batch["target_raw"].to(device)

            # log1p_hat
            log1p_hat = model_v5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                expr_norm_seq=expr_norm_seq,
            )

            # scale input = concat + topn_raw_log
            concat = model_v5.get_concat_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                expr_norm_seq=expr_norm_seq,
            )
            topn_raw_log = batch["topn_raw_log"].to(device)  # (B,1)
            scale_in = torch.cat([concat, topn_raw_log], dim=1)

            scale = scale_head(scale_in)  # (B,1) or (B,)
            if scale.ndim == 2 and scale.shape[1] == 1:
                scale = scale.squeeze(1)

            raw_pred_wo_overwrite = torch.expm1(log1p_hat) * scale.unsqueeze(1)

            # overwrite TopN genes into raw_pred_final
            overwrite_idx = batch["overwrite_idx"].to(device)
            overwrite_val = batch["overwrite_val"].to(device)

            raw_pred_final = raw_pred_wo_overwrite.clone()
            for j in range(B):
                idx = overwrite_idx[j]
                val = overwrite_val[j]
                m = idx >= 0
                if m.any():
                    raw_pred_final[j, idx[m]] = val[m]

            # write per-sample CSV
            for j in range(B):
                sample_id = batch["sample_id"][j]
                y_true = target_raw[j].detach().cpu().numpy().astype(np.float32)
                y_pred_wo = raw_pred_wo_overwrite[j].detach().cpu().numpy().astype(np.float32)
                y_final = raw_pred_final[j].detach().cpu().numpy().astype(np.float32)

                df_out = pd.DataFrame({
                    "gene": common_genes,
                    "raw_true": y_true,
                    "raw_pred_wo_overwrite": y_pred_wo,
                    "raw_pred": y_final,          # final (backwards compatible)
                    "raw_pred_final": y_final,    # alias
                })

                out_csv = out_base / f"{sample_id}_pred_vs_true_v6_raw_final.csv"
                df_out.to_csv(out_csv, index=False)

    logger.print("[DONE] Prediction finished.")
    logger.close()


if __name__ == "__main__":
    main()
