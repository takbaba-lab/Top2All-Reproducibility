#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Top2All zF_gen1 v5 (log1p space) checkpoint.

This script is a clean, non-notebook replacement for NB_02 train (v5-style):
- Load zF bulk sample metadata
- Build dataset from TopN raw CSV (Gene, Expression) and full raw CSV (Gene, Expression)
- Freeze Geneformer encoder
- Train v5 reconstruction head to predict log1p(full) from TopN inputs
- OOF mode: exclude one replicate completely and save the fixed-epoch LAST checkpoint (log1p-only; no scale_head)

Designed for Genkai HPC offline runs (local Geneformer model dir).
"""

from __future__ import print_function
import argparse
import os
import re
from datetime import datetime

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

import sys
import json
import platform
import socket
import getpass
import atexit

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    if path and (not os.path.exists(path)):
        os.makedirs(path)

class Tee(object):
    """Run metadata and model setup."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_run_logging(out_ckpt_dir, run_prefix, args):
    """Run metadata and model setup."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_ckpt_dir, "runs", f"{run_prefix}_{run_id}")
    ensure_dir(run_dir)

    log_path = os.path.join(run_dir, "train.log")
    meta_path = os.path.join(run_dir, "run_meta.json")

    run_meta = {
        "run_id": run_id,
        "run_prefix": run_prefix,
        "started_at": run_id,
        "cmd": " ".join(sys.argv),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "args": vars(args) if hasattr(args, "__dict__") else str(args),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    log_fh = open(log_path, "w", encoding="utf-8")

    import atexit
    def _safe_close_log():
        try:
            log_fh.flush()
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass
    atexit.register(_safe_close_log)


    #
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    print("[INFO] Run directory:", run_dir)
    print("[INFO] Log file      :", log_path)
    print("[INFO] Meta JSON     :", meta_path)

    return run_dir, run_id, run_meta


def parse_rep(sample_id):
    m = re.search(r"_rep(\d+)", str(sample_id))
    return int(m.group(1)) if m else None


def load_common_genes(path):
    """
    Load a headerless one-gene-per-line reference gene universe.
    """
    with open(path, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]

    if not genes:
        raise ValueError(f"Gene list is empty: {path}")

    if len(genes) != len(set(genes)):
        from collections import Counter
        dup = [g for g, n in Counter(genes).items() if n > 1]
        raise ValueError(
            f"Gene list contains duplicate identifiers: {dup[:20]}"
        )

    return genes


def load_token_dicts(pkl_dir):
    import pickle
    symbol2ensembl_path = os.path.join(pkl_dir, "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl")
    ensembl2id_path = os.path.join(pkl_dir, "MLM-re_token_dictionary_v1.pkl")
    if (not os.path.exists(symbol2ensembl_path)) or (not os.path.exists(ensembl2id_path)):
        raise FileNotFoundError("Missing token dict pkl in: %s" % pkl_dir)

    with open(symbol2ensembl_path, "rb") as f:
        symbol2ensembl = pickle.load(f)
    with open(ensembl2id_path, "rb") as f:
        ensembl2id = pickle.load(f)

        #
    if isinstance(ensembl2id, dict):
        ensembl2id = {str(k): int(v) for k, v in ensembl2id.items()}

    # ----------------------------
    # pad_id resolution
    # ----------------------------
    pad_id = None

    #
    for pad_key in ("<pad>", "[PAD]", "pad", "PAD"):
        if pad_key in ensembl2id:
            pad_id = int(ensembl2id[pad_key])
            break

    if pad_id is None:
        #
        for k, v in ensembl2id.items():
            if isinstance(k, str) and "pad" in k.lower():
                pad_id = int(v)
                break

    if pad_id is None:
        #
        print("[WARN] PAD token not found in token dict. Will resolve pad_id from tokenizer/config later.")
    else:
        print(f"[INFO] pad token found in token dict: pad_id={pad_id}")

    return symbol2ensembl, ensembl2id, pad_id


def gene_list_to_token_ids(gene_symbols, ensembl2id, symbol2ensembl, max_len):
    """
    Convert a list of gene symbols (TopN ranked) into Geneformer token ids.

    IMPORTANT: This matches your NB implementation:
      - no [CLS]/[SEP] are added
      - no padding here (padding is handled in collate_fn)
    """
    token_ids = []
    for g in gene_symbols:
        g = str(g)
        ens = symbol2ensembl.get(g, None)
        if ens is None:
            continue
        tid = ensembl2id.get(ens, None)
        if tid is None:
            continue
        token_ids.append(int(tid))
        if (max_len is not None) and (len(token_ids) >= int(max_len)):
            break
    return token_ids


# -------------------------
# Dataset / Collate
# -------------------------
class ZFBulkDatasetV5(Dataset):
    def __init__(self, meta_df, topn_dir, full_dir, common_genes, cell_type_to_id,
                 topn_field, symbol2ensembl, ensembl2id, pad_id,
                 max_seq_len=2048):
        self.meta = meta_df.reset_index(drop=True)
        self.topn_dir = topn_dir
        self.full_dir = full_dir
        self.common_genes = list(common_genes)
        self.cell_type_to_id = dict(cell_type_to_id)
        self.topn_field = topn_field

        self.symbol2ensembl = symbol2ensembl
        self.ensembl2id = ensembl2id
        self.pad_id = pad_id
        self.max_seq_len = int(max_seq_len)

    def __len__(self):
        return len(self.meta)

    def _read_csv_gene_expr(self, path):
        df = pd.read_csv(path)
        # expect Gene + Expression, but tolerate variants
        gene_col = None
        expr_col = None
        for c in ["Gene", "gene", "GENE"]:
            if c in df.columns:
                gene_col = c
                break
        for c in ["Expression", "expression", "expr", "count", "Counts", "counts"]:
            if c in df.columns:
                expr_col = c
                break
        if gene_col is None or expr_col is None:
            raise ValueError("CSV must contain Gene and Expression columns: %s (cols=%s)" % (path, list(df.columns)))
        df = df[[gene_col, expr_col]].copy()
        df.columns = ["Gene", "Expression"]
        df["Gene"] = df["Gene"].astype(str)
        df["Expression"] = pd.to_numeric(df["Expression"], errors="coerce").fillna(0.0)
        return df

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        sample_id = row["sample_id"]
        cell_type = row["cell_type"]
        cell_type_id = self.cell_type_to_id[cell_type]

        topn_rel = row[self.topn_field]
        full_rel = row["full_file"]
        topn_path = topn_rel if os.path.isabs(str(topn_rel)) else os.path.join(self.topn_dir, str(topn_rel))
        full_path = full_rel if os.path.isabs(str(full_rel)) else os.path.join(self.full_dir, str(full_rel))

        top_df = self._read_csv_gene_expr(topn_path)
        prepared = preprocess(top_df["Gene"], top_df["Expression"],
                                self.symbol2ensembl, self.ensembl2id,
                                int(self.topn_field.removeprefix("top").removesuffix("_file")),
                                self.max_seq_len)
        nominal_n = int(self.topn_field.removeprefix("top").removesuffix("_file"))
        length = len(prepared["ids"])
        token_ids = np.full(nominal_n, self.pad_id, dtype=np.int64)
        token_ids[:length] = prepared["ids"]
        expr_norm_seq = np.zeros(nominal_n, dtype=np.float32)
        expr_norm_seq[:length] = prepared["normalized"]

        # Full target (raw and log1p) aligned to common_genes
        full_df = self._read_csv_gene_expr(full_path)
        exp_map = dict(zip(full_df["Gene"].tolist(), full_df["Expression"].values.astype(np.float32)))
        target_raw = np.array([exp_map.get(g, 0.0) for g in self.common_genes], dtype=np.float32)
        target_log1p = np.log1p(target_raw).astype(np.float32)

        return {
            "sample_id": sample_id,
            "cell_type": cell_type,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "expr_norm_seq": torch.tensor(expr_norm_seq, dtype=torch.float32),
            "cell_type_ids": torch.tensor(cell_type_id, dtype=torch.long),
            "target_log1p": torch.tensor(target_log1p, dtype=torch.float32),
        }


def collate_fn_v5(batch, pad_id=0):
    # pad input_ids to max_len (they are already max_seq_len but keep generic)
    max_len = max([len(x["input_ids"]) for x in batch])

    input_ids = []
    attention_mask = []
    expr_norm_seq = []
    cell_type_ids = []
    target_log1p = []
    sample_ids = []
    cell_types = []

    for item in batch:
        ids = item["input_ids"]
        L = len(ids)
        if L < max_len:
            pad = torch.full((max_len - L,), int(pad_id), dtype=torch.long)
            ids = torch.cat([ids, pad], dim=0)
        else:
            ids = ids[:max_len]
        mask = (ids != int(pad_id)).long()

        expr = item["expr_norm_seq"]
        if len(expr) < max_len:
            expr = torch.cat([expr, torch.zeros((max_len - len(expr),), dtype=torch.float32)], dim=0)
        else:
            expr = expr[:max_len]

        input_ids.append(ids)
        attention_mask.append(mask)
        expr_norm_seq.append(expr)
        cell_type_ids.append(item["cell_type_ids"])
        target_log1p.append(item["target_log1p"])
        sample_ids.append(item["sample_id"])
        cell_types.append(item["cell_type"])

    return {
        "sample_id": sample_ids,
        "cell_type": cell_types,
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_mask, dim=0),
        "expr_norm_seq": torch.stack(expr_norm_seq, dim=0),
        "cell_type_ids": torch.stack(cell_type_ids, dim=0),
        "target_log1p": torch.stack(target_log1p, dim=0),
    }


# -------------------------
# Model (v5)
# -------------------------
class ZFReconstructionModelV5(nn.Module):
    def __init__(self, encoder, hidden_size, num_cell_types, num_genes,
                 celltype_emb_dim=32, aggregation_mode="mean", activation_type="softplus"):
        super(ZFReconstructionModelV5, self).__init__()
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
            raise ValueError("Unknown activation_type: %s" % activation_type)

    def forward(self, input_ids, attention_mask, cell_type_ids, expr_norm_seq):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, L, H)

        expr_norm_expand = expr_norm_seq.unsqueeze(-1)  # (B, L, 1)
        fused_input = torch.cat([hidden_states, expr_norm_expand], dim=-1)  # (B, L, H+1)
        fused_hidden = self.expr_fuser(fused_input)  # (B, L, H)

        if self.aggregation_mode == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = fused_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            cell_emb = sum_hidden / lengths
        else:
            cell_emb = fused_hidden[:, 0, :]

        ct_emb = self.celltype_emb(cell_type_ids)
        concat = torch.cat([cell_emb, ct_emb], dim=-1)
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
        hidden_states = outputs.last_hidden_state
        expr_norm_expand = expr_norm_seq.unsqueeze(-1)
        fused_input = torch.cat([hidden_states, expr_norm_expand], dim=-1)
        fused_hidden = self.expr_fuser(fused_input)

        if self.aggregation_mode == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            masked_hidden = fused_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            cell_emb = sum_hidden / lengths
        else:
            cell_emb = fused_hidden[:, 0, :]

        ct_emb = self.celltype_emb(cell_type_ids)
        concat = torch.cat([cell_emb, ct_emb], dim=-1)
        concat = self.input_norm(concat)
        return concat


# -------------------------
# Loss (v5)
# -------------------------
def build_loss(device, alpha_weight, pos_weight_zero, lambda_zero):
    pos_weight_tensor = torch.tensor([float(pos_weight_zero)], dtype=torch.float32, device=device)
    bce_zero = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def compute_loss_v5(true_log1p, zero_logits, y_pos, log1p_hat):
        weight = 1.0 + float(alpha_weight) * true_log1p
        mse = ((log1p_hat - true_log1p) ** 2) * weight
        mse = mse.mean()

        is_nonzero = (true_log1p > 0).float()
        zero_logits_flat = zero_logits.view(-1)
        is_nonzero_flat = is_nonzero.view(-1)
        loss_zero = bce_zero(zero_logits_flat, is_nonzero_flat)

        total_loss = mse + float(lambda_zero) * loss_zero
        return total_loss, mse, loss_zero

    return compute_loss_v5


# -------------------------
# Train / Eval
# -------------------------
def run_epoch(model, loader, compute_loss_v5, optimizer, device, train=True, clip_grad=1.0):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_zero = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        expr_norm_seq = batch["expr_norm_seq"].to(device)
        cell_type_ids = batch["cell_type_ids"].to(device)
        target_log1p = batch["target_log1p"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            zero_logits, y_pos, log1p_hat = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cell_type_ids=cell_type_ids,
                expr_norm_seq=expr_norm_seq,
            )
            loss, mse, loss_zero = compute_loss_v5(
                true_log1p=target_log1p,
                zero_logits=zero_logits,
                y_pos=y_pos,
                log1p_hat=log1p_hat,
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
                optimizer.step()

        total_loss += float(loss.item())
        total_mse += float(mse.item())
        total_zero += float(loss_zero.item())
        n_batches += 1

    if n_batches < 1:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, total_mse / n_batches, total_zero / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geneformer_model_dir", required=True)
    ap.add_argument("--pkl_dir", required=True)
    ap.add_argument("--sample_metadata_csv", required=True)
    ap.add_argument("--common_genes_csv", required=True)
    ap.add_argument("--topn_dir", required=True)
    ap.add_argument("--full_dir", required=True)
    ap.add_argument("--topn", type=int, required=True, help="TopN size (e.g., 1000)")
    ap.add_argument("--out_ckpt_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--aggregation_mode", default="mean", choices=["mean", "cls"])
    ap.add_argument("--activation_type", default="softplus", choices=["softplus", "relu"])
    ap.add_argument("--alpha_weight", type=float, default=0.3)
    ap.add_argument("--pos_weight_zero", type=float, default=3.0)
    ap.add_argument("--lambda_zero", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--heldout_rep", type=int, required=True, choices=[1, 2, 3],
        help="OOF test replicate. This replicate is excluded completely from training and is not evaluated during training."
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    ensure_dir(args.out_ckpt_dir)

    run_prefix = "log1p_top%d" % int(args.topn)
    run_dir, run_id, run_meta = setup_run_logging(args.out_ckpt_dir, run_prefix, args)


    # metadata
    meta = pd.read_csv(args.sample_metadata_csv)
    required_cols = ["sample_id", "cell_type", "full_file"]
    for c in required_cols:
        if c not in meta.columns:
            raise KeyError("sample_metadata_csv must contain column: %s" % c)

    topn_field = "top%d_file" % int(args.topn)
    if topn_field not in meta.columns:
        raise KeyError("sample_metadata_csv lacks %s. Columns=%s" % (topn_field, list(meta.columns)))

    # OOF split: held-out replicate is a true test set and is NEVER used during training.
    meta["rep"] = meta["sample_id"].apply(parse_rep)
    heldout_rep = int(args.heldout_rep)
    train_meta = meta[meta["rep"] != heldout_rep].copy()
    test_meta = meta[meta["rep"] == heldout_rep].copy()
    if len(train_meta) < 1 or len(test_meta) < 1:
        raise ValueError(
            "Could not make OOF split for heldout_rep=%d. Check sample_id format." % heldout_rep
        )

    # ---- update run_meta with dataset info ----
    run_meta.update({
        "evaluation_design": "3-fold OOF by biological replicate; held-out replicate not used during training",
        "heldout_rep": heldout_rep,
        "checkpoint_selection": "fixed_epoch_last",
        "topn_field": topn_field,
        "n_train_samples": int(len(train_meta)),
        "n_test_samples": int(len(test_meta)),
        "cell_types": sorted(meta["cell_type"].unique().tolist()),
        "train_sample_ids": train_meta["sample_id"].tolist(),
        "test_sample_ids": test_meta["sample_id"].tolist(),
    })

    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print("[RUN META] cell_types:", run_meta["cell_types"])
    print("[RUN META] heldout_rep:", heldout_rep)
    print("[RUN META] n_train / n_test:", run_meta["n_train_samples"], "/", run_meta["n_test_samples"])
    print("[RUN META] test set will NOT be evaluated during training.")

    # cell type mapping (use all)
    cell_types = sorted(meta["cell_type"].unique().tolist())
    cell_type_to_id = {ct: i for i, ct in enumerate(cell_types)}
    id_to_cell_type = {i: ct for ct, i in cell_type_to_id.items()}

    common_genes = load_common_genes(args.common_genes_csv)

    # token dicts
    symbol2ensembl, ensembl2id, pad_id = load_token_dicts(args.pkl_dir)

    # dataloaders
    train_dataset = ZFBulkDatasetV5(
        meta_df=train_meta, topn_dir=args.topn_dir, full_dir=args.full_dir,
        common_genes=common_genes, cell_type_to_id=cell_type_to_id, topn_field=topn_field,
        symbol2ensembl=symbol2ensembl, ensembl2id=ensembl2id, pad_id=pad_id,
        max_seq_len=args.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_fn_v5(b, pad_id=pad_id),
    )

    print("[INFO] train_loader batches:", len(train_loader))
    print("[INFO] held-out test samples are not loaded during training:", len(test_meta))

    # encoder
    config = AutoConfig.from_pretrained(args.geneformer_model_dir)
    encoder = AutoModel.from_pretrained(args.geneformer_model_dir, config=config)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.to(device)
    encoder.eval()

    num_cell_types = len(cell_type_to_id)
    num_genes = len(common_genes)

    model = ZFReconstructionModelV5(
        encoder=encoder,
        hidden_size=config.hidden_size,
        num_cell_types=num_cell_types,
        num_genes=num_genes,
        celltype_emb_dim=32,
        aggregation_mode=args.aggregation_mode,
        activation_type=args.activation_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    compute_loss_v5 = build_loss(
        device=device,
        alpha_weight=args.alpha_weight,
        pos_weight_zero=args.pos_weight_zero,
        lambda_zero=args.lambda_zero,
    )

    last_train_loss = None
    last_train_mse = None
    last_train_zero = None

    for epoch in range(1, int(args.epochs) + 1):
        print("===== Epoch %d/%d =====" % (epoch, int(args.epochs)))

        train_loss, train_mse, train_zero = run_epoch(
            model, train_loader, compute_loss_v5, optimizer, device, train=True
        )
        last_train_loss, last_train_mse, last_train_zero = train_loss, train_mse, train_zero
        print("  train_loss: %.6f (MSE=%.6f, zero=%.6f)" % (train_loss, train_mse, train_zero))

    # Fixed-epoch OOF checkpoint: held-out test data were never used for checkpoint selection.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_obj = {
        "pipeline_id": PIPELINE,
        "preprocess_sha256": sha(BUNDLE / "preprocessing.py"),
        "model_state_dict": model.state_dict(),
        "cell_type_to_id": cell_type_to_id,
        "id_to_cell_type": id_to_cell_type,
        "common_genes": common_genes,
        "hidden_size": int(config.hidden_size),
        "celltype_emb_dim": 32,
        "aggregation_mode": args.aggregation_mode,
        "topn_field": topn_field,
        "timestamp": timestamp,
        "loss_params": {
            "ALPHA_WEIGHT": float(args.alpha_weight),
            "POS_WEIGHT_ZERO": float(args.pos_weight_zero),
            "LAMBDA_ZERO": float(args.lambda_zero),
        },
        "activation_type": args.activation_type,
        "heldout_rep": int(args.heldout_rep),
        "checkpoint_selection": "fixed_epoch_last",
        "epochs_completed": int(args.epochs),
        "final_train_loss": float(last_train_loss),
        "final_train_mse": float(last_train_mse),
        "final_train_zero_loss": float(last_train_zero),
        "note": "Top2All zF_gen1 v5 OOF log1p checkpoint. Encoder frozen. Held-out replicate never used during training/checkpoint selection.",
        "run_id": run_id,
        "run_meta": run_meta,
    }
    last_ckpt_path = os.path.join(
        args.out_ckpt_dir,
        "zf_gen1_v5_top%d_oof_test_rep%d_last.pt" % (int(args.topn), int(args.heldout_rep))
    )
    torch.save(ckpt_obj, last_ckpt_path)

    print("[DONE] Training finished.")
    print("[DONE] Fixed-epoch OOF checkpoint:", last_ckpt_path)
    print("[DONE] Held-out rep was never evaluated during training:", int(args.heldout_rep))


if __name__ == "__main__":
    main()
