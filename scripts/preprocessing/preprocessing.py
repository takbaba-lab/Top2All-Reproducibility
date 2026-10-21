"""Shared Top2All input preprocessing."""
import numpy as np

def require(ok, message):
    if not ok: raise RuntimeError(message)

def preprocess(genes, expression, sym2ens, ens2id, nominal_n, max_len=2048):
    """No refill. Filter paired genes/values, cap, then normalize and sum."""
    require(0 < nominal_n <= 2000 and max_len >= nominal_n, 'Unsupported N/cap')
    genes = list(map(str, genes))[:nominal_n]
    raw = np.asarray(expression, dtype=np.float32)[:nominal_n]
    require(len(genes) == len(raw) == nominal_n, 'Wrong nominal input length')
    require(len(set(genes)) == nominal_n, 'Duplicate nominal input gene')
    require(np.isfinite(raw).all() and (raw >= 0).all(), 'Invalid expression')
    keep = [i for i, g in enumerate(genes) if str(sym2ens.get(g, g)) in ens2id]
    keep = keep[:max_len]
    require(bool(keep), 'No tokenizable genes')
    kept_genes = [genes[i] for i in keep]
    ids = np.asarray([int(ens2id[str(sym2ens.get(g, g))]) for g in kept_genes], dtype=np.int64)
    values = raw[keep]
    log = np.log1p(values)
    sd = float(log.std())
    normalized = ((log - float(log.mean())) / (sd if sd >= 1e-8 else 1.0)).astype(np.float32)
    total = float(values.sum())
    return dict(genes=kept_genes, ids=ids, raw=values, log1p=log,
                normalized=normalized, total=total, log_total=float(np.log(total + 1e-6)))
