#!/usr/bin/env python3
"""Decompose 8-condition x 3-replicate LOCO predictions (numpy + pandas only).

Default: transform raw expression with log1p BEFORE averaging/centering.
Fixed non-TopN = common prediction universe minus UNION of first N input genes.
Residual truth and prediction are centered by their OWN condition means.
Zero residual is a diagnostic reference, not a deployable unknown-condition model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUFFIX = '_pred_vs_true_v6_raw_final.csv'
METRICS = ['pearson', 'spearman', 'rmse', 'mae', 'r2']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, columns, gene_col=None, allow_duplicate_genes=False):
    df = pd.read_csv(path, keep_default_na=False, dtype={gene_col: str} if gene_col else None)
    require(set(columns) <= set(df), f'{path}: required columns {columns}')
    if gene_col:
        ids = df[gene_col]
        require(ids.str.strip().ne('').all() and ids.eq(ids.str.strip()).all(), f'{path}: invalid gene IDs')
        if not allow_duplicate_genes:
            require(not ids.duplicated().any(), f'{path}: duplicate gene IDs')
    return df


def numeric(df, columns, path):
    a = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    require(np.isfinite(a).all() and (a >= 0).all(), f'{path}: negative or non-finite expression')
    return a


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(x, y, eps=1e-12):
    """Columns are independent vectors; undefined correlations/R2 remain NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    require(x.shape == y.shape and x.shape[0] > 0, 'Empty or mismatched metric inputs')
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    sx, sy = np.mean(xc**2, axis=0), np.mean(yc**2, axis=0)
    def corr(a, b, valid):
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.sum(a*b, axis=0) / np.sqrt(np.sum(a*a, axis=0)*np.sum(b*b, axis=0))
        return np.where(valid, np.clip(r, -1, 1), np.nan)
    variable = (np.sqrt(sx) > eps) & (np.sqrt(sy) > eps)
    mse = np.mean((y-x)**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(np.sqrt(sx) > eps, 1-mse/sx, np.nan)
    return pd.DataFrame(dict(
        n_observations=x.shape[0], true_sd=np.sqrt(sx), pred_sd=np.sqrt(sy),
        pearson=corr(x, y, variable),
        spearman=corr(pd.DataFrame(x).rank().to_numpy(), pd.DataFrame(y).rank().to_numpy(), variable),
        rmse=np.sqrt(mse), mae=np.mean(abs(y-x), axis=0), r2=r2))


def quartiles(values, eps):
    """Value quantiles of positive values; equal values never split arbitrarily.

    Repeated boundaries can leave empty Q bins, which remain in the output.
    """
    values = np.asarray(values)
    labels = np.full(values.shape, 'Zero', dtype=object)
    active = values > eps
    cuts = np.quantile(values[active], [.25, .5, .75]) if active.any() else np.full(3, np.nan)
    labels[active] = np.array(['Q1', 'Q2', 'Q3', 'Q4'])[np.searchsorted(cuts, values[active], side='left')]
    return labels, cuts


def decompose(x, y, conditions):
    names = sorted(set(conditions))
    index = np.array([names.index(c) for c in conditions])
    xm = np.stack([x[index == i].mean(axis=0) for i in range(len(names))])
    ym = np.stack([y[index == i].mean(axis=0) for i in range(len(names))])
    return names, index, xm, ym, x-xm[index], y-ym[index]


def summarize(frame, groups):
    rows = []
    for label, mask in groups:
        sub = frame.loc[mask]
        for metric in METRICS:
            v = sub[metric].dropna()
            rows.append(dict(group=label, metric=metric, n_units=len(sub), n_valid=len(v),
                             n_undefined=len(sub)-len(v), mean=v.mean(), median=v.median(),
                             q25=v.quantile(.25), q75=v.quantile(.75)))
    return pd.DataFrame(rows)


def evaluate(a):
    root = a.root.resolve()
    def resolve(p):
        p = Path(p)
        return p.resolve() if p.is_absolute() else (root / p).resolve()
    def input_path(value, directory):
        p = resolve(value)
        return p if p.is_file() else directory / str(value)
    meta_path, base, topdir = resolve(a.metadata_csv), resolve(a.loco_base_dir), resolve(a.topn_dir)
    out = resolve(a.out_dir) if a.out_dir else base / 'evaluation_decomposition'
    meta = read(meta_path, ['sample_id', a.condition_column]).sort_values('sample_id').reset_index(drop=True)
    require(len(meta) == 24 and not meta.sample_id.duplicated().any(), 'Expected 24 unique samples')
    for col in ['sample_id', a.condition_column]:
        meta[col] = meta[col].astype(str)
        require(meta[col].str.strip().ne('').all(), f'Blank {col}')
        require(not meta[col].str.contains(r'[/\\]').any(), f'Path separator in {col}')
    reps = meta.sample_id.str.extract(r'_rep([123])$')[0]
    require(reps.notna().all(), 'Sample IDs must end in _rep1, _rep2, _rep3')
    if 'rep' in meta:
        require(meta.rep.astype(str).str.replace(r'^rep', '', regex=True).eq(reps).all(), 'rep disagrees with sample_id')
    meta['rep'] = reps.astype(int)
    counts = pd.crosstab(meta[a.condition_column], meta.rep)
    require(counts.shape == (8, 3) and (counts.to_numpy() == 1).all(), 'Expected 8 conditions x 3 unique replicates')
    candidates = [f'top{a.topn}_csv', f'top{a.topn}_file', f'top{a.topn}', 'topn_csv', 'topN_csv', 'topn_file', 'topN_file', 'topn']
    topcol = next((c for c in candidates if c in meta), None)
    require(topcol is not None, f'Metadata needs a TopN file column: {candidates}')
    fullcol = next((c for c in ['full_csv', 'full_file', 'full_path', 'fullpath', 'full'] if c in meta), None)
    require(not a.full_dir or fullcol is not None, '--full_dir requires a full-file column in metadata')
    # Search ONLY each held-out fold prediction directory; duplicates are fatal.
    paths = {}
    for condition, group in meta.groupby(a.condition_column):
        pred_dir = base / f'test_{condition}' / 'prediction'
        files = sorted(pred_dir.rglob('*' + SUFFIX))
        expected = {sid + SUFFIX for sid in group.sample_id}
        require({p.name for p in files} == expected and len(files) == 3,
                f'{pred_dir}: expected exactly {sorted(expected)}; found {[str(p) for p in files]}')
        paths.update({p.name[:-len(SUFFIX)]: p for p in files})
    union, universe, matrices, manifest, qc = set(), None, [], [], []
    for _, row in meta.iterrows():
        sid = row.sample_id
        pp, tp = paths[sid], input_path(row[topcol], topdir)
        top = read(tp, ['Gene'], 'Gene').iloc[:a.topn]
        require(len(top) == a.topn, f'{tp}: fewer than {a.topn} genes')
        union.update(top.Gene)
        pred = read(pp, ['gene', 'raw_true', 'raw_pred_wo_overwrite'], 'gene')
        current = set(pred.gene)
        if universe is None:
            universe = current
        require(current == universe, f'{pp}: gene universe differs between samples')
        require(bool(set(top.Gene) & current), f'{tp}: no TopN overlap with prediction genes')
        pred = pred.set_index('gene').sort_index()
        vals = numeric(pred, ['raw_true', 'raw_pred_wo_overwrite'], pp)
        entry = dict(sample_id=sid, condition=row[a.condition_column], prediction_path=str(pp),
                     prediction_sha256=sha(pp), topn_path=str(tp), topn_sha256=sha(tp))
        q = dict(sample_id=sid, n_genes=len(pred), truth_qc='not_checked_against_full')
        if a.full_dir:
            fp = input_path(row[fullcol], resolve(a.full_dir))
            full = read(fp, ['Gene', 'Expression'], 'Gene', allow_duplicate_genes=True)
            numeric(full, ['Expression'], fp)
            duplicate_rows = int(full.Gene.duplicated().sum())
            duplicate_ids = int(full.loc[full.Gene.duplicated(keep=False), 'Gene'].nunique())
            # Predictor uses dict(zip(Gene, Expression)): last occurrence wins.
            # Do not sum/average duplicates; that would change its truth target.
            full = full.drop_duplicates('Gene', keep='last').set_index('Gene')
            q.update(full_duplicate_rows_removed=duplicate_rows,
                     full_duplicate_gene_ids=duplicate_ids, full_duplicate_policy='last_row_wins')
            if duplicate_rows:
                print(f'[QC] {sid}: full CSV has {duplicate_ids} duplicated gene IDs; '
                      f'using last rows as in predictor ({duplicate_rows} earlier rows removed)')
            # Match predictor: genes absent from the full input map to zero.
            truth = full.Expression.reindex(pred.index, fill_value=0).to_numpy(dtype=float)
            mismatch = ~np.isclose(vals[:, 0], truth, rtol=a.truth_rtol, atol=a.truth_atol)
            q.update(truth_qc='pass' if not mismatch.any() else 'FAIL',
                     n_mismatches=int(mismatch.sum()), max_abs_difference=float(abs(vals[:, 0]-truth).max()),
                     n_genes_absent_from_full=int((~pred.index.isin(full.index)).sum()))
            require(not mismatch.any(), f'{sid}: raw_true differs from full CSV at {mismatch.sum()} genes')
            entry.update(full_path=str(fp), full_sha256=sha(fp))
        matrices.append(vals)
        manifest.append(entry)
        qc.append(q)
    gene_ids = np.array(sorted(universe))
    keep = ~np.isin(gene_ids, list(union))
    require(keep.sum() >= 2, 'Fewer than two fixed non-TopN genes')
    gene_ids = gene_ids[keep]
    raw = np.stack(matrices)[:, keep, :]
    data = np.log1p(raw) if a.space == 'log1p' else raw
    x, y = data[:, :, 0], data[:, :, 1]
    names, ci, xm, ym, xr, yr = decompose(x, y, meta[a.condition_column].tolist())
    out.mkdir(parents=True, exist_ok=True)
    def save(df, name):
        df.to_csv(out / (name + '.csv'), index=False, na_rep='NA', float_format='%.12g')
    save(pd.DataFrame(manifest), 'input_manifest')
    save(pd.DataFrame(qc), 'truth_qc')
    save(pd.DataFrame({'gene': sorted(universe), 'excluded_topn_union': np.isin(sorted(universe), list(union))}), 'gene_universe')
    save(pd.DataFrame({'gene': gene_ids}), 'fixed_nonTopN_genes')
    boundaries = {}
    for label, xt, yt, ids in [('total', x, y, meta.sample_id.tolist()),
                               ('condition_mean', xm, ym, names),
                               ('within_condition_residual', xr, yr, meta.sample_id.tolist())]:
        gene = scores(xt, yt, a.epsilon)
        gene.insert(0, 'gene', gene_ids)
        gene['trueSD_quartile'], cuts = quartiles(gene.true_sd.to_numpy(), a.epsilon)
        boundaries[label + '_trueSD'] = cuts.tolist()
        sample = scores(xt.T, yt.T, a.epsilon)
        sample.insert(0, 'condition' if label == 'condition_mean' else 'sample_id', ids)
        if label != 'condition_mean':
            sample.insert(1, 'condition', meta[a.condition_column].to_numpy())
            sample.insert(2, 'rep', meta.rep.to_numpy())
        if label == 'within_condition_residual':
            for frame, truth, pred in [(gene, xr, yr), (sample, xr.T, yr.T)]:
                frame['zero_residual_mae'] = abs(truth).mean(axis=0)
                frame['zero_residual_rmse'] = np.sqrt((truth**2).mean(axis=0))
                frame['mae_improvement_vs_zero'] = frame.zero_residual_mae - frame.mae
                frame['win_fraction_vs_zero'] = (abs(pred-truth) < abs(truth)).mean(axis=0)
        save(gene, label + '_genewise')
        save(sample, label + '_samplewise')
        gs = summarize(gene, [('all_fixed_nonTopN', np.ones(len(gene), bool)), ('true_variable_only', gene.true_sd > a.epsilon)])
        save(gs, label + '_genewise_summary')
        save(summarize(gene, [(q, gene.trueSD_quartile == q) for q in ['Zero', 'Q1', 'Q2', 'Q3', 'Q4']]), label + '_by_trueSD_quartile')
        save(summarize(sample, [('all', np.ones(len(sample), bool))]), label + '_samplewise_summary')
        print(f'\n=== {label.upper()} GENE-WISE: TRUE-VARIABLE GENES ===')
        print(gs.loc[gs.group == 'true_variable_only'].to_string(index=False))
    # Pooled residual metrics are descriptive, with dependent gene/sample pairs.
    flatx, flaty = xr.ravel(), yr.ravel()
    magnitude, cuts = quartiles(abs(flatx), a.epsilon)
    boundaries['absolute_true_residual'] = cuts.tolist()
    pooled = []
    for label, mask in [('All', np.ones(len(flatx), bool)), ('Nonzero', abs(flatx) > a.epsilon)] + [(q, magnitude == q) for q in ['Zero', 'Q1', 'Q2', 'Q3', 'Q4']]:
        tx, py = flatx[mask], flaty[mask]
        record = dict(group=label, n_pairs=len(tx))
        if len(tx):
            improvement = abs(tx)-abs(py-tx)
            record.update(scores(tx, py, a.epsilon).iloc[0].to_dict())
            record.update(abs_true_residual_min=float(abs(tx).min()), abs_true_residual_max=float(abs(tx).max()),
                          abs_true_residual_median=float(np.median(abs(tx))),
                          zero_residual_mae=float(abs(tx).mean()), zero_residual_rmse=float(np.sqrt((tx**2).mean())),
                          mean_mae_improvement=float(improvement.mean()), median_mae_improvement=float(np.median(improvement)),
                          win_fraction=float((improvement > 0).mean()))
        pooled.append(record)
    pooled = pd.DataFrame(pooled)
    save(pooled, 'residual_magnitude_quartile_summary')
    print('\n=== RESIDUAL MAGNITUDE QUARTILES / ZERO-RESIDUAL REFERENCE ===')
    print(pooled.to_string(index=False))
    # Orthogonal decomposition: SSE_total = SSE_between + SSE_within.
    total_sse = ((y-x)**2).sum(axis=0)
    between_sse = ((ym[ci]-xm[ci])**2).sum(axis=0)
    within_sse = ((yr-xr)**2).sum(axis=0)
    require(np.allclose(total_sse, between_sse+within_sse, rtol=1e-10, atol=1e-10), 'SSE decomposition failed')
    save(pd.DataFrame(dict(gene=gene_ids, total_sse=total_sse, condition_mean_sse=between_sse,
                          within_condition_sse=within_sse, decomposition_error=total_sse-between_sse-within_sse,
                          true_between_ss=((xm[ci]-x.mean(axis=0))**2).sum(axis=0),
                          true_within_ss=(xr**2).sum(axis=0))), 'variance_error_decomposition')
    for label, arr in [('condition_mean_true', xm), ('condition_mean_pred', ym),
                       ('residual_true', xr), ('residual_pred', yr)]:
        frame = pd.DataFrame(arr.T, columns=names if label.startswith('condition_mean') else meta.sample_id)
        frame.insert(0, 'gene', gene_ids)
        save(frame, label + '_matrix')
    if a.pair_output:
        save(pd.DataFrame(dict(sample_id=np.repeat(meta.sample_id.to_numpy(), len(gene_ids)),
                              condition=np.repeat(meta[a.condition_column].to_numpy(), len(gene_ids)),
                              gene=np.tile(gene_ids, len(meta)), true_residual=flatx, pred_residual=flaty,
                              magnitude_quartile=magnitude)), 'residual_gene_sample_pairs')
    report = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  metadata_sha256=sha(meta_path), script_sha256=sha(Path(__file__)),
                  python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                  n_samples=len(meta), n_conditions=len(names), n_universe=len(universe),
                  n_topn_union_in_universe=len(universe & union), n_fixed_genes=len(gene_ids),
                  transformation='log1p then mean/center' if a.space == 'log1p' else 'raw then mean/center',
                  residual='true - own true condition mean; pred - own predicted condition mean',
                  full_duplicate_policy='last_row_wins, matching predictor dict(zip(Gene, Expression))',
                  sd_ddof=0, quartile_boundaries=boundaries,
                  quartile_method='Positive (>epsilon) values only; <=epsilon is Zero; ties stay together; empty bins retained',
                  zero_reference='Diagnostic zero within-condition residual, not deployable unseen-condition expression baseline',
                  inference='Descriptive metrics only; no independent-pair p-values; fold training provenance is not verified by CSV layout',
                  total_sse=float(total_sse.sum()), condition_mean_sse=float(between_sse.sum()),
                  residual_sse=float(within_sse.sum()))
    # Strict JSON: absent quantiles are null.
    report['quartile_boundaries'] = {k: [v if np.isfinite(v) else None for v in vals] for k, vals in boundaries.items()}
    (out / 'evaluation_config.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'\n[DONE] fixed non-Top{a.topn}: {len(gene_ids)} genes; outputs: {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--loco_base_dir', default='output/paper_benchmark/loco/top1000_nocond')
    p.add_argument('--metadata_csv', '--sample_metadata_csv', default='data/metadata/sample_metadata_zF_gen1.csv')
    p.add_argument('--topn_dir', default='data/processed/zF_bulk/topN')
    p.add_argument('--full_dir', help='Optional full-expression directory for independent raw_true QC')
    p.add_argument('--topn', type=int, default=1000)
    p.add_argument('--condition_column', default='cell_type')
    p.add_argument('--out_dir')
    p.add_argument('--space', choices=['log1p', 'raw'], default='log1p')
    p.add_argument('--epsilon', type=float, default=1e-12, help='SD/magnitude threshold for constant/Zero')
    p.add_argument('--truth_rtol', type=float, default=1e-5)
    p.add_argument('--truth_atol', type=float, default=1e-6)
    p.add_argument('--pair_output', action='store_true')
    a = p.parse_args()
    try:
        require(a.topn > 0, 'topn must be positive')
        require(all(np.isfinite(v) and v >= 0 for v in [a.epsilon, a.truth_rtol, a.truth_atol]), 'Tolerances must be finite and nonnegative')
        evaluate(a)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f'[ERROR] {exc}\n')


if __name__ == '__main__':
    main()
