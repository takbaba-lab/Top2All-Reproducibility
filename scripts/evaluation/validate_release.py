#!/usr/bin/env python3
"""Read-only integrity and source-reduction checks for the paper release."""
from pathlib import Path
from collections import defaultdict, Counter
from functools import cache
import argparse, ast, csv, hashlib, json, math, re, statistics
ROOT = Path(__file__).resolve().parents[2]
@cache
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
@cache
def read(path):
    path = Path(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))
indices = {}
def select(file, selector):
    rows = read(ROOT / file)
    if any(isinstance(v, list) for v in selector.values()):
        return [r for r in rows if all(r[k] in [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in selector.items())]
    keys = tuple(sorted(selector)); token = (file, keys)
    if token not in indices:
        idx = defaultdict(list)
        for row in rows: idx[tuple(row[k] for k in keys)].append(row)
        indices[token] = idx
    return indices[token].get(tuple(str(selector[k]) for k in keys), [])
def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None
def reduce(rows, column, operation):
    assert rows, 'Empty selection'
    if operation == 'count_unique': return float(len({r[column] for r in rows})), len(rows)
    if operation == 'gene_universe_complement': return float(24410-len({r[column] for r in rows})), len(rows)
    values = [v for r in rows if (v := number(r[column])) is not None]
    if operation == 'cell': assert len(rows) == 1
    if not values: return None, 0
    fun = {'cell': lambda: values[0], 'mean': lambda: statistics.fmean(values), 'median': lambda: statistics.median(values), 'min': lambda: min(values), 'max': lambda: max(values), 'sum': lambda: sum(values)}
    return fun[operation](), len(values)
def equal(a, b):
    return a == b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path)
    ap.add_argument('--skip-checksums', action='store_true')
    args = ap.parse_args()
    registry = read(ROOT/'reproducibility/result_registry.tsv')
    by_id = {r['result_id']: r for r in registry}
    assert len(registry) == len(by_id) == 45543
    for row in registry:
        path = ROOT/row['source_file']
        assert path.is_file() and sha(path) == row['source_sha256'], row['result_id']
        selected = select(row['source_file'], json.loads(row['source_selector']))
        value, nvalid = reduce(selected, row['source_column'], row['operation'])
        assert len(selected) == int(row['n_selected']) and nvalid == int(row['n_valid']), row['result_id']
        assert row['value'] == ('NA' if value is None else format(value, '.17g')), (row['result_id'], value, row['value'])
        assert row['value_status'] == ('undefined' if value is None else 'finite')
        if row['figure_file']:
            assert sha(ROOT/row['figure_file']) == row['figure_sha256']
        if row['plotting_script']:
            assert sha(ROOT/row['plotting_script']) == row['plotting_script_sha256']
    expected = {f'Figure{i}' for i in range(1,6)} | {'FigureS1','FigureS2','FigureS3'}
    figures = list((ROOT/'figures').iterdir())
    assert {p.name for p in figures} == {f'{fig}.{ext}' for fig in expected for ext in ('png','pdf','svg')}
    manifest = read(ROOT/'reproducibility/figure_manifest.tsv')
    assert len(manifest) == 24 and all(sha(ROOT/r['path']) == r['sha256'] == r['source_export_sha256'] for r in manifest)
    links = 0
    for fig in sorted(expected):
        for row in read(ROOT/'source_data'/fig/'registry_links.tsv'):
            assert row['figure'] == fig
            rr = by_id[row['result_id']]
            assert equal(number(row['value']), number(rr['value']))
            assert row['source_file'] == rr['source_file']
            data = read(ROOT/row['plot_data_file'])
            assert equal(number(data[int(row['plot_data_row'])][row['plot_data_column']]), number(row['value']))
            links += 1
    assert links == 42157
    meta = read(ROOT/'metadata/sample_metadata.csv')
    assert len(meta) == 24 and sorted(Counter(r['cell_type'] for r in meta).values()) == [3]*8
    for file, key, nfold, ntrain, ntest in [('oof_folds.csv','fold',3,16,8),('loco_folds.csv','heldout_condition',8,21,3)]:
        rows = read(ROOT/'metadata'/file); folds = sorted({r[key] for r in rows}); assert len(folds) == nfold
        test_counts = Counter()
        for fold in folds:
            part = [r for r in rows if r[key] == fold]
            train = {r['sample_id'] for r in part if r['partition']=='train'}
            test = {r['sample_id'] for r in part if r['partition']=='test'}
            assert len(train)==ntrain and len(test)==ntest and not train & test
            test_counts.update(test)
        assert set(test_counts.values()) == {1} and len(test_counts) == 24
    assert len((ROOT/'metadata/common_genes.txt').read_text().splitlines()) == 24410
    for path in (ROOT/'scripts').rglob('*.py'): ast.parse(path.read_text())
    findings = []
    # Prefix detection covers local and cluster filesystem roots without recording identities.
    forbidden = re.compile(r'/(?:Users|home|scratch|lustre|gpfs|System/Library)/|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',re.I)
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ('.git','.local','__pycache__')): continue
        if path.suffix.lower() in ('.png','.pdf'): continue
        text = path.read_text()
        if forbidden.search(text): findings.append(str(path.relative_to(ROOT)))
        if path.name != 'validate_release.py':
            assert not re.search(r'\bcanonical\b',text,re.I), path.relative_to(ROOT)
    assert not findings, findings
    verified = 0
    if not args.skip_checksums:
        for line in (ROOT/'reproducibility/checksums.sha256').read_text().splitlines():
            digest, file = line.split('  ',1)
            assert sha(ROOT/file)==digest, file
            verified += 1
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ('.git','.local','__pycache__')) and str(p.relative_to(ROOT))!='reproducibility/checksums.sha256'}
        listed = {line.split('  ',1)[1] for line in (ROOT/'reproducibility/checksums.sha256').read_text().splitlines()}
        assert actual == listed, (actual-listed, listed-actual)
    receipt = dict(status='PASS',registry_rows=len(registry),source_reducers_replayed=len(registry),figure_links_verified=links,final_exports=24,folds_verified=True,path_secret_findings=0,checksums_verified=verified,scope='Numerical source and release integrity; training and inference require external assets.')
    print(json.dumps(receipt,indent=2))
    if args.receipt: args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
if __name__ == '__main__': main()
