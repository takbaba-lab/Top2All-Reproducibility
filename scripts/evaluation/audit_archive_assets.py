#!/usr/bin/env python3
"""Audit archive inventory; verify ignored staging bytes only when requested."""
from pathlib import Path
import argparse,csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
def audit_archive(require_local=False):
    path=ROOT/'reproducibility/ZENODO_MANIFEST.tsv'
    with path.open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    required={'asset_id','asset_type','local_source_path','intended_archive_path','size_bytes','sha256','local_status','redistribution_status','description'}
    assert required<=set(rows[0])
    assert len({r['asset_id'] for r in rows})==len(rows)
    assert len({r['intended_archive_path'] for r in rows})==len(rows)
    ready=[];missing=[]
    for r in rows:
        assert not Path(r['intended_archive_path']).is_absolute() and '..' not in Path(r['intended_archive_path']).parts
        assert re.fullmatch('[0-9a-f]{64}',r['sha256'])
        assert r['redistribution_status']=='proposed_project_asset_author_review_pending'
        assert r['asset_type'] not in ('model','dictionary','third_party_model','third_party_dictionary','font','software_package','raw_rna_seq')
        if r['local_status']=='verified_local':
            ready.append(r);p=Path(r['local_source_path'])
            assert r['local_source_path'].startswith('.local/zenodo_assets/') and '..' not in p.parts
            assert int(r['size_bytes'])>0
            if require_local:
                p=ROOT/p;assert p.is_file(),r['asset_id']
                assert p.stat().st_size==int(r['size_bytes'])
                assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['asset_id']
        else:
            assert r['local_status']=='missing_not_staged' and not r['local_source_path'] and r['size_bytes']=='unknown'
            missing.append(r)
    assert len(ready)==103 and len(missing)==917
    if require_local:
        sums=ROOT/'.local/zenodo_assets/archive_metadata/checksums.sha256'
        covered={line.split('  ',1)[1]:line.split('  ',1)[0] for line in sums.read_text().splitlines()}
        expected={r['intended_archive_path']:r['sha256'] for r in ready if r['asset_type']!='checksum'}
        assert covered==expected
        actual={str(p.relative_to(ROOT/'.local/zenodo_assets')) for p in (ROOT/'.local/zenodo_assets').rglob('*') if p.is_file()}
        assert actual=={r['intended_archive_path'] for r in ready},actual-{r['intended_archive_path'] for r in ready}
    return dict(records=len(rows),staged_files=len(ready),staged_bytes=sum(int(r['size_bytes']) for r in ready),missing_project_assets=len(missing),local_bytes_verified=require_local)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-local-staging',action='store_true');a=p.parse_args();print(json.dumps(audit_archive(a.require_local_staging),indent=2))
