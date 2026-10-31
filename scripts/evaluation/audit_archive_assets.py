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
    ready=[];excluded=[]
    for r in rows:
        assert not Path(r['intended_archive_path']).is_absolute() and '..' not in Path(r['intended_archive_path']).parts
        assert re.fullmatch('[0-9a-f]{64}',r['sha256'])
        assert r['release_version']=='v1.0.0'
        assert r['redistribution_status'] in ('project_origin_core_release', 'project_origin_separate_distribution_optional')
        assert r['asset_type'] not in ('model','dictionary','third_party_model','third_party_dictionary','font','software_package','raw_rna_seq')
        if r['local_status']=='verified_local':
            assert r['included_in_core_archive']=='true' and r['redistribution_status']=='project_origin_core_release'
            ready.append(r);p=Path(r['local_source_path'])
            assert r['local_source_path'].startswith('.local/zenodo_assets/') and '..' not in p.parts
            assert int(r['size_bytes'])>0
            if require_local:
                p=ROOT/p;assert p.is_file(),r['asset_id']
                assert p.stat().st_size==int(r['size_bytes'])
                assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['asset_id']
        else:
            assert r['local_status']=='excluded_from_core_archive' and not r['local_source_path']
            assert r['asset_type']=='checkpoint' and r['included_in_core_archive']=='false'
            assert r['genkai_status']=='located_and_hash_verified' and int(r['size_bytes'])>0
            excluded.append(r)
    assert len(ready)==846 and len(excluded)==174 and len(rows)==1020
    core=json.loads((ROOT/'reproducibility/CORE_ARCHIVE.json').read_text())
    assert core['version']=='1.0.0' and core['tag']=='v1.0.0'
    assert core['zenodo_doi']=='10.5281/zenodo.23361275'
    assert core['files']==len(ready) and core['uncompressed_bytes']==sum(int(r['size_bytes']) for r in ready)
    assert core['checkpoint_exclusion_bytes']==sum(int(r['size_bytes']) for r in excluded)
    if require_local:
        sums=ROOT/'.local/zenodo_assets/archive_metadata/checksums.sha256'
        covered={line.split('  ',1)[1]:line.split('  ',1)[0] for line in sums.read_text().splitlines()}
        expected={r['intended_archive_path']:r['sha256'] for r in ready if r['asset_type']!='checksum'}
        assert covered==expected
        actual={str(p.relative_to(ROOT/'.local/zenodo_assets')) for p in (ROOT/'.local/zenodo_assets').rglob('*') if p.is_file()}
        assert actual=={r['intended_archive_path'] for r in ready},actual-{r['intended_archive_path'] for r in ready}
    return dict(records=len(rows),staged_files=len(ready),staged_bytes=sum(int(r['size_bytes']) for r in ready),excluded_project_checkpoints=len(excluded),unlocated_project_assets=0,local_bytes_verified=require_local)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-local-staging',action='store_true');a=p.parse_args();print(json.dumps(audit_archive(a.require_local_staging),indent=2))
