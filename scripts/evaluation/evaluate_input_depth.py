#!/usr/bin/env python3
"""Reevaluate fixed common/bridge targets from the published input-depth predictions.

The numerical evaluation is retained from the final input-depth evaluator. A
portable prediction TSV replaces production scheduler/receipt path discovery.
"""
from pathlib import Path
import argparse,csv,hashlib
import numpy as np
import pandas as pd
import compare_oof_top2all_vs_mean_baselines as reference
ROOT=Path(__file__).resolve().parents[2]
SEEDS=((42,1234),(43,1235),(44,1236))
def require(ok,message):
    if not ok:raise RuntimeError(message)
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def evaluate(truth,prediction,genes,identity):
    require(truth.shape==prediction.shape==(24,len(genes)),'Metrics must use 24 OOF samples')
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(),'Nonfinite metrics input')
    detail=[dict(**identity,gene=g,**reference.calc_metrics(truth[:,j],prediction[:,j])) for j,g in enumerate(genes)]
    summary=[]
    for metric in reference.METRICS:
        v=np.asarray([r[metric] for r in detail]);v=v[np.isfinite(v)]
        summary.append(dict(**identity,metric=metric,statistic='median_over_genes',value=float(np.median(v)) if len(v) else float('nan'),n_selected=len(genes),n_valid=len(v),n_undefined=len(genes)-len(v)))
    return detail,summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction-manifest',type=Path,required=True,help='TSV: topn, seed_log1p, seed_scale, heldout_rep, sample_id, path, sha256. Paths relative to this TSV.')
    p.add_argument('--full-dir',type=Path,required=True)
    p.add_argument('--topn-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    genes=(ROOT/'metadata/common_genes.txt').read_text().splitlines()
    m=pd.read_csv(ROOT/'metadata/sample_metadata.csv').sort_values(['cell_type','rep','sample_id']).reset_index(drop=True)
    sids=m.sample_id.tolist();ix={s:i for i,s in enumerate(sids)};gi={g:i for i,g in enumerate(genes)}
    raw=np.vstack([reference.vector_aligned_to_gene_order(a.full_dir/f'{s}_full.csv',genes) for s in sids]);truth=np.log1p(raw)
    require(raw.shape==(24,24410) and np.isfinite(raw).all() and (raw>=0).all(),'Invalid truth matrix')
    primary=(ROOT/'metadata/input_depth_common_target_genes.txt').read_text().splitlines();bridge=(ROOT/'metadata/main_true_variable_genes.txt').read_text().splitlines()
    require(len(primary)==5605 and len(bridge)==6951,'Target counts')
    manifest=pd.read_csv(a.prediction_manifest,sep='\t',dtype=str).to_dict('records');require(len(manifest)==432,'Expected 54 fold jobs, 432 prediction files')
    summary=[];qcs=[]
    for n in [100,200,300,500,1000,2000]:
        union=set()
        for sid in sids:union.update(pd.read_csv(a.topn_dir/f'{sid}_top{n}.csv').Gene)
        fixed=[g for g in genes if g not in union];js=[gi[g] for g in fixed];native=np.asarray(fixed)[np.std(truth[:,js],axis=0,ddof=1)>0].tolist();sets={'primary5605':primary,'native_hidden':native}
        if n<=1000:sets['bridge6951']=bridge
        for selected in sets.values():require(not set(selected)&union,'Input/target overlap')
        for sa,sb in SEEDS:
            vv={}
            for rep in (1,2,3):
                rows=[r for r in manifest if (int(r['topn']),int(r['seed_log1p']),int(r['seed_scale']),int(r['heldout_rep']))==(n,sa,sb,rep)]
                expected=set(m[m.rep==rep].sample_id);require(len(rows)==8 and {r['sample_id'] for r in rows}==expected,'Fold sample assignment')
                for r in rows:
                    f=Path(r['path']);f=f if f.is_absolute() else a.prediction_manifest.parent/f
                    require(hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],'Prediction hash mismatch')
                    sid=r['sample_id'];d=pd.read_csv(f);require(d.gene.tolist()==genes and 'raw_pred_wo_overwrite' in d,'Full universe/column')
                    require(np.isfinite(d.raw_pred_wo_overwrite).all() and (d.raw_pred_wo_overwrite>=0).all(),'Finite prediction')
                    require(np.isclose(d.raw_true,raw[ix[sid]],rtol=1e-5,atol=1e-6).all(),'Prediction truth')
                    require(sids[int(np.argmin(np.mean((raw-d.raw_true.to_numpy())**2,axis=1)))]==sid,'Best-match-self')
                    require(sid not in vv,'Duplicate OOF sample');vv[sid]=np.log1p(d.raw_pred_wo_overwrite.to_numpy(float))
                    for selected in sets.values():
                        kk=[gi[g] for g in selected];require(np.array_equal(d.raw_pred_wo_overwrite.to_numpy()[kk],d.raw_pred_final.to_numpy()[kk]),'Hidden overwrite')
                qcs.append(dict(topn=n,seed_log1p=sa,heldout_rep=rep,prediction_files=8,status='PASS'))
            require(set(vv)==set(sids),'OOF24 incomplete');pred=np.vstack([vv[s] for s in sids])
            for label,selected in sets.items():
                js=[gi[g] for g in selected];detail,ss=evaluate(truth[:,js],pred[:,js],selected,dict(pipeline='Top2All',topn=n,seed_log1p=sa,seed_scale=sb,method='Top2All',target_set=label));write(a.out_dir/f'top{n}_seed{sa}_{label}_genewise.tsv',detail);summary.extend(ss)
    require(len(qcs)==54,'Completion count');write(a.out_dir/'input_depth_summary.tsv',summary);write(a.out_dir/'fold_qc.tsv',qcs)
if __name__=='__main__':main()
