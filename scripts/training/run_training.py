#!/usr/bin/env python3
"""Plan or run one published reconstruction/scale/prediction fold."""
from pathlib import Path
import argparse, json, subprocess, sys
ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['condition_oof','nocond_oof','nocond_loco'],required=True)
    p.add_argument('--heldout-rep',type=int,choices=[1,2,3])
    p.add_argument('--heldout-condition')
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--topn',type=int,choices=[100,200,300,500,1000,2000],default=1000)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--dictionary-dir',type=Path,required=True)
    p.add_argument('--full-dir',type=Path,required=True)
    p.add_argument('--topn-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--execute',action='store_true',help='Actually train and predict; default prints the command plan only.')
    a=p.parse_args()
    if a.family != 'nocond_oof' and a.topn != 1000: p.error('Only no-condition OOF uses input-depth sensitivity.')
    loco=a.family=='nocond_loco'
    if loco and not a.heldout_condition: p.error('--heldout-condition is required for LOCO.')
    if not loco and a.heldout_rep is None: p.error('--heldout-rep is required for OOF.')
    train={'condition_oof':'oof','nocond_oof':'oof_nocond','nocond_loco':'loco_nocond'}[a.family]
    pred={'condition_oof':'predict.py','nocond_oof':'predict_nocond.py','nocond_loco':'predict_loco_nocond.py'}[a.family]
    output=a.output.resolve(); logdir=output/'training/log1p'; scaledir=output/'training/scale'
    if loco: logck=logdir/f'zf_gen1_v5_nocond_top{a.topn}_loco_test_{a.heldout_condition}_last.pt'
    elif a.family=='nocond_oof': logck=logdir/f'zf_gen1_v5_nocond_top{a.topn}_oof_test_rep{a.heldout_rep}_last.pt'
    else: logck=logdir/f'zf_gen1_v5_top{a.topn}_oof_test_rep{a.heldout_rep}_last.pt'
    shared=['--geneformer_model_dir',str(a.model_dir.resolve()),'--pkl_dir',str(a.dictionary_dir.resolve()),'--sample_metadata_csv',str(ROOT/'metadata/sample_metadata.csv'),'--common_genes_csv',str(ROOT/'metadata/common_genes.txt'),'--full_dir',str(a.full_dir.resolve()),'--topn_dir',str(a.topn_dir.resolve()),'--topn',str(a.topn),'--device',a.device]
    fold=['--heldout_condition',a.heldout_condition] if loco else ['--heldout_rep',str(a.heldout_rep)]
    filt=['--condition_filter',a.heldout_condition] if loco else ['--rep_filter',str(a.heldout_rep)]
    model_args=['--epochs','30','--batch_size','4','--max_seq_len','2048' if a.family=='nocond_oof' else '1000','--lr','0.0001','--weight_decay','0.01','--aggregation_mode','mean','--activation_type','softplus','--alpha_weight','0.3','--pos_weight_zero','3.0','--lambda_zero','1.0']
    cmds=[
        [sys.executable,str(ROOT/f'scripts/training/train_log1p_{train}.py'),*shared,*fold,'--out_ckpt_dir',str(logdir),'--seed',str(a.seed),*model_args],
        [sys.executable,str(ROOT/f'scripts/training/train_scale_head_{train}.py'),*shared,*fold,'--out_scale_dir',str(scaledir),'--log1p_ckpt',str(logck),'--seed',str(a.seed+1192),'--epochs','50','--batch_size','4','--lr','0.0001','--weight_decay','0.0','--eps','0.000001'],
        [sys.executable,str(ROOT/f'scripts/prediction/{pred}'),*shared,*filt,'--out_dir',str(output/'prediction'),'--log1p_ckpt',str(logck),'--scale_ckpt',str(scaledir/'scale_head_last.pt'),'--batch_size','4']]
    print(json.dumps(cmds,indent=2))
    if not a.execute:return
    for d in [a.model_dir,a.dictionary_dir,a.full_dir,a.topn_dir]:
        if not d.is_dir():p.error(f'Missing asset directory: {d}')
    output.mkdir(parents=True,exist_ok=False)
    (output/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    for i,cmd in enumerate(cmds):
        with (output/f'stage{i+1}.log').open('w') as log: subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
if __name__=='__main__':main()
