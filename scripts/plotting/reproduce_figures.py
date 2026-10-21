#!/usr/bin/env python3
"""Regenerate all eight figures from release source tables into a separate folder."""
from pathlib import Path
import argparse, os, subprocess, sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    target = args.out_dir.resolve()
    target.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ, TOP2ALL_PLOT_OUT=str(target), MPLBACKEND='Agg')
    for figure in ['Figure1','Figure2','Figure3','Figure4','Figure5','FigureS1','FigureS2','FigureS3']:
        with (target/(figure+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(scripts/f'plot_{figure.lower()}.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for ext in ('png','pdf','svg'):
            assert (target/figure/(figure+'.'+ext)).is_file()
        print(figure+': PASS',flush=True)
if __name__ == '__main__': main()
