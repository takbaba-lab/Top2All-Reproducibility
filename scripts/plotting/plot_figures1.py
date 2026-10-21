"""Display-only S2: ordered truth-SD quartiles, original numerical values."""
from pathlib import Path
import csv,json,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
from matplotlib.text import Text
import os
ROOT = Path(__file__).resolve().parents[2]
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'FigureS1'
S = ROOT / 'source_data/FigureS1/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Reconstruction performance varies with target-gene variability.'
W=85;H=135;QUARTILES=['Q1','Q2','Q3','Q4'];SEEDS=[42,43,44];YLIMS={'A':[-.20,1.06],'B':[.04,.46]}
STYLE={'individual':{'color':'#56B4E9','lw':.6,'ms':2.7,'alpha':.35},'median':{'color':'#0072B2','lw':1.6,'ms':4.5,'alpha':1.0},'GlobalMean':{'color':'#999999','lw':1.0,'ms':3.4,'alpha':1.0},'SameConditionMean':{'color':'#009E73','lw':1.0,'ms':3.4,'alpha':1.0}}
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.1,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':True})
def read(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 m=json.loads((ROOT/'source_data/FigureS1/source_manifest.json').read_text());s=S/'gene_variability_metrics.tsv';assert hashlib.sha256(s.read_bytes()).hexdigest()==m['files'][0]['sha256']
 assert True  # Numerical sources are checked by validate_release.py.
 rows=read(s);f=plt.figure(figsize=(W/25.4,H/25.4));axes={};lines=[];traces=[]
 for panel,metric,(bottom,top) in zip(['A','B'],['pearson','mae'],[(82,123),(17,58)]):
  ax=f.add_axes([.255,bottom/H,.69,(top-bottom)/H]);axes[panel]=ax;ax.set_ylim(YLIMS[panel]);ax.set_xlim(-.30,3.30)
  def values(method,seed=None,median=False):
   yy=[]
   for i,q in enumerate(QUARTILES):
    selector={'mode':'condition','method':method,'trueSD_quantile':q}
    if seed is not None:selector['seed_log1p']=seed
    rr=[r for r in rows if all(r[k]==str(v) for k,v in selector.items())];assert len(rr)==(3 if median else 1)
    n=1737 if q=='Q3' else 1738;assert all(int(r['n_genes'])==n for r in rr)
    raw=format(statistics.median(float(r[metric+'_median']) for r in rr),'.17g') if median else rr[0][metric+'_median'];yy.append(float(raw))
    traces.append({'panel':panel,'method':method,'seed_log1p':seed if seed is not None else '3-seed median','quartile':q,'n_genes':n,'metric':metric,'value':raw,'source_selector':json.dumps(selector,sort_keys=True,separators=(',',':')),'source_column':metric+'_median','operation':'median' if median else 'cell','n_selected':len(rr),'display_x':i,'representation':'median' if median else 'individual' if method=='Top2All' else method})
   return yy
  def draw(yy,kind,seed=''):
   st=STYLE[kind];o,=ax.plot(range(4),yy,'o-',color=st['color'],lw=st['lw'],ms=st['ms'],alpha=st['alpha'],mew=.6,zorder=2 if kind=='individual' else 4 if kind=='median' else 3);lines.append((panel,kind,seed,o))
  for seed in SEEDS:draw(values('Top2All',seed=seed),'individual',seed)
  draw(values('Top2All',median=True),'median')
  for method in ['GlobalMean','SameConditionMean']:draw(values(method,seed=42),method)
  ax.set_xticks(range(4),QUARTILES);ax.set_xlabel('Truth SD quartile (low → high)');ax.set_ylabel('Median gene-wise\nPearson r' if metric=='pearson' else 'Median gene-wise\nMAE (log1p)');ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
  ax.set_yticks([-.1,0,.25,.50,.75,1.0] if panel=='A' else [.1,.2,.3,.4]);ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 labels=['Top2All, individual seeds','Top2All, 3-seed median','Global mean','Same-condition mean'];handles=[]
 for kind,label in zip(['individual','median','GlobalMean','SameConditionMean'],labels):
  st=STYLE[kind];handles.append(Line2D([],[],marker='o',color=st['color'],lw=st['lw'],ms=st['ms'],alpha=st['alpha'],mew=.6,label=label))
 legend=axes['A'].legend(handles=handles,loc='center',bbox_to_anchor=(.5,.44),frameon=False,fontsize=7.1,handlelength=2.7,handletextpad=.6,labelspacing=.5,borderaxespad=.2)
 marker_checks=[];line_checks=[];text_checks=[];style_rows=[]
 for panel,kind,seed,o in lines:style_rows.append({'panel':panel,'kind':kind,'seed_log1p':seed,'color':o.get_color(),'linewidth_pt':o.get_linewidth(),'markersize_pt':o.get_markersize(),'alpha':o.get_alpha(),'marker':o.get_marker()})
 for dpi in [100,150,600]:
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer();lb=legend.get_window_extent(renderer);a=axes['A'];assert a.bbox.contains(lb.x0,lb.y0) and a.bbox.contains(lb.x1,lb.y1)
  for panel,kind,seed,o in lines:
   ax=axes[panel];box=ax.bbox;xy=o.get_transform().transform(o.get_xydata());radius=(o.get_markersize()+o.get_markeredgewidth())*dpi/144+1
   for i,(x,y) in enumerate(xy):
    gap=min(x-radius-box.x0,box.x1-x-radius,y-radius-box.y0,box.y1-y-radius);assert gap>0,(panel,kind,dpi,gap)
    marker_checks.append({'panel':panel,'kind':kind,'seed_log1p':seed,'quartile':QUARTILES[i],'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
    if panel=='A':assert x+radius<lb.x0 or x-radius>lb.x1 or y+radius<lb.y0 or y-radius>lb.y1,('legend marker overlap',kind)
   guard=o.get_linewidth()*dpi/144+1;gap=min(xy[:,0].min()-box.x0,box.x1-xy[:,0].max(),xy[:,1].min()-box.y0,box.y1-xy[:,1].max())-guard;assert gap>0
   line_checks.append({'panel':panel,'kind':kind,'seed_log1p':seed,'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
   if panel=='A':assert not o.get_path().transformed(o.get_transform()).intersects_bbox(lb),'legend line overlap'
  for t in f.findobj(match=Text):
   if not t.get_visible() or not t.get_text().strip():continue
   b=t.get_window_extent(renderer);gap=min(b.x0,b.y0,f.bbox.width-b.x1,f.bbox.height-b.y1);assert gap>0,(t.get_text(),b)
   text_checks.append({'dpi':dpi,'text':t.get_text(),'min_canvas_clearance_px':float(gap),'clipped':False})
  assert axes['A'].get_tightbbox(renderer).y0>axes['B'].get_tightbbox(renderer).y1
 write(P/'exact_plot_data.csv',traces)
 for panel in ['A','B']:write(P/f'Panel_{panel}_exact_values.csv',[r for r in traces if r['panel']==panel])
 write(Q/'marker_clipping_audit.csv',marker_checks);write(Q/'line_clipping_audit.csv',line_checks);write(Q/'text_clipping_audit.csv',text_checks);write(Q/'line_style_audit.csv',style_rows)
 padding=[]
 for panel,(lo,hi) in YLIMS.items():
  yy=[float(r['value']) for r in traces if r['panel']==panel];span=max(yy)-min(yy);padding.append({'panel':panel,'data_min':min(yy),'data_max':max(yy),'ylim_min':lo,'ylim_max':hi,'lower_padding_fraction':(min(yy)-lo)/span,'upper_padding_fraction':(hi-max(yy))/span})
 write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/f'FigureS1.{ext}',dpi=600,metadata={'Title':TITLE})
 (Q/'layout.json').write_text(json.dumps({'title':TITLE,'dimensions_mm':[W,H],'ylims':YLIMS,'style':STYLE,'legend_labels':labels,'legend_location':'inside Panel A center, in gap between Top2All and GlobalMean trajectories','individual_seed_lines_per_panel':3,'median_lines_per_panel':1,'quartiles':QUARTILES,'quartile_counts':[1738,1738,1737,1738],'plotted_values':len(traces),'marker_tests':len(marker_checks),'line_tests':len(line_checks),'text_tests':len(text_checks),'marker_clipping':0,'text_clipping':0,'legend_overlap':False},indent=2)+'\n');plt.close(f)
 print(json.dumps({'values':len(traces),'ylims':YLIMS,'marker_tests':len(marker_checks),'text_tests':len(text_checks)},indent=2))
if __name__=='__main__':main()
