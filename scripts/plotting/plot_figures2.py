"""S1-only display revision; immutable summaries, no evaluation or model execution."""
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
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'FigureS2'
S = ROOT / 'source_data/FigureS2/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Main conclusions are robust across the three prespecified seed pairs.'
SEEDS=[(42,1234),(43,1235),(44,1236)];COLORS=['#0072B2','#D55E00','#009E73'];SETTINGS=['OOF_condition','OOF_nocond','LOCO_no_condition','strict_foldlocal_no_condition'];LABELS=['OOF\n+ cond.','OOF\nno cond.','LOCO','Strict\nOOF'];W=85;H=130;LIMITS={'A':[.575,.815],'B':[.225,.275]}
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.3,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':True})
def read(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 m=json.loads((ROOT/'source_data/FigureS2/source_manifest.json').read_text());source=S/'seed_robustness_metrics.tsv';assert hashlib.sha256(source.read_bytes()).hexdigest()==m['files'][0]['sha256']
 # Numerical sources are checked by validate_release.py.
 rows=read(source);f=plt.figure(figsize=(W/25.4,H/25.4));axes={};markers=[];bars=[];traces=[];stats=[]
 for panel,metric,(bottom,top) in zip(['A','B'],['pearson','mae'],[(80,119),(17,56)]):
  ax=f.add_axes([.255,bottom/H,.69,(top-bottom)/H]);axes[panel]=ax;ax.set_ylim(LIMITS[panel]);ax.set_xlim(-.45,3.45)
  for i,setting in enumerate(SETTINGS):
   vals=[]
   for j,((sa,sb),color) in enumerate(zip(SEEDS,COLORS)):
    rr=[r for r in rows if r['analysis']==setting and r['seed_log1p']==str(sa) and r['seed_scale']==str(sb) and r['metric']==metric];assert len(rr)==1;r=rr[0];y=float(r['value']);x=i+(j-1)*.10;vals.append(y)
    obj=ax.scatter(x,y,s=24,facecolor=color,edgecolor=color,linewidth=.8,zorder=3);markers.append((panel,obj))
    traces.append({'panel':panel,'analysis':setting,'seed_log1p':sa,'seed_scale':sb,'metric':metric,'value':r['value'],'display_x':repr(x),'color':color,'source_summary':r['source']})
   med=statistics.median(vals);line,=ax.plot([i-.21,i+.21],[med,med],color='black',lw=1,zorder=2);bars.append((panel,line))
   stats.append({'panel':panel,'analysis':setting,'metric':metric,'n_seeds':3,'three_seed_median':format(med,'.17g'),'x_min':i-.21,'x_max':i+.21})
  ax.set_xticks(range(4),LABELS);ax.set_ylabel('Median gene-wise\nPearson' if metric=='pearson' else 'Median gene-wise\nMAE (log1p)');ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
  ax.set_yticks([.60,.65,.70,.75,.80] if panel=='A' else [.23,.24,.25,.26,.27]);ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 legend=axes['A'].legend(handles=[Line2D([],[],marker='o',markersize=3.5,ls='',color=c,label=f'{sa} / {sb}') for (sa,sb),c in zip(SEEDS,COLORS)],title='log1p / scale seed',loc='lower left',bbox_to_anchor=(.015,.035),frameon=False,fontsize=7.3,title_fontsize=7.3,handlelength=.8,handletextpad=.5,labelspacing=.35,borderaxespad=.2)
 checks=[];linechecks=[];textchecks=[]
 for dpi in [100,150,600]:
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer()
  for panel,o in markers:
   ax=axes[panel];box=ax.bbox;x,y=ax.transData.transform(o.get_offsets())[0];path=o.get_paths()[0].get_extents();size=float(o.get_sizes()[0]);lw=float(o.get_linewidths()[0]);rx=max(abs(path.x0),abs(path.x1))*np.sqrt(size)*dpi/72+lw*dpi/144+1;ry=max(abs(path.y0),abs(path.y1))*np.sqrt(size)*dpi/72+lw*dpi/144+1;gap=min(x-rx-box.x0,box.x1-x-rx,y-ry-box.y0,box.y1-y-ry);assert gap>0
   checks.append({'panel':panel,'dpi':dpi,'x':float(o.get_offsets()[0,0]),'value':float(o.get_offsets()[0,1]),'min_edge_clearance_px':float(gap),'clipped':False})
  for panel,line in bars:
   xy=line.get_transform().transform(line.get_xydata());box=axes[panel].bbox;guard=line.get_linewidth()*dpi/144+1;gap=min(xy[:,0].min()-box.x0,box.x1-xy[:,0].max(),xy[:,1].min()-box.y0,box.y1-xy[:,1].max())-guard;assert gap>0
   linechecks.append({'panel':panel,'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
  for t in f.findobj(match=Text):
   if not t.get_visible() or not t.get_text().strip():continue
   b=t.get_window_extent(renderer);gap=min(b.x0,b.y0,f.bbox.width-b.x1,f.bbox.height-b.y1);assert gap>0,(dpi,t.get_text(),b)
   textchecks.append({'dpi':dpi,'text':t.get_text(),'min_canvas_clearance_px':float(gap),'clipped':False})
  lb=legend.get_window_extent(renderer);a=axes['A'];assert a.bbox.contains(lb.x0,lb.y0) and a.bbox.contains(lb.x1,lb.y1)
  for panel,o in markers:
   if panel=='A':
    x,y=axes[panel].transData.transform(o.get_offsets())[0];radius=(np.sqrt(float(o.get_sizes()[0]))+.8)*dpi/144+1;assert not lb.expanded(1.02,1.02).contains(x,y) and (x+radius<lb.x0 or x-radius>lb.x1 or y+radius<lb.y0 or y-radius>lb.y1),('legend overlaps marker',dpi,x,y,lb)
  for panel,line in bars:
   if panel=='A':assert not line.get_path().transformed(line.get_transform()).intersects_bbox(lb)
  assert axes['A'].get_tightbbox(renderer).y0>axes['B'].get_tightbbox(renderer).y1
 assert all(np.ptp(line.get_ydata())==0 for _,line in bars);assert sum(len(ax.lines) for ax in axes.values())==8
 write(P/'exact_plot_data.csv',traces);write(P/'three_seed_median_bars.csv',stats);write(Q/'marker_clipping_audit.csv',checks);write(Q/'median_bar_clipping_audit.csv',linechecks);write(Q/'text_clipping_audit.csv',textchecks)
 padding=[]
 for panel in ['A','B']:
  v=[float(r['value']) for r in traces if r['panel']==panel];lo,hi=LIMITS[panel];span=max(v)-min(v);padding.append({'panel':panel,'data_min':min(v),'data_max':max(v),'ylim_min':lo,'ylim_max':hi,'lower_padding_fraction':(min(v)-lo)/span,'upper_padding_fraction':(hi-max(v))/span})
 write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/f'FigureS2.{ext}',dpi=600,metadata={'Title':TITLE})
 (Q/'layout.json').write_text(json.dumps({'title':TITLE,'dimensions_mm':[W,H],'ylims':LIMITS,'jitter_offsets':[-.1,0,.1],'seed_colors':dict(zip(['42/1234','43/1235','44/1236'],COLORS)),'median_bar_definition':'unweighted median of three original seed-specific plotted values within each setting and metric','median_bars':8,'seed_connecting_lines':0,'legend':'inside A lower left','legend_title':'log1p / scale seed','marker_tests':len(checks),'median_bar_tests':len(linechecks),'text_tests':len(textchecks),'marker_clipping':0,'text_clipping':0,'plotted_values':len(traces)},indent=2)+'\n');plt.close(f)
 print(json.dumps({'ylims':LIMITS,'marker_tests':len(checks),'text_tests':len(textchecks),'values':len(traces)},indent=2))
if __name__=='__main__':main()
