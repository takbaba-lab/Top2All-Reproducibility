"""S3-only display revision. Frozen corrected54-job summaries, no metric evaluation."""
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
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'FigureS3'
S = ROOT / 'source_data/FigureS3/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Reconstruction performance is non-monotonic with input depth.'
W=85;H=200;SEEDS=[(42,1234),(43,1235),(44,1236)];COLORS=['#0072B2','#D55E00','#009E73']
YLIMS={'A':[.68,.85],'B':[.17,.27],'C':[.735,.855],'D':[.165,.245]}
TITLES={'A':'Common target: non-Top2000\n(5,605 genes)','B':'Common target: non-Top2000\n(5,605 genes)','C':'Main-target bridge: non-Top1000\n(6,951 genes)','D':'Main-target bridge: non-Top1000\n(6,951 genes)'}
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.1,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':True})
def read(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 m=json.loads((ROOT/'source_data/FigureS3/source_manifest.json').read_text());s=S/'input_depth_metrics.tsv';assert hashlib.sha256(s.read_bytes()).hexdigest()==m['files'][0]['sha256'];assert True  # Numerical sources are checked by validate_release.py.
 rows=read(s);f=plt.figure(figsize=(W/25.4,H/25.4));axes={};traces=[];lines=[];markers=[];styles=[]
 for panel,target,metric,(bottom,top) in zip('ABCD',['primary5605','primary5605','bridge6951','bridge6951'],['pearson','mae','pearson','mae'],[(158,183),(110,135),(62,87),(14,39)]):
  ns=[100,200,300,500,1000,2000] if target=='primary5605' else [100,200,300,500,1000];count=5605 if target=='primary5605' else 6951;x=np.arange(len(ns));ax=f.add_axes([.255,bottom/H,.69,(top-bottom)/H]);axes[panel]=ax;ax.set_xlim(-.35,len(ns)-.65);ax.set_ylim(YLIMS[panel])
  def values(seed=None):
   yy=[]
   for i,n in enumerate(ns):
    selector={'target_set':target,'topn':n,'metric':metric}
    if seed is not None:selector['seed_log1p']=seed
    rr=[r for r in rows if all(r[k]==str(v) for k,v in selector.items())];assert len(rr)==(1 if seed is not None else 3) and all(r['pipeline']=='Top2All' and r['n_selected']==r['n_valid']==str(count) for r in rr)
    raw=rr[0]['value'] if seed is not None else format(statistics.median(float(r['value']) for r in rr),'.17g');yy.append(float(raw))
    traces.append({'panel':panel,'target_set':target,'topn':n,'seed_log1p':seed if seed is not None else '3-seed median','seed_scale':seed+1192 if seed is not None else '3-seed median','metric':metric,'n_genes':count,'value':raw,'source_selector':json.dumps(selector,sort_keys=True,separators=(',',':')),'source_column':'value','operation':'cell' if seed is not None else 'median','display_x':i})
   return yy
  for (sa,sb),color in zip(SEEDS,COLORS):
   yy=values(sa);line,=ax.plot(x,yy,color=color,lw=.65,alpha=.6,zorder=3);dots,=ax.plot(x,yy,ls='None',marker='o',color=color,ms=3.2,mew=.6,alpha=1,zorder=4);lines.append((panel,str(sa),line));markers.append((panel,str(sa),dots));styles.append({'panel':panel,'seed_log1p':sa,'seed_scale':sb,'color':color,'linewidth_pt':.65,'line_alpha':.6,'markersize_pt':3.2,'marker_alpha':1,'line_zorder':3,'marker_zorder':4})
  yy=values();line,=ax.plot(x,yy,'o-',color='black',lw=1.6,ms=4.6,mew=.6,alpha=1,zorder=2);lines.append((panel,'median',line));markers.append((panel,'median',line));styles.append({'panel':panel,'seed_log1p':'3-seed median','seed_scale':'3-seed median','color':'black','linewidth_pt':1.6,'line_alpha':1,'markersize_pt':4.6,'marker_alpha':1,'line_zorder':2,'marker_zorder':2})
  ax.set_xticks(x,[str(n) for n in ns]);ax.set_xlabel('Nominal input depth, TopN');ax.set_ylabel('Median gene-wise\nPearson r' if metric=='pearson' else 'Median gene-wise\nMAE (log1p)');ax.set_title(TITLES[panel],loc='left',pad=8);ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
  ax.set_yticks({'A':[.70,.75,.80],'B':[.18,.22,.26],'C':[.75,.80,.85],'D':[.18,.21,.24]}[panel]);ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 labels=['42 / 1234','43 / 1235','44 / 1236','3-seed median'];handles=[Line2D([],[],color=c,marker='o',lw=.65,ms=3.2,mew=.6,label=lab) for c,lab in zip(COLORS,labels[:3])]+[Line2D([],[],color='black',marker='o',lw=1.6,ms=4.6,mew=.6,label=labels[3])]
 legend=axes['A'].legend(handles=handles,loc='lower left',bbox_to_anchor=(.01,.015),frameon=False,fontsize=7.1,handlelength=2,handletextpad=.6,labelspacing=.35,borderaxespad=.2)
 marker_checks=[];line_checks=[];text_checks=[]
 for dpi in [100,150,600]:
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer();lb=legend.get_window_extent(renderer);a=axes['A'];assert a.bbox.contains(lb.x0,lb.y0) and a.bbox.contains(lb.x1,lb.y1)
  for panel,kind,o in markers:
   ax=axes[panel];box=ax.bbox;xy=o.get_transform().transform(o.get_xydata());radius=(o.get_markersize()+o.get_markeredgewidth())*dpi/144+1
   for i,(x,y) in enumerate(xy):
    gap=min(x-radius-box.x0,box.x1-x-radius,y-radius-box.y0,box.y1-y-radius);assert gap>0,(panel,kind,dpi,gap)
    marker_checks.append({'panel':panel,'seed_or_median':kind,'position':i,'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
    if panel=='A':assert x+radius<lb.x0 or x-radius>lb.x1 or y+radius<lb.y0 or y-radius>lb.y1,('legend/marker overlap',kind,lb)
  for panel,kind,o in lines:
   xy=o.get_transform().transform(o.get_xydata());box=axes[panel].bbox;guard=o.get_linewidth()*dpi/144+1;gap=min(xy[:,0].min()-box.x0,box.x1-xy[:,0].max(),xy[:,1].min()-box.y0,box.y1-xy[:,1].max())-guard;assert gap>0
   line_checks.append({'panel':panel,'seed_or_median':kind,'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
   if panel=='A':assert not o.get_path().transformed(o.get_transform()).intersects_bbox(lb),'legend/line overlap'
  for t in f.findobj(match=Text):
   if not t.get_visible() or not t.get_text().strip():continue
   b=t.get_window_extent(renderer);gap=min(b.x0,b.y0,f.bbox.width-b.x1,f.bbox.height-b.y1);assert gap>0,(dpi,t.get_text(),b)
   text_checks.append({'dpi':dpi,'text':t.get_text(),'min_canvas_clearance_px':float(gap),'clipped':False})
  tight=[axes[p].get_tightbbox(renderer) for p in 'ABCD']
  for upper,lower in zip(tight,tight[1:]):assert upper.y0>lower.y1,('panel collision',dpi,upper,lower)
  for p in 'ABCD':
   boxes=[t.get_window_extent(renderer) for t in axes[p].get_xticklabels()];assert all(a.x1<b.x0 for a,b in zip(boxes,boxes[1:])),('tick overlap',p)
 write(P/'exact_plot_data.csv',traces)
 for p in 'ABCD':write(P/f'Panel_{p}_exact_values.csv',[r for r in traces if r['panel']==p])
 write(Q/'marker_clipping_audit.csv',marker_checks);write(Q/'line_clipping_audit.csv',line_checks);write(Q/'text_clipping_audit.csv',text_checks);write(Q/'seed_median_style_audit.csv',styles)
 padding=[]
 for p,(lo,hi) in YLIMS.items():
  yy=[float(r['value']) for r in traces if r['panel']==p];span=max(yy)-min(yy);padding.append({'panel':p,'data_min':min(yy),'data_max':max(yy),'ylim_min':lo,'ylim_max':hi,'lower_padding_fraction':(min(yy)-lo)/span,'upper_padding_fraction':(hi-max(yy))/span})
 write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/f'FigureS3.{ext}',dpi=600,metadata={'Title':TITLE})
 (Q/'layout.json').write_text(json.dumps({'title':TITLE,'panel_titles':TITLES,'dimensions_mm':[W,H],'ylims':YLIMS,'legend_labels':labels,'legend_location':'inside A lower left','x_label':'Nominal input depth, TopN','x_axis_scale':'linear categorical equal spacing','primary_N':[100,200,300,500,1000,2000],'bridge_N':[100,200,300,500,1000],'primary_n':5605,'bridge_n':6951,'seed_markers_above_median':True,'seed_marker_zorder':4,'median_zorder':2,'Top1000_highlight':False,'plotted_values':len(traces),'marker_tests':len(marker_checks),'line_tests':len(line_checks),'text_tests':len(text_checks),'marker_clipping':0,'text_clipping':0,'legend_overlap':False},indent=2)+'\n');plt.close(f)
 print(json.dumps({'values':len(traces),'ylims':YLIMS,'marker_tests':len(marker_checks),'text_tests':len(text_checks),'dimensions_mm':[W,H]},indent=2))
if __name__=='__main__':main()
