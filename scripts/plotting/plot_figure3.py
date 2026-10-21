"""Figure3 display-only revision. Reads frozen Top2All summaries; no model/metric code."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import os
ROOT = Path(__file__).resolve().parents[2]
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'Figure3'
S = ROOT / 'source_data/Figure3/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Condition labels do not consistently improve reconstruction beyond the sparse expression profile.'
SEEDS=[42,43,44];COLORS=['#0072B2','#D55E00','#009E73'];MODES=['no_condition','condition_aware'];W_MM=85;H_MM=200
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.3,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':False})
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'source_data/Figure3/source_manifest.json').read_text())
 for r in manifest['files']:assert sha(S/r['copy'])==r['sha256']
 rows=read(S/'fig3ab_condition_vs_nocond_3seeds.csv');knn=read(S/'fig3c_condition_fingerprint_knn.csv')
 def value(seed,mode,metric):
  rr=[r for r in rows if r['seed']==str(seed) and r['model_type']==mode and r['metric']==metric];assert len(rr)==1 and int(rr[0]['n_genes_evaluable'])==6951
  return rr[0]['median']
 f=plt.figure(figsize=(W_MM/25.4,H_MM/25.4))
 def axis(bottom,top):return f.add_axes([.25,bottom/H_MM,.71,(top-bottom)/H_MM])
 a=axis(142,185);b=[axis(108,129),axis(79,100),axis(50,71)];c=axis(12,37)
 marker_records=[];traces=[];a_rows=[];b_rows=[];c_rows=[]
 def dot(ax,x,y,color,mode,panel,metric,seed,size):
  obj=ax.scatter([x],[y],s=size,facecolor='white' if mode=='no_condition' else color,edgecolor=color,linewidth=1,zorder=3)
  marker_records.append((ax,obj,panel,metric,seed,mode))
 for j,metric in enumerate(['pearson','spearman']):
  for k,seed in enumerate(SEEDS):
   xx=np.array([j-.14,j+.14])+.025*(k-1);yy=[float(value(seed,mode,metric)) for mode in MODES]
   a.plot(xx,yy,color=COLORS[k],lw=.8,zorder=2)
   for mode,x,y in zip(MODES,xx,yy):
    dot(a,x,y,COLORS[k],mode,'A',metric,seed,23);v=value(seed,mode,metric);a_rows.append(dict(metric=metric,seed_log1p=seed,seed_scale=seed+1192,condition_mode=mode,value=v,display_x=format(x,'.17g')))
    traces.append(dict(panel='A',source_file='fig3ab_condition_vs_nocond_3seeds.csv',source_selector=json.dumps(dict(seed=seed,model_type=mode,metric=metric),sort_keys=True),source_column='median',value=v))
 a.set_xticks([0,1],['Pearson','Spearman']);a.set_xlim(-.35,1.35);a.set_ylim(.64,.82);a.set_ylabel('Median gene-wise correlation')
 seedlegend=a.legend(handles=[Line2D([],[],color=color,label=str(seed),lw=1) for color,seed in zip(COLORS,SEEDS)],loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False,fontsize=7.3,borderaxespad=.15,columnspacing=1.0,handlelength=1.5)
 a.add_artist(seedlegend)
 statuslegend=a.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec='black',ms=4,label='no-condition'),Line2D([],[],marker='o',ls='',mfc='black',mec='black',ms=4,label='condition-aware')],loc='lower left',bbox_to_anchor=(.015,.015),frameon=False,fontsize=7.3,borderaxespad=.15,handlelength=1,handletextpad=.5,labelspacing=.35)
 limits={};padding=[]
 for ax,metric,quant in zip(b,['rmse','mae','r2'],[.005,.005,.01]):
  vals=[float(value(seed,mode,metric)) for seed in SEEDS for mode in MODES];lo=min(vals);hi=max(vals);span=hi-lo
  ymin=round(math.floor((lo-.10*span)/quant)*quant,12);ymax=round(math.ceil((hi+.10*span)/quant)*quant,12);limits[metric]=[ymin,ymax]
  ax.set_ylim(ymin,ymax);ax.set_xlim(-.5,2.5)
  for k,seed in enumerate(SEEDS):
   xx=[k-.12,k+.12];yy=[float(value(seed,mode,metric)) for mode in MODES];ax.plot(xx,yy,color=COLORS[k],lw=.8,zorder=2)
   for mode,x,y in zip(MODES,xx,yy):
    dot(ax,x,y,COLORS[k],mode,'B',metric,seed,22);v=value(seed,mode,metric);b_rows.append(dict(metric=metric,seed_log1p=seed,seed_scale=seed+1192,condition_mode=mode,value=v,display_x=format(x,'.17g')))
    traces.append(dict(panel='B',source_file='fig3ab_condition_vs_nocond_3seeds.csv',source_selector=json.dumps(dict(seed=seed,model_type=mode,metric=metric),sort_keys=True),source_column='median',value=v))
  ax.set_xticks(range(3),['42','43','44']);ax.set_ylabel({'rmse':'RMSE\n(log1p)','mae':'MAE\n(log1p)','r2':'R²'}[metric]);ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3));ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f' if metric=='r2' else '%.3f'))
  padding.append(dict(metric=metric,data_min=lo,data_max=hi,ylim_min=ymin,ylim_max=ymax,lower_pad_fraction=(lo-ymin)/span,upper_pad_fraction=(ymax-hi)/span))
  assert lo-ymin>=.10*span-1e-12 and ymax-hi>=.10*span-1e-12
 kk=[1,3,5];yy=[]
 for k in kk:
  r=next(r for r in knn if int(r['k'])==k);yy.append(float(r['same_condition_fraction']));c_rows.append(dict(k=k,observed_fraction=r['same_condition_fraction'],selected_neighbor_rows=r['selected_neighbor_rows'],chance_fraction='0.125',chance_definition='1/8 = 2/16'))
  traces.append(dict(panel='C',source_file='fig3c_condition_fingerprint_knn.csv',source_selector=json.dumps(dict(k=k),sort_keys=True),source_column='same_condition_fraction',value=r['same_condition_fraction']))
 observed,=c.plot(kk,yy,'o-',color='#555555',lw=1,ms=6,label='Observed fraction');chance=c.axhline(.125,color='.65',ls='--',lw=.7,label='Chance = 1/8 = 0.125');c.set_xticks(kk);c.set_xlabel('Number of neighbors, k');c.set_ylabel('Same-condition\nfraction');c.set_ylim(0,1.12)
 clegend=c.legend(handles=[observed,chance],loc='upper right',frameon=False,fontsize=7.1,borderaxespad=.15,handlelength=1.8,handletextpad=.45,labelspacing=.3)
 for ax,panel,title in [(a,'A','Condition label effects'),(b[0],'B','Paired metric values'),(c,'C','Sparse-input condition fingerprint')]:
  ax.set_title(title,loc='left',pad=8);ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
 def audit(dpi):
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer();out=[]
  for ax,obj,panel,metric,seed,mode in marker_records:
   rect=ax.get_window_extent(renderer);path=obj.get_paths()[0].vertices;size=float(obj.get_sizes()[0]);edge=float(obj.get_linewidths()[0]);rx=max(abs(path[:,0]))*math.sqrt(size)*dpi/72+edge*dpi/144+1;ry=max(abs(path[:,1]))*math.sqrt(size)*dpi/72+edge*dpi/144+1
   x,y=ax.transData.transform(obj.get_offsets()[0]);margins=[x-rx-rect.x0,rect.x1-x-rx,y-ry-rect.y0,rect.y1-y-ry];clipped=any(v<0 for v in margins);out.append(dict(dpi=dpi,panel=panel,metric=metric,seed=seed,condition_mode=mode,min_edge_clearance_px=min(margins),clipped=clipped))
  # C uses Line2D markers: include edge width and one raster-pixel guard.
  rect=c.get_window_extent(renderer);radius=(observed.get_markersize()+observed.get_markeredgewidth())*dpi/144+1
  for k,y in zip(kk,yy):
   x0,y0=c.transData.transform([k,y]);mm=[x0-radius-rect.x0,rect.x1-x0-radius,y0-radius-rect.y0,rect.y1-y0-radius];out.append(dict(dpi=dpi,panel='C',metric='fraction',seed='',condition_mode=str(k),min_edge_clearance_px=min(mm),clipped=any(v<0 for v in mm)))
  assert not any(r['clipped'] for r in out),out
  # Chance/observed legend stays inside C and does not cover observed markers or connecting segments.
  cb=clegend.get_window_extent(renderer);assert rect.contains(cb.x0,cb.y0) and rect.contains(cb.x1,cb.y1)
  for x,y in c.transData.transform(np.column_stack([kk,yy])):assert not cb.expanded(1.03,1.03).contains(x,y)
  assert not MplPath(c.transData.transform(np.column_stack([kk,yy]))).intersects_bbox(cb.expanded(1.02,1.02),filled=False),'C legend covers observed line'
  assert cb.y0>c.transData.transform([1,.125])[1]+dpi/72,'Legend covers chance line'
  for ax in f.axes:
   for t in ax.texts:
    r=t.get_window_extent(renderer);assert r.x0>=0 and r.y0>=0 and r.x1<=f.bbox.width and r.y1<=f.bbox.height
  return out,dict(C_legend_bbox=list(cb.bounds),C_axes_bbox=list(rect.bounds),A_ticks=[t.get_text() for t in a.get_xticklabels()],B_subpanel_gap_mm=[(b[i].get_position().y0-b[i+1].get_position().y1)*H_MM for i in range(2)])
 checks=[]
 for dpi in [100,150,600]:
  cc,geometry=audit(dpi);checks.extend(cc)
 write(Q/'marker_clipping_audit.csv',checks);write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/('Figure3.'+ext),dpi=600,metadata={'Title':TITLE})
 write(P/'Panel_A_paired_correlations.csv',a_rows);write(P/'Panel_B_paired_errors.csv',b_rows);write(P/'Panel_C_condition_fingerprint.csv',c_rows);write(P/'source_value_trace.csv',traces)
 (Q/'layout.json').write_text(json.dumps(dict(title=TITLE,dimensions_mm=[W_MM,H_MM],B_ylims=limits,marker_clipping=0,marker_instances_per_dpi=33,checked_dpi=[100,150,600],C_legend_location='inside upper right',C_legend_entries=['Observed fraction','Chance = 1/8 = 0.125'],C_line_style='observed solid circles; chance dashed',error_bars_present=False,geometry=geometry),indent=2));plt.close(f)
 print(json.dumps(dict(dimensions_mm=[W_MM,H_MM],B_ylims=limits,clipped_markers=0),indent=2))
if __name__=='__main__':main()
