"""Display-only Figure2 r1; frozen Top2All metrics in, no training/inference/evaluation."""
from pathlib import Path
import csv,json,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,cbook,ticker
from matplotlib.lines import Line2D
import os
ROOT = Path(__file__).resolve().parents[2]
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'Figure2'
S = ROOT / 'source_data/Figure2/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
METHODS=['Top2All','GlobalMean','SameConditionMean'];LABELS=['Top2All','Global\nmean','Same-cond.\nmean'];COLORS={'Top2All':'#0072B2','GlobalMean':'#999999','SameConditionMean':'#009E73','knn1':'#D55E00','knn3':'#E69F00','knn5':'#CC79A7'}
WIDTH_MM=85;HEIGHT_MM=205;DISPLAY_JITTER_SEED=20261019;BOX_WIDTH=.45
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':False})
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 meta=json.loads((ROOT/'source_data/Figure2/source_manifest.json').read_text())
 for r in meta['files']:assert sha(S/Path(r['source']).name)==r['sha256']
 a=read(S/'fig2a_samplewise_baselines.csv');g=read(S/'fig2bc_genewise_true_variable.csv');ds=read(S/'fig2d_samplewise_method_ranking.csv');dg=read(S/'fig2d_genewise_method_ranking.csv')
 assert len(a)==72 and len(g)==6951*3
 sampleids=sorted({r['sample_id'] for r in a});assert len(sampleids)==24
 rng=np.random.default_rng(DISPLAY_JITTER_SEED);jitter=dict(zip(sampleids,rng.uniform(-.10,.10,24)))
 ar=[];br=[];cr=[];dr=[];stats=[];report=[]
 f,axs=plt.subplots(4,1,figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4));f.subplots_adjust(left=.25,right=.97,bottom=.075,top=.965,hspace=.65)
 for i,m in enumerate(METHODS):
  aa=sorted([r for r in a if r['method']==m],key=lambda r:r['sample_id']);assert len(aa)==24 and {r['sample_id'] for r in aa}==set(sampleids)
  yy=np.array([float(r['pearson']) for r in aa]);xx=np.array([i+jitter[r['sample_id']] for r in aa]);assert np.isfinite(yy).all()
  axs[0].scatter(xx,yy,s=13,color=COLORS[m],alpha=.82,edgecolor='none',zorder=3)
  med=statistics.median(yy);axs[0].plot([i-.20,i+.20],[med,med],color='black',lw=1.1,zorder=4)
  for r,x in zip(aa,xx):ar.append(dict(method=m,sample_id=r['sample_id'],pearson=r['pearson'],display_x=format(x,'.17g')))
  report.append(dict(panel='A',method=m,n=24,median=med))
 axs[0].set_ylim(.990,1.000);axs[0].set_yticks([.990,.995,1.000]);axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'));axs[0].set_ylabel('Sample-wise Pearson')
 for panel,ax,col,out in [('B',axs[1],'pearson',br),('C',axs[2],'mae',cr)]:
  bs=[]
  for i,m in enumerate(METHODS):
   gg=[r for r in g if r['method']==m];assert len(gg)==6951 and len({r['gene'] for r in gg})==6951;assert all(r['true_variable'].lower()=='true' and int(r['n_samples'])==24 for r in gg)
   if i==0:reference_genes={r['gene'] for r in gg}
   assert {r['gene'] for r in gg}==reference_genes
   values=np.array([float(r[col]) for r in gg]);assert np.isfinite(values).all()
   b=cbook.boxplot_stats(values,whis=1.5)[0];b['label']=m;bs.append(b)
   for r in gg:out.append(dict(method=m,gene=r['gene'],n_samples=r['n_samples'],true_variable=r['true_variable'],**{col:r[col]}))
   stats.append(dict(panel=panel,method=m,n_genes=6951,metric=col,median=b['med'],q1=b['q1'],q3=b['q3'],whislo=b['whislo'],whishi=b['whishi'],n_fliers=len(b['fliers']),flier_points_displayed=False,definition='box=IQR; whiskers=furthest observed values within 1.5 IQR; all 6951 genes used'))
   report.append(dict(panel=panel,method=m,n=6951,median=b['med']))
  artists=ax.bxp(bs,positions=range(3),widths=BOX_WIDTH,showfliers=False,patch_artist=True,manage_ticks=False,medianprops=dict(color='black',linewidth=1.1),whiskerprops=dict(linewidth=.8),capprops=dict(linewidth=.8))
  for box,m in zip(artists['boxes'],METHODS):box.set(facecolor=COLORS[m]+'30',edgecolor=COLORS[m],linewidth=.9)
  for j,m in enumerate(METHODS):
   for x in artists['whiskers'][2*j:2*j+2]+artists['caps'][2*j:2*j+2]:x.set_color(COLORS[m])
  ax.set_ylabel('Gene-wise Pearson' if panel=='B' else 'Gene-wise MAE (log1p)')
  if panel=='B':ax.set_ylim(-1,1);ax.set_yticks([-1,-.5,0,.5,1])
  else:ax.set_ylim(0,max(b['whishi'] for b in bs)*1.10)
 for ax in axs[:3]:ax.set_xticks(range(3),LABELS);ax.set_xlim(-.5,2.5)
 # D retains the existing six coordinates, method colors and marker mapping.
 for method in ['seed42_1234','seed43_1235','seed44_1236','knn1','knn3','knn5']:
  x=next(r for r in ds if r['method']==method);y=next(r for r in dg if r['method']==method);is_model=method.startswith('seed');color=COLORS['Top2All' if is_model else method];marker='o' if is_model else {'knn1':'s','knn3':'^','knn5':'D'}[method]
  axs[3].scatter(float(x['pearson_median']),float(y['pearson_median']),color=color,marker=marker,s=25,zorder=3)
  dr.append(dict(method=method,seed_log1p=method[4:6] if is_model else '',seed_scale=method.split('_')[1] if is_model else '',samplewise_pearson=x['pearson_median'],genewise_pearson=y['pearson_median'],sample_summary_column='pearson_median',gene_summary_column='pearson_median'))
 axs[3].set_xlabel('Sample-wise Pearson');axs[3].set_ylabel('Gene-wise Pearson');axs[3].ticklabel_format(axis='x',style='plain',useOffset=False);axs[3].locator_params(axis='x',nbins=3)
 legend=axs[3].legend(handles=[Line2D([],[],marker=mk,color=COLORS[m],ls='',markersize=4,label=label) for m,mk,label in [('Top2All','o','Top2All'),('knn1','s','kNN1'),('knn3','^','kNN3'),('knn5','D','kNN5')]],loc='lower left',bbox_to_anchor=(.025,.025),frameon=False,borderaxespad=.1,handlelength=1,handletextpad=.4,labelspacing=.25,fontsize=7.5)
 for ax,panel,title in zip(axs,'ABCD',['Sample-wise','Gene-wise','Gene-wise','Strict fold-local evaluation']):
  ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom');ax.set_title(title,loc='left',pad=8)
 f.canvas.draw();renderer=f.canvas.get_renderer();lb=legend.get_window_extent(renderer);ab=axs[3].get_window_extent(renderer)
 assert ab.contains(lb.x0,lb.y0) and ab.contains(lb.x1,lb.y1),'Legend outside D'
 for r in dr:
  px,py=axs[3].transData.transform([float(r['samplewise_pearson']),float(r['genewise_pearson'])]);assert not lb.expanded(1.08,1.08).contains(px,py),'Legend overlaps a data point'
 for ax in axs:
  for t in ax.texts:
   b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=f.bbox.width and b.y1<=f.bbox.height
 title='High sample-wise correlation overestimates reconstruction performance.'
 for ext in ['png','pdf','svg']:f.savefig(R/('Figure2.'+ext),dpi=600,metadata={'Title':title})
 write(P/'Panel_A_samplewise_Pearson.csv',ar);write(P/'Panel_B_genewise_Pearson.csv',br);write(P/'Panel_C_genewise_MAE.csv',cr);write(P/'Panel_D_strict_scatter.csv',dr);write(P/'boxplot_display_statistics.csv',stats)
 write(P/'median_display_statistics.csv',report)
 geometry=dict(dimensions_mm=[WIDTH_MM,HEIGHT_MM],legend_location='Panel D inside lower left',legend_bbox_px=list(lb.bounds),D_axes_bbox_px=list(ab.bounds),legend_inside_D=True,legend_covers_data_points=False,figure_legends=len(f.legends),A_ylim=list(axs[0].get_ylim()),B_ylim=list(axs[1].get_ylim()),C_ylim=list(axs[2].get_ylim()),D_xlim=list(axs[3].get_xlim()),D_ylim=list(axs[3].get_ylim()),box_width=BOX_WIDTH,jitter_seed=DISPLAY_JITTER_SEED,jitter_scope='horizontal display positions only; not a training seed',showfliers=False,box_whisker_rule='1.5 IQR',panels_vertical=True)
 (Q/'geometry.json').write_text(json.dumps(geometry,indent=2));plt.close(f)
 print('Figure2 r1 rendered; exact plot-data tables saved')
if __name__=='__main__':build()
