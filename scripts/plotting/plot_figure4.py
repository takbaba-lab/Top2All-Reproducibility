"""Figure4-only display revision: frozen Top2All scalar summaries; no ML/metric evaluation."""
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
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'Figure4'
S = ROOT / 'source_data/Figure4/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Top2All captures condition-associated transcriptomic structure but poorly reconstructs replicate-specific variation.'
SEEDS=[42,43,44];COLORS=['#0072B2','#D55E00','#009E73'];BLUE=COLORS[0];W=85;H=200
YLIMS={'A':[.16,.58],'B':[.045,.100],'C':[-.045,.012],'D':[0,.12]}
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.3,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':True})
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((ROOT/'source_data/Figure4/source_manifest.json').read_text())
 for r in m['files']:assert sha(S/r['copy'])==r['sha256']
 f=plt.figure(figsize=(W/25.4,H/25.4));axes={}
 for panel,(bottom,top) in zip('ABCD',[(156,188),(108,140),(63,92),(14,44)]):
  ax=f.add_axes([.25,bottom/H,.69,(top-bottom)/H]);axes[panel]=ax;ax.set_ylim(YLIMS[panel])
 markers=[];segments=[];trace=[];tables={p:[] for p in 'ABCD'}
 def val(panel,file,selector,column,operation='cell'):
  rr=[r for r in read(S/file) if all(r[k]==str(v) for k,v in selector.items())];assert rr
  if operation=='cell':assert len(rr)==1;raw=rr[0][column];value=float(raw)
  else:value=statistics.median([float(r[column]) for r in rr]);raw=format(value,'.17g')
  rec={'panel':panel,'source_file':file,'source_selector':json.dumps(selector,sort_keys=True,separators=(',',':')),'source_column':column,'operation':operation,'value':raw}
  trace.append(rec);return value,raw
 def dot(panel,x,y,color,open=False,size=25,lw=1):
  obj=axes[panel].scatter(x,y,s=size,facecolor='white' if open else color,edgecolor=color,linewidth=lw,zorder=3)
  markers.append((panel,obj));return obj
 def line(panel,x,y,**kw):
  obj,=axes[panel].plot(x,y,**kw);segments.append((panel,obj));return obj
 def label(panel,title):
  ax=axes[panel];ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom');ax.set_title(title,loc='left',pad=8)
 file='fig4b_oof_sample_residuals.csv';a=axes['A']
 for j,mode in enumerate(['no_condition','condition_aware']):
  rr=[r for r in read(S/file) if r['model_type']==mode];assert len(rr)==24
  xx=j+np.linspace(-.10,.10,24);yy=[]
  for x,r in zip(xx,rr):
   y,raw=val('A',file,{'sample_id':r['sample_id'],'model_type':mode},'resid_pearson');yy.append(y)
   tables['A'].append({'sample_id':r['sample_id'],'model_type':mode,'resid_pearson':raw,'display_x':repr(float(x))})
  dot('A',xx,yy,BLUE,open=(j==0),size=15,lw=.7)
  y,raw=val('A',file,{'model_type':mode},'resid_pearson','median');line('A',[j-.2,j+.2],[y,y],color='black',lw=1)
 a.set_xlim(-.4,1.4);a.set_xticks([0,1],['No condition','+ condition']);a.set_ylabel('Residual Pearson r');label('A','OOF replicate residuals')
 file='fig4c_oof_residual_summary.csv';b=axes['B']
 for j,(mode,col,name,color) in enumerate([('no_condition','overall_baseline_mae','SameConditionMean',COLORS[2]),('no_condition','overall_top2all_mae','Top2All no condition',BLUE),('condition_aware','overall_top2all_mae','Top2All + condition',BLUE)]):
  y,raw=val('B',file,{'model_type':mode},col);dot('B',j,y,color,open=(j==1),size=30)
  tables['B'].append({'method':name,'model_type':mode,'metric':'true_expression_overall_mae','value':raw,'display_x':j})
 b.set_xlim(-.45,2.45);b.set_xticks([0,1,2],['Same-cond.\nmean','Top2All\nno cond.','Top2All\n+ cond.']);b.set_ylabel('Measured-expression\noverall MAE (log1p)');b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'));label('B','OOF prediction error')
 file='fig4d_loco_residual_genewise_3seeds.csv';c=axes['C']
 for j,seed in enumerate(SEEDS):
  yy=[]
  for x,metric in enumerate(['pearson','spearman']):
   y,raw=val('C',file,{'seed':seed,'metric':metric},'median');yy.append(y)
   tables['C'].append({'seed':seed,'metric':metric,'median':raw,'display_x':x})
  line('C',[0,1],yy,color=COLORS[j],lw=.8);dot('C',[0,1],yy,COLORS[j],size=9,lw=.8)
 c.axhline(0,color='.6',ls='--',lw=.7);c.set_xlim(-.25,1.25);c.set_xticks([0,1],['Pearson','Spearman']);c.set_ylabel('Median residual\ncorrelation');label('C','LOCO replicate residuals')
 legend=c.legend(handles=[Line2D([],[],marker='o',markersize=3,color=col,label=str(seed),lw=.8) for col,seed in zip(COLORS,SEEDS)],loc='center',bbox_to_anchor=(.5,.5),ncol=3,frameon=False,fontsize=7.3,handlelength=1.4,columnspacing=1.1,handletextpad=.6)
 file='fig4e_residual_amplitude_3seeds.csv';d=axes['D']
 for j,seed in enumerate(SEEDS):
  y,raw=val('D',file,{'seed':seed},'predicted_true_ratio');dot('D',j,y,COLORS[j],size=25)
  tables['D'].append({'seed':seed,'predicted_true_residual_sd_ratio':raw,'display_x':j})
 d.set_xlim(-.4,2.4);d.set_xticks(range(3),['42','43','44']);d.set_xlabel('Log1p-model seed');d.set_ylabel('Predicted / measured\nresidual SD');label('D','LOCO residual amplitude')
 marker_checks=[];text_checks=[];line_checks=[]
 for dpi in [100,150,600]:
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer()
  for panel,obj in markers:
   ax=axes[panel];box=ax.bbox;positions=ax.transData.transform(obj.get_offsets());path=obj.get_paths()[0].get_extents();sizes=obj.get_sizes();widths=obj.get_linewidths()
   for i,(x,y) in enumerate(positions):
    sz=float(sizes[i%len(sizes)]);lw=float(widths[i%len(widths)]);rx=max(abs(path.x0),abs(path.x1))*np.sqrt(sz)*dpi/72+lw*dpi/144+1;ry=max(abs(path.y0),abs(path.y1))*np.sqrt(sz)*dpi/72+lw*dpi/144+1
    gap=min(x-rx-box.x0,box.x1-x-rx,y-ry-box.y0,box.y1-y-ry);assert gap>0,(panel,dpi,i,gap)
    marker_checks.append({'panel':panel,'dpi':dpi,'marker_index':i,'min_edge_clearance_px':float(gap),'clipped':False})
  for panel,obj in segments:
   xy=obj.get_transform().transform(obj.get_xydata());box=axes[panel].bbox;guard=obj.get_linewidth()*dpi/144+1
   gap=min(xy[:,0].min()-box.x0,box.x1-xy[:,0].max(),xy[:,1].min()-box.y0,box.y1-xy[:,1].max())-guard;assert gap>0,(panel,'line',gap)
   line_checks.append({'panel':panel,'dpi':dpi,'min_edge_clearance_px':float(gap),'clipped':False})
  for t in f.findobj(match=Text):
   if not t.get_visible() or not t.get_text().strip():continue
   box=t.get_window_extent(renderer);gap=min(box.x0,box.y0,f.bbox.width-box.x1,f.bbox.height-box.y1);assert gap>0,(dpi,t.get_text(),box,f.bbox)
   text_checks.append({'dpi':dpi,'text':t.get_text(),'min_canvas_clearance_px':float(gap),'clipped':False})
  lb=legend.get_window_extent(renderer);assert c.bbox.contains(lb.x0,lb.y0) and c.bbox.contains(lb.x1,lb.y1)
  for panel,obj in segments:
   if panel=='C':assert not obj.get_path().transformed(obj.get_transform()).intersects_bbox(lb),'C legend/line overlap'
  # Check neighboring axes' full title/tick/label extents do not collide vertically.
  tight=[axes[p].get_tightbbox(renderer) for p in 'ABCD']
  for upper,lower in zip(tight,tight[1:]):assert upper.y0>lower.y1,('panel text collision',dpi,upper,lower)
 write(Q/'marker_clipping_audit.csv',marker_checks);write(Q/'text_clipping_audit.csv',text_checks);write(Q/'line_clipping_audit.csv',line_checks)
 for panel,rr in tables.items():write(P/f'Panel_{panel}_exact_values.csv',rr)
 write(P/'source_value_trace.csv',trace)
 padding=[]
 for panel in 'ABCD':
  ys=[float(r['value']) for r in trace if r['panel']==panel];lo,hi=YLIMS[panel];padding.append({'panel':panel,'data_min':min(ys),'data_max':max(ys),'ylim_min':lo,'ylim_max':hi,'lower_padding':min(ys)-lo,'upper_padding':hi-max(ys)})
 write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/f'Figure4.{ext}',dpi=600,metadata={'Title':TITLE})
 (Q/'layout.json').write_text(json.dumps({'title':TITLE,'dimensions_mm':[W,H],'ylims':YLIMS,'marker_clipping':0,'text_clipping':0,'line_clipping':0,'marker_tests':len(marker_checks),'text_tests':len(text_checks),'line_tests':len(line_checks),'checked_dpi':[100,150,600],'C_xlim':[-.25,1.25],'C_legend':'inside center; no residual-line overlap','A_sample_count_per_mode':24,'B_title':'OOF prediction error','B_metric':'overall measured-expression prediction MAE (log1p), not residual MAE','D_scale':'ratio, no percent conversion'},indent=2))
 plt.close(f);print(json.dumps({'ylims':YLIMS,'marker_clipping':0,'text_clipping':0,'dimensions_mm':[W,H],'observations_and_medians':len(trace)},indent=2))
if __name__=='__main__':main()
