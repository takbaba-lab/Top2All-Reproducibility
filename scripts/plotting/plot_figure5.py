"""Figure5 display-only revision; immutable Top2All source summaries and original metrics."""
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
R = Path(os.environ.get('TOP2ALL_PLOT_OUT', 'reproduced_figures')) / 'Figure5'
S = ROOT / 'source_data/Figure5/numerical_sources'
P = R / 'plot_data'
Q = R / 'display_checks'
for directory in (R, P, Q): directory.mkdir(parents=True, exist_ok=True)
TITLE='Reference retrieval provides a stronger baseline than learned reconstruction for both unseen conditions and held-out replicates.'
W=85;H=205;SEEDS=[42,43,44];SC=['#0072B2','#D55E00','#009E73']
ENC={'Top2All':('#0072B2','o'),'kNN1':('#D55E00','s'),'kNN3':('#E69F00','^'),'kNN5':('#CC79A7','D'),'ZeroShift':('#555555','o')}
YLIMS={'A':[.07,.52],'B_Pearson':[.27,.61],'B_MAE':[.05,.117],'C':[.72,.94],'D':[.09,.26]}
GROUP_X=[0,1,2,3.7,4.7,5.7];GROUP_CENTERS=[1,4.7]
# Arial must be installed and discoverable by Matplotlib.
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':7.3,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.unicode_minus':True})
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((ROOT/'source_data/Figure5/source_manifest.json').read_text())
 for r in m['files']:assert sha(S/r['copy'])==r['sha256']
 f=plt.figure(figsize=(W/25.4,H/25.4));axes={}
 for key,(bottom,top) in zip(YLIMS,[(168,193),(130,152),(100,122),(61,84),(17,40)]):
  ax=f.add_axes([.25,bottom/H,.69,(top-bottom)/H]);axes[key]=ax;ax.set_ylim(YLIMS[key])
 trace=[];tables={p:[] for p in 'ABCD'};marks=[];group_texts=[]
 def val(panel,axis,file,selector,column,operation='cell'):
  rr=[r for r in read(S/file) if all(r[k]==str(v) for k,v in selector.items())];assert rr
  if operation=='cell':assert len(rr)==1;raw=rr[0][column];value=float(raw)
  else:
   vv=[float(r[column]) for r in rr if r[column]!=''];assert len(vv)==len(rr);value=statistics.median(vv);raw=format(value,'.17g')
  trace.append({'panel':panel,'axis':axis,'source_file':file,'source_selector':json.dumps(selector,sort_keys=True,separators=(',',':')),'source_column':column,'operation':operation,'n_selected':len(rr),'value':raw})
  return value,raw
 def dot(key,x,y,color,marker,size=24):
  o=axes[key].scatter(x,y,s=size,c=color,marker=marker,linewidth=.8,zorder=3);marks.append((key,o))
 def lab(key,panel,title):
  ax=axes[key];ax.text(-.28,1.08,panel,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom');ax.set_title(title,loc='left',pad=8)
 a=axes['A'];file='fig5b_centroid_shift_3seeds.csv'
 for i,metric in enumerate(['slope','pred_true_sd_ratio']):
  for j,seed in enumerate(SEEDS):
   y,raw=val('A','A',file,{'seed':seed},metric,'median');x=i+(j-1)*.10;dot('A',x,y,SC[j],'o')
   tables['A'].append({'seed':seed,'metric':metric,'median_over_eight_conditions':raw,'display_x':repr(x),'color':SC[j],'marker':'o'})
 a.set_xlim(-.38,1.38);a.set_xticks([0,1],['Shift slope','SD ratio']);a.set_ylabel('Median over\neight conditions');lab('A','A','LOCO centroid shifts')
 seedlegend=a.legend(handles=[Line2D([],[],color=c,marker='o',markersize=3.5,ls='',label=str(s)) for c,s in zip(SC,SEEDS)],loc='center',bbox_to_anchor=(.5,.52),ncol=3,frameon=False,handlelength=.8,columnspacing=1.2,handletextpad=.5,fontsize=7.3)
 for key,metric in [('B_Pearson','pearson'),('B_MAE','mae')]:
  ax=axes[key];methods=['Top2All','kNN1','kNN3','kNN5']+(['ZeroShift'] if metric=='mae' else [])
  for j,method in enumerate(methods):
   y,raw=val('B',key,'fig5c_loco_retrieval_by_condition.csv',{'method':method},metric,'median');color,marker=ENC[method];dot(key,j,y,color,marker)
   tables['B'].append({'axis':key,'method':method,'metric':metric,'median_over_eight_conditions':raw,'display_x':j,'color':color,'marker':marker})
  ax.set_xlim(-.6,len(methods)-.4);ax.set_xticks(range(len(methods)),['Zero\nshift' if k=='ZeroShift' else k for k in methods]);ax.set_ylabel('Shift Pearson r' if metric=='pearson' else 'Shift MAE (log1p)')
  ax.set_yticks([.3,.4,.5,.6] if metric=='pearson' else [.06,.08,.10]);ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
 lab('B_Pearson','B','LOCO retrieval')
 methods=['seed42_1234','seed43_1235','seed44_1236','knn1','knn3','knn5']
 for key,column,ylabel in [('C','pearson_median','Gene-wise Pearson r'),('D','mae_median','Gene-wise MAE (log1p)')]:
  ax=axes[key]
  for x,method in zip(GROUP_X,methods):
   name='Top2All' if method.startswith('seed') else 'kNN'+method[-1];color,marker=ENC[name]
   y,raw=val(key,key,'fig5d_strict_oof_retrieval.csv',{'method':method},column);dot(key,x,y,color,marker)
   tables[key].append({'method':method,'metric':column,'value':raw,'display_x':x,'color':color,'marker':marker})
  ax.set_xlim(-.5,6.2);ax.set_xticks(GROUP_X,['42','43','44','1','3','5']);ax.set_ylabel(ylabel);lab(key,key,'Strict OOF retrieval')
  for x,txt in zip(GROUP_CENTERS,['Top2All seeds','kNN, k']):
   t=ax.text(x,-.42,txt,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8.5,clip_on=False);group_texts.append((key,x,txt,t))
 marker_checks=[];text_checks=[];group_checks=[];legend_checks=[]
 for dpi in [100,150,600]:
  f.set_dpi(dpi);f.canvas.draw();renderer=f.canvas.get_renderer()
  for key,obj in marks:
   ax=axes[key];box=ax.bbox;path=obj.get_paths()[0].get_extents();size=float(obj.get_sizes()[0]);lw=float(obj.get_linewidths()[0]);x,y=ax.transData.transform(obj.get_offsets())[0]
   rx=max(abs(path.x0),abs(path.x1))*np.sqrt(size)*dpi/72+lw*dpi/144+1;ry=max(abs(path.y0),abs(path.y1))*np.sqrt(size)*dpi/72+lw*dpi/144+1
   bottom_clearance=y-ry-box.y0;gap=min(x-rx-box.x0,box.x1-x-rx,bottom_clearance,box.y1-y-ry);mm=bottom_clearance*25.4/dpi
   assert gap>0,(key,'clipping',gap);assert mm>=1.5,(key,'insufficient visible bottom gap',mm)
   marker_checks.append({'axis':key,'dpi':dpi,'x':float(obj.get_offsets()[0,0]),'value':float(obj.get_offsets()[0,1]),'min_edge_clearance_px':float(gap),'marker_edge_to_xaxis_mm':float(mm),'clipped':False,'xaxis_overlap':False})
  for t in f.findobj(match=Text):
   if not t.get_visible() or not t.get_text().strip():continue
   b=t.get_window_extent(renderer);gap=min(b.x0,b.y0,f.bbox.width-b.x1,f.bbox.height-b.y1);assert gap>0,(dpi,t.get_text(),b)
   text_checks.append({'dpi':dpi,'text':t.get_text(),'min_canvas_clearance_px':float(gap),'clipped':False})
  for key,x,txt,t in group_texts:
   b=t.get_window_extent(renderer);center=axes[key].transData.transform((x,0))[0];error=abs((b.x0+b.x1)/2-center);assert error<1e-8
   group_checks.append({'panel':key,'dpi':dpi,'label':txt,'center_data_x':x,'center_error_px':float(error)})
  lb=seedlegend.get_window_extent(renderer);assert a.bbox.contains(lb.x0,lb.y0) and a.bbox.contains(lb.x1,lb.y1)
  assert not any(lb.contains(*axes[k].transData.transform(obj.get_offsets())[0]) for k,obj in marks if k=='A')
  legend_checks.append({'dpi':dpi,'inside_A':True,'overlap_A_markers':False})
  # Full extents include group labels; neighboring panels must remain separated.
  tight=[axes[k].get_tightbbox(renderer) for k in YLIMS]
  for upper,lower in zip(tight,tight[1:]):assert upper.y0>lower.y1,('panel-label collision',dpi,upper,lower)
  # B labels and group labels must be distinct, with no horizontal text overlap.
  for key in ['B_Pearson','B_MAE']:
   boxes=[t.get_window_extent(renderer) for t in axes[key].get_xticklabels()]
   assert all(x.x1<y.x0 for x,y in zip(boxes,boxes[1:])),('B method label collision',key)
 for panel,rows in tables.items():write(P/f'Panel_{panel}_exact_values.csv',rows)
 write(P/'source_value_trace.csv',trace);write(Q/'marker_clipping_audit.csv',marker_checks);write(Q/'text_clipping_audit.csv',text_checks);write(Q/'group_label_alignment.csv',group_checks);write(Q/'A_legend_geometry.csv',legend_checks)
 padding=[]
 for key,(lo,hi) in YLIMS.items():
  yy=[float(r['value']) for r in trace if r['axis']==key];span=max(yy)-min(yy)
  padding.append({'axis':key,'data_min':min(yy),'data_max':max(yy),'ylim_min':lo,'ylim_max':hi,'lower_padding_fraction':(min(yy)-lo)/span,'upper_padding_fraction':(hi-max(yy))/span})
 write(Q/'ylim_padding.csv',padding)
 f.set_dpi(100)
 for ext in ['png','pdf','svg']:f.savefig(R/f'Figure5.{ext}',dpi=600,metadata={'Title':TITLE})
 (Q/'layout.json').write_text(json.dumps({'title':TITLE,'dimensions_mm':[W,H],'ylims':YLIMS,'A_seed_legend':'inside center; no marker overlap','method_encoding':ENC,'group_positions':GROUP_X,'group_centers':GROUP_CENTERS,'within_group_step':1,'between_group_step':1.7,'C_metric':'gene-wise Pearson','D_metric':'gene-wise MAE','marker_tests':len(marker_checks),'text_tests':len(text_checks),'marker_clipping':0,'text_clipping':0,'marker_xaxis_overlap':0,'minimum_marker_edge_to_xaxis_mm':min(r['marker_edge_to_xaxis_mm'] for r in marker_checks),'plotted_values':len(trace),'B_ZeroShift_Pearson':'undefined, omitted'},indent=2))
 plt.close(f);print(json.dumps({'ylims':YLIMS,'marker_tests':len(marker_checks),'text_tests':len(text_checks),'minimum_marker_edge_to_xaxis_mm':min(r['marker_edge_to_xaxis_mm'] for r in marker_checks),'values':len(trace)},indent=2))
if __name__=='__main__':main()
