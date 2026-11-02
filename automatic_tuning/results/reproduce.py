"""Reproduce compact result figures and scores from bundled daily summaries.

No model integration, fitting, sampling, external archives or network calls.
Inference examples live separately in ../examples and contain training data only.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from wis11 import QUANTILES,metrics,independent_checks

HERE=Path(__file__).resolve().parent
CASES=['case1','case2','toronto']
METHODS=['manual','automatic','turn']
NAMES={'case1':'Synthetic case one','case2':'Synthetic case two','toronto':'Toronto'}
LABELS={'manual':'Manual locations','automatic':'Automatic location selection','turn':'Additional consolidation'}
COLORS={0:'#202020',-5:'#267bb5',5:'#d06a19'}

def load_and_check():
    """Check the published tables, including WIS and its independent example."""
    schema=json.loads((HERE/'summary_schema.json').read_text())
    for name,digest in schema['sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Summary checksum mismatch: '+name)
    with (HERE/'daily_summaries.csv').open(newline='')as f:rows=list(csv.DictReader(f))
    with (HERE/'timing_metrics.csv').open(newline='')as f:records=list(csv.DictReader(f))
    configs=json.loads((HERE/'timing_configurations.json').read_text())
    groups={};maximum_error=0.
    for c in configs:
        rr=sorted([r for r in rows if r['id']==c['id']],key=lambda r:int(r['day']))
        if [int(r['day'])for r in rr]!=list(range(200)):raise ValueError('Unexpected daily grid')
        a={key:np.array([float(r[key])if r[key]else np.nan for r in rr])for key in
            ['day','I_mean','beta_mean','beta_q025','beta_q975','training_observation','target','generating_beta']}
        a['q']=np.array([[float(r[f'I_q{i:02d}'])for r in rr]for i in range(23)])
        np.testing.assert_allclose(QUANTILES,schema['quantile_levels'],atol=1e-14,rtol=0)
        record=next(r for r in records if r['id']==c['id'])
        for prefix,days in [('fit',np.arange(1,160)),('forecast',np.arange(160,200))]:
            score=metrics(a['target'],a['q'],a['I_mean'],days)
            for field in ['WIS11','RMSE']+(['coverage95','mean_width95','signed_bias']if prefix=='forecast'else[]):
                error=abs(score[field]-float(record[prefix+'_'+field]));maximum_error=max(maximum_error,error)
                if error>1e-7:raise ValueError('Published metric mismatch: '+c['id']+'/'+field)
        if len(c['locations'])+3!=int(record['p']):raise ValueError('Parameter count mismatch')
        expected='fixed50_experimental'if c['case']=='case2'else'original_adaptive'
        if c['scientific_configuration']['tmcmc']['mutation_policy']!=expected:raise ValueError('Mutation policy mismatch')
        if c['case']=='case2'and 156 in c['locations']:raise ValueError('Unapproved provisional date')
        groups[c['id']]=(c,a,record)
    if len(groups)!=27:raise ValueError('Expected all27 configurations')
    return groups,dict(status='PASS',configurations=27,days_per_configuration=200,
        maximum_metric_error=maximum_error,WIS_independent_check=independent_checks(),
        new_fits=0,new_sampling=0,new_forecast_propagation=0)

def plot(groups,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(output/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    def config(case,method,shift):
        return next(v for v in groups.values()if v[0]['case']==case and v[0]['procedure']==method and v[0]['shift']==shift)
    def backdrop(ax,case,a,beta=False):
        t=a['day']
        if not beta:
            ax.scatter(t[:160],a['training_observation'][:160],c='#888888',s=8,alpha=.6,label='Training observations',zorder=4)
            if case=='toronto':ax.scatter(t[160:],a['target'][160:],facecolors='none',edgecolors='#444444',s=13,lw=.7,label='Held-out observations',zorder=4)
            else:ax.plot(t,a['target'],c='#b62424',ls='--',lw=1.1,label='Generating I')
        elif case!='toronto':ax.plot(t,a['generating_beta'],c='#b62424',ls='--',lw=1.1,label='Generating beta')
        ax.axvline(160,c='#777777',ls=':',lw=1.2);ax.grid(alpha=.2)
        ax.set(xlim=(0,200),xlabel='Model day',ylabel=r'$\beta$ (day$^{-1}$)'if beta else'Infectious / active counts (persons)')
    def curve(ax,c,a,beta=False):
        color=COLORS[c['shift']];label='Baseline'if c['shift']==0 else f"{c['shift']:+d} days";t=a['day']
        mean,lo,hi=(a['beta_mean'],a['beta_q025'],a['beta_q975'])if beta else(a['I_mean'],a['q'][1],a['q'][-2])
        ax.plot(t,mean,c=color,lw=1.4,label=label);ax.fill_between(t,lo,hi,color=color,alpha=.12)
    fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
    for case,ax in zip(CASES,axs):
        c,a,_=config(case,'automatic',0);backdrop(ax,case,a);curve(ax,c,a)
        kind='observation-predictive'if case=='toronto'else'latent'
        ax.set_title(f"{NAMES[case]} — Automatic location selection, {len(c['locations'])} sigmoids; 95% {kind} interval")
        ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Accepted automatic baselines — total-population denominator\nSaved daily summaries; 4,000 particles. Cutoff at day160.',fontsize=12)
    save(fig,'accepted_baselines')
    for beta,training,name in [(False,False,'timing_sensitivity'),(True,False,'beta_sensitivity'),(False,True,'training_detail')]:
        fig,axs=plt.subplots(3,3,figsize=(17,12),layout='constrained')
        for i,case in enumerate(CASES):
            for j,method in enumerate(METHODS):
                ax=axs[i,j];c,a,_=config(case,method,0);backdrop(ax,case,a,beta);ys=[a['training_observation'][:160],a['target'][:160]]
                for shift in [0,-5,5]:
                    cc,aa,_=config(case,method,shift);curve(ax,cc,aa,beta);ys.extend([aa['q'][1,:160],aa['q'][-2,:160]])
                if training:
                    lo=min(float(np.min(y))for y in ys);hi=max(float(np.max(y))for y in ys);pad=.06*(hi-lo)
                    ax.set(xlim=(0,160),ylim=(max(0,lo-pad),hi+pad))
                ax.set_title(f"{NAMES[case]}: {LABELS[method]}, {len(c['locations'])} sigmoids",fontsize=9)
                ax.legend(fontsize=7,ncol=2)
        label='Training-only detail'if training else'Beta sensitivity'if beta else'Training and forecast sensitivity'
        fig.suptitle(label+' — saved daily means and pointwise 95% intervals\nSynthetic I: latent; Toronto I: observation-predictive; beta: latent.',fontsize=12)
        save(fig,name)
    fig,axs=plt.subplots(3,1,figsize=(10,9),layout='constrained')
    for case,ax in zip(CASES,axs):
        ax.axis('off');ax.set_title(NAMES[case]);cells=[]
        for method in METHODS:cells.append([f"{float(config(case,method,s)[2]['forecast_WIS11']):,.2f}"for s in [0,-5,5]])
        table=ax.table(cellText=cells,rowLabels=[LABELS[m]for m in METHODS],colLabels=['Baseline','−5 days','+5 days'],cellLoc='center',bbox=[.34,.12,.64,.75]);table.auto_set_font_size(False);table.set_fontsize(10)
    fig.suptitle('Forecast WIS — 11 central intervals plus median; lower is better\nDaily days160–199. Our implementation; Brandon-code equivalence unverified.',fontsize=12)
    save(fig,'WIS11')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--output',type=Path,default=HERE.parent/'outputs/summary_figures')
    args=parser.parse_args();groups,report=load_and_check()
    if not args.check_only:plot(groups,args.output);report['figures']=str(args.output.resolve())
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
