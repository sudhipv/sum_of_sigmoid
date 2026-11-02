"""Load saved SoS particles and regenerate trajectories/figures; never sample.

Canonical array orientation is (draw, time). All particles, including duplicates,
are retained with their final equal weights. No historical-folder imports.
"""
from pathlib import Path
import argparse,csv,json,sys,time,hashlib
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
# Source checkout: sibling sumsigmoid/. Standalone bundle: child sumsigmoid/.
SOURCE_ROOT=HERE if (HERE/'sumsigmoid').is_dir() else HERE.parent
sys.path.insert(0,str(SOURCE_ROOT))
from sumsigmoid.model import Model,DENOMINATOR

CASES=['case1','case2','toronto']
CASE_OPTIONS=CASES+['case2_seed107','case2_seed108']

def base_case(case):return 'case2' if case.startswith('case2_seed') else case

def available_cases(root):
    """Failed/unavailable cases remain in STATUS.json, never as posterior arrays."""
    cases=[]
    for c in CASE_OPTIONS:
        folder=Path(root)/'cases'/c
        if not (folder/'posterior_samples.csv').exists() or not (folder/'validation_status.json').exists():continue
        status=read(folder/'validation_status.json')
        if status.get('training_checks_passed') and status.get('forecast_domain_passed'):cases.append(c)
    return cases
QUANTILES=np.array([.01,.025,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.975,.99])
LEVELS=[10,20,30,40,50,60,70,80,90,95,98]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_case(root,case,allow_pending=False):
    folder=Path(root)/'cases'/case
    config=read(folder/'configuration.json');meta=read(folder/'parameter_order.json')
    status=read(folder/'validation_status.json')
    if not status.get('training_checks_passed') or (not status.get('forecast_domain_passed') and not allow_pending):
        raise RuntimeError('This population is not cleared for this saved-output handoff')
    if config['transmission_denominator']!=DENOMINATOR:raise RuntimeError('Denominator mismatch')
    a=np.genfromtxt(folder/'posterior_samples.csv',delimiter=',',names=True)
    p=np.column_stack([a[n] for n in meta['export_columns']]);weights=a['weight']
    if len(p)!=4000 or not np.array_equal(weights,np.full(4000,1/4000)):raise RuntimeError('Unexpected particle count/weights')
    y=np.loadtxt(folder/'training_observations.csv',delimiter=',')
    model=Model(y,meta['locations'],config)
    if len(y)!=160 or model.names!=meta['model_parameter_order']:raise RuntimeError('Training/parameter order mismatch')
    return folder,model,p,weights,meta

def predictive_quantiles(infectious):
    """Exact all-particle Gaussian-mixture CDF inversion at daily times.

    This reconstructs the approved observation distribution; it is not Brandon's
    unavailable WIS implementation. Numerical CDF accuracy is recorded separately.
    """
    out=np.empty((len(QUANTILES),infectious.shape[1]));residual=0.
    for t,col in enumerate(infectious.T):
        if not np.all(col>0):raise RuntimeError('Nonpositive observation scale; no clipping')
        sd=.1*col;lo=float(np.min(col-12*sd));hi=float(np.max(col+12*sd))
        for k,q in enumerate(QUANTILES):
            f=lambda value:float(np.mean(ndtr((value-col)/sd)))-q
            v=brentq(f,lo,hi,xtol=1e-8);out[k,t]=v;residual=max(residual,abs(f(v)))
    return out,residual

def summaries(x):
    return dict(mean=x.mean(axis=0),q025=np.quantile(x,.025,axis=0),
                median=np.quantile(x,.5,axis=0),q975=np.quantile(x,.975,axis=0))

def generate(root,case,allow_pending=False):
    """Pure posterior forward evaluation; no refit, filtering or held-out update."""
    folder,model,p,weights,meta=load_case(root,case,allow_pending);begun=time.perf_counter()
    n=len(p);grid=np.arange(0,200,.1);arrays={v:np.empty((n,len(grid))) for v in ['E','I','beta']}
    mass_error=0.;minimum=0.;min_beta=np.inf;ll=np.empty(n)
    for j,point in enumerate(p):
        result=model.forward(point,199.9);state=result['states']
        if not np.isfinite(state).all() or not np.isfinite(result['beta']).all():raise RuntimeError('Nonfinite forecast; no sample removal')
        mass_error=max(mass_error,float(np.max(abs(state.sum(axis=1)-model.config['population']))))
        minimum=min(minimum,float(state.min()));min_beta=min(min_beta,float(result['beta'].min()))
        arrays['E'][j]=state[:,1];arrays['I'][j]=state[:,2];arrays['beta'][j]=result['beta'];ll[j]=result['training_loglikelihood']
    if mass_error>1e-6 or minimum< -1e-8 or min_beta< -1e-12:
        raise RuntimeError(f'Invalid forecast domain: mass={mass_error}, compartment={minimum}, beta={min_beta}; no filtering')
    propagation_seconds=time.perf_counter()-begun
    native={**arrays,'time':grid,'weights':weights,'draw_id':np.arange(n),'training_loglikelihood':ll}
    noise_started=time.perf_counter();rng=np.random.RandomState(meta['observation_noise_seed'])
    # One independent Gaussian realization per draw/time. No truncation/rounding.
    native['I_obs']=arrays['I']*(1+.1*rng.normal(size=arrays['I'].shape))
    daily={v:values[:,::10] for v,values in native.items() if v in ['E','I','beta','I_obs']}
    daily.update(time=grid[::10],weights=weights,draw_id=np.arange(n))
    q,residual=predictive_quantiles(daily['I']);daily['predictive_quantile_levels']=QUANTILES;daily['predictive_I_quantiles']=q
    noise_seconds=time.perf_counter()-noise_started
    started=time.perf_counter();ss={v:summaries(native[v]) for v in ['E','I','beta']}
    mean=ss['I']['mean'][::10];y=model.y;forecast=np.arange(160,200)
    data=np.genfromtxt(folder/'observed_daily.csv',delimiter=',',names=True)
    target=np.asarray(data['I'])[forecast];target_type='observed active I'
    if (folder/'generating_curves.npz').exists():
        truth=np.load(folder/'generating_curves.npz');target=truth['I'][::10][forecast];target_type='saved generating latent I'
    lo,hi=q[1,forecast],q[-2,forecast]
    if base_case(case)!='toronto':lo=ss['I']['q025'][::10][forecast];hi=ss['I']['q975'][::10][forecast]
    metric=dict(training_RMSE_observed_days0_159=float(np.sqrt(np.mean((mean[:160]-y)**2))),
        fit_RMSE_observed_days1_159=float(np.sqrt(np.mean((mean[1:160]-y[1:160])**2))),
        heldout_RMSE_daily_days160_199=float(np.sqrt(np.mean((mean[forecast]-target)**2))),
        heldout_signed_bias_prediction_minus_target=float(np.mean(mean[forecast]-target)),
        heldout_coverage95=float(np.mean((lo<=target)&(target<=hi))),heldout_mean_interval_width95=float(np.mean(hi-lo)),
        heldout_target=target_type,interval_type='observation-predictive' if base_case(case)=='toronto' else 'latent',
        evaluation_days=forecast.tolist(),WIS=None,WIS_status='pending Brandon exact scorer; not substituted with historical three-interval WIS',
        quantile_method_latent='numpy.quantile linear; empirical equal-weight particles',
        quantile_method_predictive='all-particle normalized Gaussian mixture; scipy.brentq CDF inversion',
        maximum_predictive_CDF_residual=residual,minimum_forecast_compartment=minimum,
        minimum_beta_full_domain=min_beta,maximum_population_error_persons=mass_error,
        propagation_seconds=propagation_seconds,observation_predictive_seconds=noise_seconds,
        summary_and_diagnostic_scoring_seconds=time.perf_counter()-started,
        inference_conditions='Fixed data-selected locations and ML-selected signs; selection uncertainty excluded',
        no_new_sampling=True,no_convergence_claim=True)
    compact={v+'_'+k:value for v,s in ss.items() for k,value in s.items()}
    return native,daily,compact,metric

def compare(root,case,native,daily,compact,metric):
    """Compare deterministic arrays/intervals to reference, not timings or plots."""
    base=Path(root)/'cases'/case/'reference';errors={}
    for name,values in [('latent_predictive_native',native),('daily',daily),('summaries',compact)]:
        with np.load(base/(name+'.npz')) as saved:
            if set(saved.files)!=set(values):raise RuntimeError('Reference array-name mismatch')
            for key,v in values.items():
                error=float(np.max(abs(np.asarray(v)-saved[key])))
                if not np.allclose(v,saved[key],rtol=1e-12,atol=1e-7):raise RuntimeError(f'Reference mismatch: {name}/{key} {error}')
                errors[name+'/'+key]=error
    ref=read(base/'metrics.json')
    for k,v in metric.items():
        if isinstance(v,(float,int)) and not isinstance(v,bool) and not k.endswith('_seconds'):
            if not np.isclose(v,ref[k],rtol=1e-12,atol=1e-7):raise RuntimeError('Metric mismatch:'+k)
    return dict(status='PASS',array_errors=errors,absolute_tolerance=1e-7,relative_tolerance=1e-12,
                exact_same_model_import=digest(Path(SOURCE_ROOT)/'sumsigmoid/model.py'),no_sampling=True)

def figures(root,case,native,daily,compact,output):
    """Paper-style blue SoS panels, with explicit equal-tail interval semantics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42})
    folder,model,p,weights,meta=load_case(root,case);out=Path(output);out.mkdir(exist_ok=True,parents=True)
    observed=np.genfromtxt(folder/'observed_daily.csv',delimiter=',',names=True);truth=None
    if (folder/'generating_curves.npz').exists():truth=np.load(folder/'generating_curves.npz')
    blue='#1f77b4';t=native['time'];titles={'case1':'Synthetic case one','case2':'Synthetic case two','toronto':'Toronto'}
    sampling=read(folder/'sampling_configuration.json')
    seed=sampling.get('seed',106)
    policy=sampling.get('mutation_controls',{}).get('policy','original_adaptive')
    control_label='experimental fixed 50 attempts' if policy=='fixed50_experimental' else 'original adaptive 5–10 attempts'
    title=titles[base_case(case)]+f' — seed {seed}; '+control_label
    fig,axes=plt.subplots(3,2,figsize=(15,9),layout='constrained')
    for row,v in enumerate(['beta','E','I']):
        label={'beta':'Infection rate\n'+r'$\beta(t)$ (day$^{-1}$)','E':'Exposed compartment\n'+r'$E(t)$ (persons)','I':'Infectious compartment\n'+r'$I(t)$ (persons)'}[v]
        for col in range(2):
            ax=axes[row,col];ax.plot(t,compact[v+'_mean'],c=blue,lw=2,label='SoS mean')
            ax.fill_between(t,compact[v+'_q025'],compact[v+'_q975'],color=blue,alpha=.22,label='95% latent interval')
            if v=='I':
                q=daily['predictive_I_quantiles'];td=daily['time']
                ax.plot(td,q[1],c=blue,lw=1,ls=':',label='95% observation-predictive interval (daily)')
                ax.plot(td,q[-2],c=blue,lw=1,ls=':')
                valid=observed['is_observed']>0
                ax.scatter(observed['day'][valid],observed['I'][valid],s=12,facecolors='white',edgecolors='#333333',linewidths=.6,label='Raw observed active I',zorder=5)
            if truth is not None:ax.plot(truth['time'],truth[v],c='tab:red',ls='--',lw=1.5,label='Generating truth')
            ax.axvline(160,color='black',ls='--',lw=1,label='Inference 0–159; forecast starts 160')
            ax.set(xlim=(-1,161) if col==0 else (158,200),xlabel='Time (days)',ylabel=label)
            ax.grid(True,alpha=.45)
            # Select y limits by the visible interval, not the hidden full horizon.
            visible=(t<160.1) if col==0 else (t>=159)
            ys=[compact[v+'_q025'][visible],compact[v+'_q975'][visible]]
            if truth is not None:ys.append(truth[v][visible])
            if v=='I':
                dmask=(daily['time']<=160) if col==0 else (daily['time']>=159)
                visible_obs=valid & ((observed['day']<160) if col==0 else (observed['day']>=159))
                ys.extend([q[1,dmask],q[-2,dmask],observed['I'][visible_obs]])
            low=min(float(x.min()) for x in ys);high=max(float(x.max()) for x in ys);pad=max(.07*(high-low),1e-6)
            ax.set_ylim(max(0,low-pad),high+pad)
        axes[row,0].set_title('Training fit');axes[row,1].set_title('Forecast (no state reset or future assimilation)')
    handles,labels=axes[2,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    note='Toronto E/beta have no known truth' if case=='toronto' else 'Generating curves preserved; no future data assimilated'
    fig.suptitle(title+': SoS; fixed locations and ML-selected signs\nPointwise equal-tail intervals; '+note,fontsize=13)
    fig.savefig(out/'fit_forecast.pdf',bbox_inches='tight');fig.savefig(out/'fit_forecast.png',dpi=140,bbox_inches='tight');plt.close(fig)
    # Marginals represent final particles, not ordinary independent MCMC chains.
    ncols=4;nrows=(p.shape[1]+ncols-1)//ncols
    fig,axs=plt.subplots(nrows,ncols,figsize=(12,2.4*nrows),layout='constrained')
    ml=np.asarray(read(folder/'ML_result.json')['parameters']);prior=read(folder/'TMCMC_priors.json')
    for j,ax in enumerate(axs.flat):
        if j>=p.shape[1]:ax.axis('off');continue
        ax.hist(p[:,j],bins=35,density=True,color=blue,alpha=.35,label='Final SoS particles')
        ax.axvline(ml[j],c='black',ls='--',lw=1.2,label='Joint ML')
        ax.set_title(meta['export_columns'][j],fontsize=11);ax.set_ylabel('Density');ax.grid(alpha=.3)
        ax.text(.02,.98,f"Prior [{prior['lower'][j]:.4g}, {prior['upper'][j]:.4g}]",transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle(title+': posterior marginals, conditional on dates/signs\nNot chain traces; no generating coefficient recovery claim',fontsize=12)
    fig.savefig(out/'posterior_diagnostic.pdf',bbox_inches='tight');fig.savefig(out/'posterior_diagnostic.png',dpi=120,bbox_inches='tight');plt.close(fig)

def regenerate(root,case,output,verify=True):
    started=time.perf_counter();out=Path(output)/case;out.mkdir(parents=True,exist_ok=True)
    native,daily,compact,metric=generate(root,case)
    verification=compare(root,case,native,daily,compact,metric) if verify else dict(status='not_requested')
    figures(root,case,native,daily,compact,out)
    for name,a in [('latent_predictive_native',native),('daily',daily),('summaries',compact)]:np.savez_compressed(out/(name+'.npz'),**a)
    write(out/'metrics.json',metric);write(out/'verification.json',verification)
    return dict(case=case,output=str(out),metrics=metric,verification=verification,regeneration_seconds=time.perf_counter()-started)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=HERE);ap.add_argument('--case',choices=CASE_OPTIONS+['all'],default='all');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    for case in available_cases(a.root) if a.case=='all' else [a.case]:
        result=regenerate(a.root,case,a.output,a.verify);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
