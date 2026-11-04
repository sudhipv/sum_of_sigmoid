"""Reconstruct accepted posterior outputs with the shared inference Model.

No optimizer, sampler or detector is imported. Arrays are (draw, time), all
saved particles retain their equal weights, and no held-out value updates a
state. Summary/scoring conventions match the accepted total-population study.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
from .model import Model, DENOMINATOR
from .prepared import validate
from .io import read, save, sha
from .scoring import QUANTILES, metrics as score_metrics


def load_inputs(run_configuration, samples, data_file, truth_file=None):
    """Use a run-owned configuration, never the current preparation directory."""
    run=read(run_configuration)
    model=validate(run)
    metadata=run['posterior_export']
    if metadata.get('source_model_sha256') is not None and metadata['source_model_sha256'] != sha(Path(__file__).with_name('model.py')):
        raise RuntimeError('Posterior forward-model source hash mismatch')
    table=np.atleast_1d(np.genfromtxt(samples,delimiter=',',names=True))
    columns=metadata['columns']
    if list(table.dtype.names) != ['draw_id','weight']+columns:
        raise RuntimeError('Saved sample columns do not match the run parameter ordering')
    points=np.column_stack([table[name] for name in columns])
    weights=np.asarray(table['weight'])
    if points.shape != (metadata['particle_count'],len(model.names)):
        raise RuntimeError('Sample dimension mismatch')
    np.testing.assert_array_equal(table['draw_id'],np.arange(len(points)))
    np.testing.assert_array_equal(weights,np.full(len(points),1/len(points)))
    if sha(samples)!=metadata['samples_sha256']:raise RuntimeError('Sample file hash mismatch')
    if sha(data_file)!=metadata['observations_sha256']:raise RuntimeError('Observation file hash mismatch')
    observations=np.genfromtxt(data_file,delimiter=',',names=True)
    np.testing.assert_array_equal(observations['day'],np.arange(200))
    np.testing.assert_array_equal(observations['I'][:len(model.y)],model.y)
    lower,upper=run['priors']['lower'],run['priors']['upper']
    if np.any(points<lower) or np.any(points>upper):raise RuntimeError('Samples outside declared priors')
    truth=None
    if truth_file is not None:
        if run['dataset']['id']=='toronto':raise RuntimeError('Toronto has no generating truth')
        if sha(truth_file)!=metadata['truth_sha256']:raise RuntimeError('Generating-curve hash mismatch')
        with np.load(truth_file,allow_pickle=False) as z:truth={key:z[key] for key in z.files}
    elif run['dataset']['id']!='toronto':
        raise RuntimeError('Synthetic generating reference missing; it must not be inferred from observations')
    if run['dataset']['id']=='toronto':
        if run['dataset']['observation_quantity']!='reported active cases':
            raise RuntimeError('Toronto observation identity mismatch')
        if run['time_coordinates']['calendar_origin']!='2020-04-01':
            raise RuntimeError('Unexpected Toronto calendar origin')
        if not np.all(observations['is_observed']==1):raise RuntimeError('Toronto observations missing')
    return run,model,points,weights,observations,truth


def predictive_quantiles(infectious,sigma):
    """Accepted all-particle Gaussian-mixture CDF inversion at daily times.

    This is not the empirical quantile of one noise realization per particle.
    It reuses the accepted helper's operation order, bounds and tolerance.
    """
    out=np.empty((len(QUANTILES),infectious.shape[1]));residual=0.
    for t,col in enumerate(infectious.T):
        if not np.all(col>0):raise RuntimeError('Nonpositive observation-noise scale; no clipping')
        sd=sigma*col;lo=float(np.min(col-12*sd));hi=float(np.max(col+12*sd))
        for k,q in enumerate(QUANTILES):
            f=lambda value:float(np.mean(ndtr((value-col)/sd)))-q
            v=brentq(f,lo,hi,xtol=1e-8);out[k,t]=v;residual=max(residual,abs(f(v)))
    return out,residual


def summarize(arrays):
    out={}
    for key in ('E','I','beta'):
        values=arrays[key]
        out.update({key+'_mean':values.mean(axis=0),key+'_median':np.quantile(values,.5,axis=0),
                    key+'_q025':np.quantile(values,.025,axis=0),key+'_q975':np.quantile(values,.975,axis=0)})
    return out


def generate(model,points,weights,noise_seed,draw_id=None):
    """Run either all saved particles or one explicit row from the same file."""
    if draw_id is not None and (type(draw_id) is not int or not 0<=draw_id<len(points)):
        raise ValueError('draw_id must be None (all) or a saved integer row')
    ids=np.arange(len(points)) if draw_id is None else np.array([draw_id])
    grid=np.arange(0,200,.1)
    arrays={key:np.empty((len(ids),len(grid))) for key in ('E','I','beta')}
    ll=np.empty(len(ids));mass=0.;minimum=0.;beta_min=np.inf
    started=time.perf_counter()
    for row,index in enumerate(ids):
        forward=model.forward(points[index],199.9);states=forward['states']
        if not np.isfinite(states).all() or not np.isfinite(forward['beta']).all():
            raise RuntimeError('Nonfinite saved-particle propagation; no filtering')
        arrays['E'][row]=states[:,1];arrays['I'][row]=states[:,2];arrays['beta'][row]=forward['beta']
        ll[row]=forward['training_loglikelihood']
        mass=max(mass,float(np.max(abs(states.sum(axis=1)-model.config['population']))))
        minimum=min(minimum,float(states.min()));beta_min=min(beta_min,float(forward['beta'].min()))
    if mass>1e-6 or minimum< -1e-8 or beta_min< -1e-12:
        raise RuntimeError('Saved population fails physical-domain checks; no filtering')
    # Re-evaluate one sample independently to check indexing, order and initial
    # states against its ensemble row. No resampling or mutation takes place.
    check_id=int(ids[0]);single=model.forward(points[check_id],199.9)
    for name,vector in [('E',single['states'][:,1]),('I',single['states'][:,2]),('beta',single['beta'])]:
        np.testing.assert_array_equal(arrays[name][0],vector)
    propagation_seconds=time.perf_counter()-started
    rng=np.random.RandomState(noise_seed)
    if draw_id is not None:
        # Advance only the independent predictive-noise stream so a selected
        # row also matches its noise realization in an all-sample calculation.
        for _ in range(draw_id):rng.normal(size=len(grid))
    arrays['I_obs']=arrays['I']*(1+model.settings['sigma']*rng.normal(size=arrays['I'].shape))
    arrays.update(time=grid,draw_id=ids,weights=weights[ids],training_loglikelihood=ll)
    daily={key:arrays[key][:,::10] for key in ('E','I','beta','I_obs')}
    daily.update(time=grid[::10],draw_id=ids,weights=weights[ids])
    summary=summarize(arrays)
    diagnostics=dict(propagation_seconds=propagation_seconds,maximum_population_error_persons=mass,
        minimum_compartment=minimum,minimum_beta_full_domain=beta_min,
        selected_sample_equals_ensemble_row=True,checked_draw_id=check_id,
        no_new_sampling=True,no_optimization=True,number_of_evaluated_particles=len(ids))
    return arrays,daily,summary,diagnostics


def scores(run,model,daily,observations,truth):
    """Daily accepted scores, with synthetic latent and Toronto observed targets."""
    quantile,predictive_error=predictive_quantiles(daily['I'],model.settings['sigma'])
    daily.update(predictive_I_quantiles=quantile,predictive_quantile_levels=QUANTILES)
    if truth is None:
        target=observations['I'];q=quantile;kind='observation-predictive against observed active I'
    else:
        np.testing.assert_allclose(truth['time'][::10],np.arange(200),rtol=0,atol=1e-10)
        target=truth['I'][::10];q=np.quantile(daily['I'],QUANTILES,axis=0)
        kind='latent I against saved generating I'
    mean=daily['I'].mean(axis=0)
    result=dict(target=kind,training_RMSE_observed_days0_159=float(np.sqrt(np.mean((mean[:160]-model.y)**2))),
        maximum_predictive_CDF_residual=predictive_error,
        WIS_implementation='Published eleven-interval implementation; not verified against Brandon code',
        point_summary='posterior mean trajectory, not trajectory at posterior mean parameters',
        intervals='pointwise central quantiles; not simultaneous bands')
    for prefix,days in [('fit',np.arange(1,160)),('forecast',np.arange(160,200))]:
        result.update({prefix+'_'+key:value for key,value in score_metrics(target,q,mean,days).items()})
    return result


def verify_reference(reference,arrays,daily,summary,scored,diagnostics):
    """Compact saved references contain summaries, never large ensembles."""
    reference=Path(reference);errors={}
    with np.load(reference/'summary_reference.npz',allow_pickle=False) as expected:
        values={**summary,'time':arrays['time'],
                'training_loglikelihood':arrays['training_loglikelihood'],
                'predictive_I_quantiles':daily['predictive_I_quantiles'],
                'draw0_E':arrays['E'][0],'draw0_I':arrays['I'][0],
                'draw0_beta':arrays['beta'][0],'draw0_I_obs':arrays['I_obs'][0]}
        if set(expected.files)!=set(values):raise RuntimeError('Reference key mismatch')
        for key,value in values.items():
            np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-7,err_msg=key)
            errors[key]=float(np.max(abs(value-expected[key])))
    saved=read(reference/'metrics_reference.json')
    for key,value in scored.items():
        if key.startswith(('fit_','forecast_')) or key=='training_RMSE_observed_days0_159':
            np.testing.assert_allclose(value,saved[key],rtol=1e-12,atol=1e-7,err_msg=key)
            errors[key]=abs(value-saved[key])
    return dict(status='PASS',absolute_tolerance=1e-7,relative_tolerance=1e-12,
                errors=errors,selected_sample_equals_ensemble_row=diagnostics['selected_sample_equals_ensemble_row'],
                verification_type='deterministic saved-particle reconstruction; no new inference')


def plot(run,observations,truth,arrays,daily,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    t=arrays['time'];all_samples=len(arrays['draw_id'])>1
    for ax,key in zip(axes,('I','beta')):
        ax.plot(t,summary[key+'_mean'],color='#0072B2',label='SoS posterior mean' if all_samples else 'Selected posterior sample')
        if all_samples:
            ax.fill_between(t,summary[key+'_q025'],summary[key+'_q975'],color='#0072B2',alpha=.2,label='95% latent interval')
        if truth is not None:ax.plot(truth['time'],truth[key],color='black',ls='--',label='Generating trajectory')
        ax.axvline(160,color='#9c3b13',ls=':',label='Training cutoff / forecast starts')
        ax.set_xlim(0,199.9);ax.grid(alpha=.2)
    valid=observations['is_observed'].astype(bool)
    train=valid & (observations['day']<160);heldout=valid & ~train
    axes[0].scatter(observations['day'][train],observations['I'][train],s=12,c='#666666',label='Training observations')
    if heldout.any():axes[0].scatter(observations['day'][heldout],observations['I'][heldout],s=16,facecolors='none',edgecolors='#666666',label='Held-out observations')
    if all_samples:
        q=daily['predictive_I_quantiles'];levels=daily['predictive_quantile_levels']
        axes[0].plot(daily['time'],q[np.argmin(abs(levels-.025))],c='#D55E00',lw=.8,label='95% observation-predictive limits')
        axes[0].plot(daily['time'],q[np.argmin(abs(levels-.975))],c='#D55E00',lw=.8)
    axes[0].set_ylabel('Infectious / active cases (persons)')
    axes[1].set(ylabel='Transmission rate β (day⁻¹)',xlabel='Time (days)')
    for ax in axes:ax.legend(fontsize=8,ncol=2)
    fig.suptitle(run['dataset']['id']+': saved posterior reconstruction\nFixed data-selected dates and ML-selected signs; selection uncertainty excluded',fontsize=12)
    fig.savefig(output/'fit_forecast_beta.pdf')
    fig.savefig(output/'fit_forecast_beta.svg')
    plt.close(fig)


def regenerate(run_configuration,samples,data_file,truth_file,output,draw_id=None,reference=None):
    """Public notebook/CLI helper. These are evaluations of existing samples."""
    run,model,points,weights,observations,truth=load_inputs(run_configuration,samples,data_file,truth_file)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arrays,daily,summary,diagnostics=generate(model,points,weights,run['posterior_export']['observation_noise_seed'],draw_id)
    scored=scores(run,model,daily,observations,truth) if draw_id is None else dict(status='single sample; ensemble scores not calculated')
    verification=verify_reference(reference,arrays,daily,summary,scored,diagnostics) if reference is not None and draw_id is None else dict(status='not applicable to selected sample or no reference supplied')
    plot(run,observations,truth,arrays,daily,summary,output)
    for name,value in [('latent_predictive_native',arrays),('daily',daily),('summaries',summary)]:
        np.savez_compressed(output/(name+'.npz'),**value)
    save(output/'metrics.json',scored);save(output/'diagnostics.json',diagnostics);save(output/'verification.json',verification)
    save(output/'run_configuration.json',run)
    return dict(dataset=run['dataset']['id'],output=str(output.resolve()),metrics=scored,
                diagnostics=diagnostics,verification=verification,
                forward_model_file=str(Path(__file__).with_name('model.py').resolve()),
                forward_model_sha256=sha(Path(__file__).with_name('model.py')))


def export_inference_inputs(inference,data_file,truth_file,output,noise_seed=7106):
    """Connect a new inference output to these same posterior notebooks/helpers.

    Inputs are the run-owned snapshot and named CSV, not preparation globals.
    data_file is a separate daily day,I,is_observed table through day199; this
    explicit interface preserves the distinction between observed values and
    case-one placeholders. It is never read by inference or the likelihood.
    """
    import shutil
    inference,output=Path(inference),Path(output)
    if output.exists():raise FileExistsError('Posterior input export exists; choose a new path')
    checks=read(inference/'training_checks.json')
    if not (checks['best_NLL_gap_to_ML']<10 and checks['particles_within10NLLofML']>0):
        raise RuntimeError('Saved run failed the existing training-quality screen; no posterior forecast export')
    run=read(inference/'run_configuration.json');validate(run)
    order=read(inference/'parameter_order.json')
    expected=['A0']+[f'A_day{d}'for d in run['locations']]+['E0','I0']
    if order['export_columns']!=expected or order['model_parameter_order']!=run['parameter_names']:
        raise RuntimeError('Inference parameter export mismatch')
    output.mkdir(parents=True)
    shutil.copy2(inference/'posterior_samples.csv',output/'posterior_samples.csv')
    shutil.copy2(data_file,output/'observed_daily.csv')
    if truth_file is not None:shutil.copy2(truth_file,output/'generating_curves.npz')
    table=np.atleast_1d(np.genfromtxt(output/'posterior_samples.csv',delimiter=',',names=True))
    run['posterior_export']=dict(columns=expected,particle_count=len(table),
        samples_sha256=sha(output/'posterior_samples.csv'),observations_sha256=sha(output/'observed_daily.csv'),
        observation_noise_seed=int(noise_seed),
        source_model_sha256=sha(Path(__file__).with_name('model.py')),
        truth_sha256=sha(output/'generating_curves.npz')if truth_file is not None else None)
    from .prepared import configuration_digest
    run['content_sha256']=configuration_digest(run)
    save(output/'run_configuration.json',run)
    # Load now to reject mismatched data, sample labels, supports or identity
    # before any trajectories or scores are produced.
    load_inputs(output/'run_configuration.json',output/'posterior_samples.csv',
                output/'observed_daily.csv',output/'generating_curves.npz'if truth_file is not None else None)
    return output
