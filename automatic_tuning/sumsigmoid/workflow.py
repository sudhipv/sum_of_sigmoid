"""Training-only preparation and ML; optional sampling is explicitly separate."""
from pathlib import Path
import csv,time
import numpy as np
from . import config as configuration
from .io import save,read,sha,load_training,plain
from .candidates import prepare as propose
from .optimization import Problem,one_start,OPTIONS
from .acceptance import run_batches

def run(config_path,enable_tmcmc=False,output_override=None):
    """Prepare one training prefix and jointly fit all K+3 parameters.

    Frozen detection/ML functions stay separate from output orchestration.
    Only an explicit flag/configuration can proceed to optional sampling.
    """
    config=configuration.load(config_path);folder=Path(config['output_directory'])
    if output_override is not None:
        folder=Path(output_override).resolve();config['output_directory']=str(folder)
    if folder.exists():raise FileExistsError('Output directory already exists;choose a new path to preserve results')
    folder.mkdir(parents=True);save(folder/'configuration.json',config)
    started=time.perf_counter()
    try:
        y,cutoff=load_training(config)
        print(f'[prepare] Loaded {len(y)} daily training observations; cutoff={cutoff}',flush=True)
        np.savetxt(folder/'training_observations.csv',y,delimiter=',',fmt='%.17g')
        save(folder/'input_record.json',dict(file_sha256=sha(config['data']['file']),training_observations=len(y),
            model_days=[0,cutoff],quantity='infectious counts in persons',preprocessing='none;detector-only seven-day smoother'))
        result=propose(y,cutoff,config['procedure'])
        print(f'[detection] {len(result["locations"])} fixed candidates: {result["locations"]}',flush=True)
        for name,value in [('detection',result['raw']),('identical_date_groups',result['identical_date_groups']),
            ('legacy_consolidation',result['legacy']),('optional_turn_consolidation',result['optional_turn']),('candidates',result['groups'])]:save(folder/(name+'.json'),value)
        locations=result['locations'];problem=Problem(y,locations,config)
        save(folder/'ML_bounds.json',dict(parameter_names=problem.names,lower=problem.low,upper=problem.high,
            initial_start=problem.initial_start,scale=problem.scale,optimizer='SLSQP',options=OPTIONS,
            objective='full normalized NLL;optimizer uses original fixed NLL/160 divisor',assumptions_not_confidence_intervals=True))
        def fit_one(p,start,index):
            # Keep completed attempts even if a later feasible-start search fails.
            import json
            try:call=one_start(p,start,'joint_ML',index)
            except Exception as exc:
                call=dict(start=index,initial_parameters=start.tolist(),parameters=None,NLL=None,success=False,reliable=False,status=-1,message=repr(exc))
                with (folder/'ML_calls.jsonl').open('a') as stream:stream.write(json.dumps(plain(call),allow_nan=False)+'\n')
                raise
            with (folder/'ML_calls.jsonl').open('a') as stream:stream.write(json.dumps(plain(call),allow_nan=False)+'\n')
            print(f'[ML] start {index+1}: NLL={call["NLL"]}; status={call["status"]}; reliable={call["reliable"]}',flush=True)
            return call
        try:fit=run_batches(problem,fit_one,int(config['seed']),folder/'optimizer_attempts.json')
        except Exception as exc:
            save(folder/'status.json',dict(status='unresolved',stage='starting_point_generation_or_optimization',reason=str(exc),TMCMC_started=False))
            raise
        decision=fit['decision'];p=decision['best_parameters']
        save(folder/'ML_result.json',dict(status=decision['status'],accepted=decision['accepted'],locations=locations,
            parameter_names=problem.names,parameters=p,NLL=decision['best_NLL'],optimizer_decision=decision))
        if p is not None:
            save_fit(folder,problem,np.asarray(p),y,locations)
            with (folder/'ML_parameters.csv').open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['parameter','activation_day','value','amplitude_sign'])
                for j,name in enumerate(problem.names):
                    is_amplitude=1<=j<=len(locations)
                    writer.writerow([name,locations[j-1] if is_amplitude else '',p[j],int(np.sign(p[j])) if is_amplitude else ''])
        if not decision['accepted']:
            save(folder/'status.json',dict(status='unresolved',reason=decision['reason'],attempts=decision['attempts'],TMCMC_started=False))
            return dict(status='unresolved',output=str(folder),reason=decision['reason'],
                NLL=decision['best_NLL'],outputs=output_paths(folder,p is not None,False))
        signs=np.sign(np.array(p)[1:-2]).astype(int).tolist()
        lower=[0.]+[-.5 if s<0 else 0. for s in signs]+[0.,0.]
        upper=[.5]+[0. if s<0 else .5 for s in signs]+[.001*config['population']]*2
        save(folder/'TMCMC_priors.json',dict(parameter_names=problem.names,locations=locations,signs=signs,lower=lower,upper=upper,
            kind='independent base Uniforms with training-grid beta admissibility factor',
            signs_from='accepted joint full-training ML',assumptions_not_ML_confidence_intervals=True,
            selection_and_location_uncertainty_excluded=True))
        # Scientific settings are distinct from the verbatim original engine.
        from .sampler_provenance import load_original
        _,_,source=load_original()
        save(folder/'TMCMC_configuration.json',dict(sampler_source=source,
            settings=config['tmcmc'],model=config['model'],denominator='living S+E+I+R',
            population=config['population'],parameter_names=problem.names,locations=locations,
            lower=lower,upper=upper,free_E0_I0=True,beta_admissibility_tolerance=1e-12,
            original_parallel_rng_can_duplicate=True,no_sampling_during_preparation=True))
        script=Path(__file__).resolve().parents[1]/'run.py'
        command=f'python "{script}" tmcmc --prepared "{folder}"'
        (folder/'optional_TMCMC_command.txt').write_text(command+'\n')
        save(folder/'status.json',dict(status='accepted',attempts=decision['attempts'],preparation_seconds=time.perf_counter()-started,
            TMCMC_started=False,optional_command=command))
        if enable_tmcmc or config['tmcmc']['enabled']:
            from .optional_tmcmc import run as sample
            sample(folder)
        return dict(status='accepted',output=str(folder),locations=locations,NLL=decision['best_NLL'],
            outputs=output_paths(folder,True,True),optional_TMCMC_command=command)
    except Exception as exc:
        if not (folder/'status.json').exists():save(folder/'status.json',dict(status='error',reason=str(exc),TMCMC_started=False))
        raise

def output_paths(folder,has_fit,has_priors):
    """Always print the concrete files a reviewer should inspect."""
    return dict(ML_result=str(folder/'ML_result.json'),
        fitted_parameters=str(folder/'ML_parameters.csv') if has_fit else None,
        training_metrics=str(folder/'training_metrics.json') if has_fit else None,
        training_figure=str(folder/'training_fit.pdf') if has_fit else None,
        TMCMC_configuration=str(folder/'TMCMC_configuration.json') if has_priors else None,
        TMCMC_priors=str(folder/'TMCMC_priors.json') if has_priors else None,
        optimizer_attempts=str(folder/'optimizer_attempts.json'),status=str(folder/'status.json'))

def save_fit(folder,problem,p,y,locations):
    ll,states=problem.model.evaluate(p,keep_states=True);beta=problem.beta(p);every=round(1/problem.model.settings['dt'])
    rmse=float(np.sqrt(np.mean((states[::every,2]-y)**2)))
    save(folder/'training_metrics.json',dict(NLL=-ll,RMSE=rmse,beta_minimum=float(beta.min()),
        minimum_compartment=float(states.min()),parameter_order=problem.names))
    np.savez_compressed(folder/'ML_training_trajectory.npz',time=problem.model.grid,states=states,beta=beta,parameters=p)
    with (folder/'fitted_values.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['model_day','observed_I','S','E','I','R','D','beta'])
        for j,row in enumerate(states[::every]):w.writerow([j,y[j],*row,beta[j*every]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    axes[0].scatter(np.arange(len(y)),y,c='#555555',s=10,label='Training observations')
    axes[0].plot(problem.model.grid,states[:,2],label='Joint ML infectious trajectory');axes[0].set_ylabel('Persons');axes[0].legend()
    axes[1].plot(problem.model.grid,beta);axes[1].set(xlabel='Model day',ylabel='Beta (day$^{-1}$)')
    for ax in axes:
        for date in locations:ax.axvline(date,color='#999999',lw=.5,alpha=.4)
    fig.suptitle('Training-only ML fit;no forecast or posterior intervals')
    fig.savefig(folder/'training_fit.pdf');plt.close(fig)
