"""Export only freshly verified final baselines into a standalone handoff.

Run from a source checkout. Scientific results are supplied as explicit paths;
no old experiment is implicitly selected. No optimization or sampling occurs.
"""
from pathlib import Path
import argparse,csv,json,shutil,sys,hashlib,time,re
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(HERE))
import handoff_tools as tools
from sumsigmoid.sampler_provenance import load_original


def stage_source(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in ['handoff_tools.py','forward_model.ipynb','README.md','DATA_LICENSES.txt','requirements-notebook.txt']:
        if (HERE/name).exists():shutil.copy2(HERE/name,output/name)
    for name in ['requirements.txt','LICENSE','CITATION.cff','SAMPLER_REFERENCE.json']:
        shutil.copy2(SOURCE/name,output/name)
    for f in (SOURCE/'sumsigmoid').rglob('*'):
        if f.is_file() and f.suffix in ['.py','.json']:
            dest=output/'sumsigmoid'/f.relative_to(SOURCE/'sumsigmoid');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    return output

def public_provenance(value):
    """Keep hashes/settings while removing machine-specific historical paths."""
    if isinstance(value,dict):return {k:public_provenance(v) for k,v in value.items()}
    if isinstance(value,list):return [public_provenance(v) for v in value]
    if isinstance(value,str) and Path(value).is_absolute():return 'archived_source/'+Path(value).name
    return value


def audit(folder):
    complete=tools.read(folder/'tmcmc/sampling_complete.json');checks=tools.read(folder/'tmcmc/training_checks.json')
    priors=tools.read(folder/'TMCMC_priors.json');ml=tools.read(folder/'ML_result.json')
    raw=np.load(folder/'tmcmc/particles.npz');p=raw['particles']
    if p.shape!=(4000,len(priors['parameter_names'])):raise RuntimeError('Particle dimension mismatch')
    _,_,source=load_original()
    for name in ['tmcmc_mod.py','pdfs.py']:
        # Absolute import paths legitimately differ between archived runners.
        if complete['original_source'][name]['sha256']!=source[name]['sha256']:
            raise RuntimeError('Original-source provenance mismatch')
    controls=complete.get('mutation_controls',dict(policy='original_adaptive'))
    if controls['policy']=='fixed50_experimental':
        executed=complete['executed_sampler']
        if executed['sha256']!='5f38aad7fce446037f43f16db24c582b310c038fc7f28babdfbef12da08a2142':
            raise RuntimeError('Unverified experimental control derivative')
    if not complete['fresh_prior_exact_match'] or checks['max_direct_likelihood_error']>1e-7:raise RuntimeError('Provenance/likelihood failure')
    gate=tools.read(folder/'training_review.json')
    if not gate['forecast_authorized_after_review']:raise RuntimeError('Training review did not pass')
    low=np.array(priors['lower']);high=np.array(priors['upper']);width=high-low
    if not np.isfinite(p).all() or np.any(p<low) or np.any(p>high):raise RuntimeError('Invalid posterior parameter support')
    rows=[]
    for j,name in enumerate(priors['parameter_names']):
        q=np.quantile(p[:,j],[.025,.5,.975]);amplitude=0<j<len(p[0])-2
        rows.append(dict(parameter=name,activation_day=priors['locations'][j-1] if amplitude else None,
            units='persons' if name in ['E0','I0'] else 'day^-1',ML=ml['parameters'][j],mean=float(p[:,j].mean()),median=float(q[1]),q025=float(q[0]),q975=float(q[2]),
            lower=float(low[j]),upper=float(high[j]),fraction_in_lower_1pct=float(np.mean(p[:,j]<=low[j]+.01*width[j])),
            fraction_in_upper_1pct=float(np.mean(p[:,j]>=high[j]-.01*width[j])),
            fraction_in_lower_5pct=float(np.mean(p[:,j]<=low[j]+.05*width[j])),fraction_in_upper_5pct=float(np.mean(p[:,j]>=high[j]-.05*width[j])),
            exact_lower_hits=int(np.sum(p[:,j]==low[j])),exact_upper_hits=int(np.sum(p[:,j]==high[j]))))
    log=(folder/'tmcmc/sampler_status.txt').read_text();accept=[float(v) for v in re.findall(r'acceptance rate = ([\d.]+)',log)]
    stages=np.load(folder/'tmcmc/original_stage_records.npz');stage_rows=[];constant=checks['Uniform_logprior_constant']
    for j in range(complete['stages']+1):
        values=-(stages[f'engine_value_{j}']+constant);finite=values[np.isfinite(values)]
        # Row j is the population at exponent p_j; next_temperature_j is p_(j+1).
        exponent=0. if j==0 else float(stages[f'next_temperature_{j-1}'])
        final=j==complete['stages']
        stage_rows.append(dict(population=j,tempering_exponent=exponent,best_NLL=float(finite.min()),median_NLL=float(np.median(values)),finite_subset_median_NLL=float(np.median(finite)),
            finite_particles=len(finite),importance_ESS_next=None if final else float(stages[f'ESS_{j}']),
            next_temperature=None if final else float(stages[f'next_temperature_{j}']),logged_acceptance_to_next=accept[j] if j<len(accept) else None))
    report=dict(**checks,sampling_seconds=complete['sampling_seconds'],stages=complete['stages'],
        mutation_controls=controls,executed_sampler=complete.get('executed_sampler',source['tmcmc_mod.py']),
        logged_acceptance_range=[min(accept),max(accept)],importance_ESS_range=[min(s['importance_ESS_next'] for s in stage_rows[:-1]),max(s['importance_ESS_next'] for s in stage_rows[:-1])],
        unique_final_particles=int(len(np.unique(p,axis=0))),warnings=[l for l in log.splitlines() if 'warn' in l.lower()],
        original_worker_policy='No initializer; shared spawn-import seed; duplicate initial RNG states possible',
        importance_ESS_is_not_chain_ESS=True,no_ordinary_chain_diagnostics_from_resampled_indices=True)
    return rows,stage_rows,report


def write_table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export_case(run_folder,case,input_folder,output):
    run_folder=Path(run_folder);input_folder=Path(input_folder);output=stage_source(output)
    dest=output/'cases'/case
    if dest.exists():raise FileExistsError('Case output already exists; no overwrite or rerun')
    rows,stages,diagnostics=audit(run_folder);dest.mkdir(parents=True)
    config=tools.read(run_folder/'configuration.json');config['data']['file']='observed_daily.csv';config['data']['date_column']='day';config['data']['count_column']='I';config['time_origin']=0;config['training_cutoff']=159;config['output_directory']='notebook_generated';config.pop('configuration_file',None)
    # Calendar origin is explicitly retained separately when data are numeric days.
    base=tools.base_case(case)
    config['calendar_origin']='2020-04-01' if base=='toronto' else None
    tools.write(dest/'configuration.json',config)
    for name in ['ML_result.json','TMCMC_priors.json','training_observations.csv']:
        shutil.copy2(run_folder/name,dest/name)
    model_names=tools.read(run_folder/'TMCMC_priors.json')['parameter_names'];dates=tools.read(run_folder/'TMCMC_priors.json')['locations']
    columns=['A0']+[f'A_day{d}' for d in dates]+['E0','I0']
    meta=dict(model_parameter_order=model_names,export_columns=columns,locations=dates,
        array_orientation='draw,time',sample_csv_orientation='draws in rows; named parameters in columns',particle_count=4000,
        units=['day^-1']*(len(dates)+1)+['persons','persons'],observation_noise_seed=7106+tools.CASES.index(base),
        observation_noise='Independent epsilon~Normal(0,0.1^2); I_obs=I*(1+epsilon); no clipping or rounding',
        analytic_predictive_quantiles='Equal-weight mixture over all4000particles; independent of finite I_obs draw noise',
        inference_days=list(range(160)),fit_scoring_days=list(range(1,160)),forecast_scoring_days=list(range(160,200)),
        calendar_origin=config['calendar_origin'],source_publication_start_commit='7612237bcf273ffebcef15da79f0c63c6176e334',
        source_model_sha256=tools.digest(SOURCE/'sumsigmoid/model.py'),signed_priors_are_conditional_on_joint_training_ML=True,
        location_uncertainty_excluded=True,all_particles_preserved=True)
    tools.write(dest/'parameter_order.json',meta)
    raw=np.load(run_folder/'tmcmc/particles.npz');p=raw['particles'];weights=raw['weights']
    np.savetxt(dest/'posterior_samples.csv',np.column_stack([np.arange(len(p)),weights,p]),delimiter=',',header=','.join(['draw_id','weight']+columns),comments='',fmt='%.17g')
    np.savez_compressed(dest/'posterior_particles.npz',parameters=p,weights=weights,draw_id=np.arange(len(p)))
    y=np.loadtxt(input_folder/({'case1':'synthetic_case1_data.csv','case2':'synthetic_case2_data.dat','toronto':'30-Toronto.csv'}[base]),delimiter=',')[:200]
    observed_mask=np.arange(200)<160 if base=='case1' else np.ones(200,dtype=bool)
    np.savetxt(dest/'observed_daily.csv',np.column_stack([np.arange(200),y,observed_mask]),delimiter=',',header='day,I,is_observed',comments='',fmt='%.17g')
    tools.write(dest/'observation_availability.json',dict(original_values_preserved=True,
        missing_heldout_noisy_observations=base=='case1',
        explanation='Case1 original generator allocates zeros and writes noise only before d_cut=160. Days160-199 are placeholders, not observations; exclude using is_observed. Saved generating truth is available.' if base=='case1' else 'All200exported daily values are available observed/noisy counts. Inference still uses only0-159.'))
    import datetime
    with (dest/'time_coordinates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['model_day','calendar_date','inference','fit_score','forecast_score'])
        for d in range(200):w.writerow([d,(datetime.date(2020,4,1)+datetime.timedelta(days=d)).isoformat() if base=='toronto' else '',d<=159,1<=d<=159,d>=160])
    if base=='case1':
        a=np.loadtxt(input_folder/'synthetic_case1_truth_all.csv',delimiter=',');b=np.loadtxt(input_folder/'synthetic_case1_truth_beta.csv',delimiter=',')
        np.savez_compressed(dest/'generating_curves.npz',time=np.arange(0,200,.1),S=a[:,0],E=a[:,1],I=a[:,2],R=a[:,3],D=a[:,4],beta=b)
        originals=['synthetic_case1_data.csv','synthetic_case1_truth_all.csv','synthetic_case1_truth_beta.csv','synthetic_case1_truth.csv']
    elif base=='case2':
        a=np.loadtxt(input_folder/'synthetic_case2_truth.dat');np.savez_compressed(dest/'generating_curves.npz',time=a[0],S=a[1],E=a[2],I=a[3],R=a[4],D=a[5],beta=a[6]);originals=['synthetic_case2_data.dat','synthetic_case2_truth.dat']
    else:originals=['30-Toronto.csv']
    for n in originals:(dest/'original_inputs').mkdir(exist_ok=True);shutil.copy2(input_folder/n,dest/'original_inputs'/n)
    tools.write(dest/'original_input_hashes.json',{n:tools.digest(input_folder/n) for n in originals})
    write_table(dest/'parameter_summaries.csv',rows);write_table(dest/'stage_diagnostics.csv',stages);tools.write(dest/'sampler_diagnostics.json',public_provenance(diagnostics))
    # Source/config provenance without leaking machine-specific paths into bundle.
    plan=tools.read(run_folder/'tmcmc/configuration.json')
    for record in plan['original_source'].values():
        if not isinstance(record,dict):continue
        for k in list(record):
            if 'path' in k and isinstance(record[k],str) and record[k].startswith('/'):record[k]='sumsigmoid/original/'+Path(record[k]).name
    tools.write(dest/'sampling_configuration.json',public_provenance(plan))
    if diagnostics['mutation_controls']['policy']=='fixed50_experimental':
        source=Path(diagnostics['executed_sampler']['actual_import'])
        provenance=dest/'provenance';provenance.mkdir()
        shutil.copy2(source,provenance/'executed_fixed50_sampler.py')
    shutil.copy2(run_folder/'tmcmc/direct_likelihood_checks.csv',dest/'direct_likelihood_checks.csv')
    tools.write(dest/'validation_status.json',dict(training_checks_passed=True,forecast_domain_passed=False,
        status='pending forward-domain check; not cleared for distribution',
        posterior_convergence_established=False,case2_reliability='see top-level STATUS.json'))
    begun=time.perf_counter();native,daily,compact,metrics=tools.generate(output,case,allow_pending=True)
    tools.write(dest/'validation_status.json',dict(training_checks_passed=True,forecast_domain_passed=True,
        status='passed deterministic training and full-domain physical checks; scientific review required',
        posterior_convergence_established=False,case2_reliability='see top-level STATUS.json'))
    reference=dest/'reference';reference.mkdir()
    for name,a in [('latent_predictive_native',native),('daily',daily),('summaries',compact)]:np.savez_compressed(reference/(name+'.npz'),**a)
    tools.write(reference/'metrics.json',metrics)
    tools.figures(output,case,native,daily,compact,reference/'figures')
    tools.write(dest/'export_timing.json',dict(export_and_plots_seconds=time.perf_counter()-begun))
    print(json.dumps(dict(case=case,metrics=metrics,sampler=diagnostics,output=str(dest)),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=tools.CASE_OPTIONS,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export_case(a.run,a.case,a.inputs,a.output)

if __name__=='__main__':main()
