"""Forward calculation from existing posterior samples; no fitting or TMCMC.

Parameters/rates: day^-1; states: persons. Archives contain inputs only.
"""
from pathlib import Path
import csv, hashlib, json, textwrap
import numpy as np
STATES = ('S','E','I','R','D')

def read(path):return json.loads(Path(path).read_text())

def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def case_ids(root):return read(Path(root)/'datasets.json')['case_ids']

def configuration(path):
    path=Path(path).resolve();cfg=read(path)
    required=['locations','scientific_configuration','sample_schema','files','propagation']
    for key in required:
        if key not in cfg:raise ValueError(f'Configuration is missing {key!r}.')
    science=cfg['scientific_configuration'];model=science.get('model',{})
    locations=np.asarray(cfg['locations'],dtype=float)
    if locations.ndim!=1 or not np.isfinite(locations).all()or np.any(np.diff(locations)<=0):
        raise ValueError('Sigmoid locations must be finite and strictly ordered.')
    pop=float(science.get('population',0));step=float(model.get('dt',0))
    if pop<=0 or not np.isfinite(pop):raise ValueError('Population must be finite and positive.')
    if step<=0 or abs(round(1/step)*step-1)>1e-10:
        raise ValueError('This archived solver requires a positive timestep dividing one day.')
    for name in ['sigma','sigmoid_width','gamma_e','gamma_i','gamma_r','gamma_d']:
        value=float(model.get(name,float('nan')))
        if not np.isfinite(value)or value<0 or (name in ['sigma','sigmoid_width']and value==0):
            raise ValueError(f'Invalid model setting {name}.')
    if science.get('transmission_denominator')!='total S+E+I+R+D':
        raise ValueError('The packaged forward model requires the total-population denominator.')
    if cfg['sample_schema'].get('time_units')!='days'or cfg['sample_schema'].get('state_units')!='persons'or cfg['sample_schema'].get('rate_units')!='day^-1':
        raise ValueError('Configuration must declare time in days, states in persons, and rates in day^-1.')
    tags=[str(int(d))if float(d).is_integer()else format(d,'g').replace('-','m').replace('.','p')for d in locations]
    expected=['beta0']+['A_day'+tag for tag in tags]+['E0','I0']
    if cfg['sample_schema'].get('parameter_columns')!=expected:
        raise ValueError('Parameter-column metadata does not match the configured locations/order. Supply matching configuration metadata for a new location set.')
    if cfg.get('K',len(locations))!=len(locations)or cfg.get('p',len(expected))!=len(expected):
        raise ValueError('K/p metadata must match the configured locations and K+3 parameter schema.')
    return path,cfg

def observations(config_path,cfg):
    base=Path(config_path).parent
    p=base/cfg['files']['observations']
    try:a=np.genfromtxt(p,delimiter=',',names=True,dtype=None,encoding='utf-8')
    except Exception as exc:raise ValueError(f'Cannot read observation CSV {p.name}: {exc}')from exc
    for name in ['day','observed_I','training_mask','forecast_mask','fit_scoring_mask']:
        if name not in a.dtype.names:raise ValueError(f'Observation CSV missing {name!r}.')
    a=np.atleast_1d(a);days=np.asarray(a['day'],dtype=float);y=np.asarray(a['observed_I'],dtype=float)
    if not np.array_equal(days,np.arange(len(days)))or not np.isfinite(y).all()or np.any(y<0):
        raise ValueError('Observations require consecutive model days starting at zero and finite nonnegative counts.')
    training=np.asarray(a['training_mask'],dtype=bool);indices=np.flatnonzero(training)
    if not len(indices)or not np.array_equal(indices,np.arange(len(indices))):
        raise ValueError('Training observations must form an initial consecutive prefix.')
    if np.any(training & np.asarray(a['forecast_mask'],dtype=bool)):
        raise ValueError('Training and forecast masks overlap.')
    forecast=days[np.asarray(a['forecast_mask'],dtype=bool)]
    for field,actual in [('training_days',days[training]),('forecast_observation_days',forecast)]:
        if field in cfg and not np.array_equal(actual,np.arange(cfg[field][0],cfg[field][1]+1)):
            raise ValueError(f'{field} metadata does not match the observation mask.')
    end=float(cfg['propagation']['end_time']);step=float(cfg['scientific_configuration']['model']['dt'])
    if not np.isfinite(end)or end<days[-1]or abs(round(end/step)*step-end)>1e-10:
        raise ValueError('Propagation endpoint must cover all declared observation days and lie on the solver grid.')
    if cfg.get('calendar_origin')is not None:
        if 'date'not in a.dtype.names:raise ValueError('Calendar dates missing.')
        try:date=a['date'].astype('datetime64[D]')
        except Exception as exc:raise ValueError('Invalid calendar dates.')from exc
        if not np.array_equal(date,np.datetime64(cfg['calendar_origin'])+np.arange(len(days))):
            raise ValueError('Calendar dates do not match the declared time origin.')
    return a,y[training]

def normal_weights(weights,n):
    w=np.asarray(weights,dtype=float)
    if w.shape!=(n,)or not np.isfinite(w).all()or np.any(w<0):
        raise ValueError('Weights must be finite, nonnegative and have one value per sample.')
    total=float(w.sum())
    if not np.isfinite(total)or total<=0:raise ValueError('Weights need a finite, positive total.')
    return w/total,total

def samples(path,cfg):
    """Named CSV or labelled NPZ, any positive row count; weights are explicit."""
    path=Path(path);names=cfg['sample_schema']['parameter_columns']
    if path.suffix.lower()=='.csv':
        try:raw=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8'))
        except Exception as exc:raise ValueError(f'Cannot read sample CSV: {exc}')from exc
        fields=raw.dtype.names or ()
        missing=[n for n in names+['weight']if n not in fields]
        if missing:raise ValueError('Sample CSV missing required columns: '+', '.join(missing)+'. Explicit weights are required; no equal-weight assumption is made.')
        extra=[n for n in fields if n.startswith('A_day')and n not in names]
        if extra:raise ValueError('Unexpected amplitude columns '+', '.join(extra)+'; provide matching configuration metadata for all terms.')
        p=np.column_stack([raw[n]for n in names]);w=np.asarray(raw['weight'])
        ids=raw['original_index_zero_based']if'original_index_zero_based'in fields else np.arange(len(p))
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False)as z:
            for key in ['parameters','parameter_names','weights']:
                if key not in z:raise ValueError(f'Alternative NPZ missing {key!r}; use labelled NPZ schema or the named CSV.')
            provided=z['parameter_names'].astype(str).tolist()
            if len(set(provided))!=len(provided)or set(provided)!=set(names):
                raise ValueError('NPZ parameter names do not match the configured locations and parameter set.')
            original=z['parameters']
            if original.ndim!=2 or original.shape[1]!=len(provided):raise ValueError('NPZ parameter matrix has incompatible dimensions.')
            p=original[:,[provided.index(n)for n in names]];w=z['weights']
            ids=z['original_row_index']if'original_row_index'in z else np.arange(len(p))
            if'locations'in z and not np.array_equal(z['locations'],cfg['locations']):raise ValueError('NPZ locations do not match the configuration.')
    else:raise ValueError('Use a named .csv or labelled .npz sample file.')
    if p.ndim!=2 or p.shape[0]<1 or p.shape[1]!=len(names)or not np.isfinite(p).all():
        raise ValueError('Parameters must be a nonempty finite (samples, K+3) matrix.')
    if 'prior_support'in cfg:
        lo=np.asarray(cfg['prior_support']['lower']);hi=np.asarray(cfg['prior_support']['upper'])
        if lo.shape!=(p.shape[1],)or hi.shape!=lo.shape:raise ValueError('Declared prior supports do not match the parameter schema.')
        if np.any(p<lo-1e-12)or np.any(p>hi+1e-12):
            raise ValueError('Sample values are outside the prior supports declared by the configuration. Supply matching metadata; bounds are not widened automatically.')
    if not np.isfinite(ids).all()or np.any(ids!=np.rint(ids))or len(set(ids))!=len(ids):
        raise ValueError('Sample row IDs must be distinct integers; duplicate parameter vectors remain allowed.')
    used,total=normal_weights(w,len(p))
    pop=cfg['scientific_configuration']['population']
    if np.any(p[:,-2]<0)or np.any(p[:,-1]<=0)or np.any(p[:,-2]+p[:,-1]>pop):
        raise ValueError('Initial states require E0>=0, I0>0 and E0+I0<=population. No rows were removed.')
    return p,np.asarray(w,dtype=float),used,np.asarray(ids,dtype=int),total

def weighted_quantiles(values,weights,levels=(.025,.5,.975)):
    """Uniform positive weights: NumPy linear. Otherwise inverse weighted ECDF."""
    a=np.asarray(values,dtype=float);w,_=normal_weights(weights,len(a));positive=w>0
    if not np.isfinite(a[positive]).all():raise ValueError('Positive-weight trajectories contain nonfinite values.')
    a=a[positive];w=w[positive];w=w/w.sum()
    if np.all(w==w[0]):return np.quantile(a,levels,axis=0,method='linear')
    order=np.argsort(a,axis=0,kind='stable');ordered=np.take_along_axis(a,order,axis=0)
    mass=np.take_along_axis(np.broadcast_to(w[:,None],a.shape),order,axis=0)
    cumulative=np.cumsum(mass,axis=0);cumulative[-1]=1.
    return np.array([ordered[np.argmax(cumulative>=q,axis=0),np.arange(a.shape[1])]for q in levels])

def summaries(data):
    w,total=normal_weights(data['weights'],len(data['weights']));out={'time':data['time']}
    for name in STATES+('beta',):
        q=weighted_quantiles(data[name],w)
        out.update({name+'_mean':np.average(data[name],axis=0,weights=w),name+'_q025':q[0],name+'_median':q[1],name+'_q975':q[2]})
    return out

def make_output(path):
    path=Path(path)
    if path.exists()and any(path.iterdir()):raise ValueError(f'Output directory {path} is not empty; choose a new output directory.')
    path.mkdir(parents=True,exist_ok=True);return path

def plot_data(data,obs,cfg,summary,truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=data['time'];train_days=obs['day'][obs['training_mask'].astype(bool)];forecast_days=obs['day'][obs['forecast_mask'].astype(bool)]
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for row,var in enumerate(['I','E','beta']):
        for col,days in enumerate([train_days,forecast_days]):
            if not len(days):continue
            ax=axes[row,col];lo,hi=float(days[0]),float(days[-1]);mask=(t>=lo)&(t<=hi)
            ax.plot(t[mask],summary[var+'_median'][mask],color='#0072B2',label='Posterior model-output median')
            ax.fill_between(t[mask],summary[var+'_q025'][mask],summary[var+'_q975'][mask],color='#0072B2',alpha=.2,label='95% model-output interval')
            if var=='I':
                ax.scatter(days,obs['observed_I'][days.astype(int)],s=11,color='#555555',label='Observed infectious/active counts')
            if truth is not None:
                tm=(truth['time']>=lo)&(truth['time']<=hi)
                ax.plot(truth['time'][tm],truth[var][tm],color='black',ls='--',lw=1,label='Generating trajectory')
            ax.set(xlim=(lo,hi),xlabel='Time (days)',ylabel=r'$\beta(t)$ (day$^{-1}$)'if var=='beta'else var+' (persons)',title='Fitting'if col==0 else'Forecast')
            ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=7)
    label=cfg.get('label',cfg.get('dataset',cfg.get('id','SoS results')))
    fig.suptitle(textwrap.fill(label,width=105),fontsize=11)
    return fig

def calculate_from_samples(config_path,sample_path,output,conditioning='none'):
    from forward_model import Model
    path,cfg=configuration(config_path);obs,y=observations(path,cfg)
    sample_path=Path(sample_path)if sample_path is not None else path.parent/cfg['files']['samples']
    points,original_w,weights,ids,total=samples(sample_path,cfg)
    if conditioning not in ['none','supplied']:raise ValueError('Conditioning must be none or supplied.')
    if conditioning=='supplied':
        hist=cfg.get('historical_conditioning')
        if not hist:raise ValueError('No supplied conditioning is declared for this configuration.')
        if digest(sample_path)!=hist['sample_file_sha256']:
            raise ValueError('Supplied conditioning is restricted to the exact archived sample file. It cannot be applied to a different sample set.')
        signature=hashlib.sha256(json.dumps(dict(population=cfg['scientific_configuration']['population'],
            model=cfg['scientific_configuration']['model'],locations=cfg['locations'],end_time=cfg['propagation']['end_time'],training_counts=y.tolist()),sort_keys=True).encode()).hexdigest()
        from forward_model import __file__ as model_file
        if signature!=hist['model_signature']or digest(model_file)!=hist['forward_model_sha256']:
            raise ValueError('Supplied conditioning is invalid for changed model settings, observations or horizon. No historical mask applied.')
        excluded=np.load(path.parent/hist['exclusion_mask'],allow_pickle=False)
        if excluded.dtype!=bool:raise ValueError('Historical exclusion mask must be boolean.')
        keep=~excluded
        if keep.shape!=(len(points),):raise ValueError('Historical mask/sample dimensions differ.')
        weights=np.where(keep,weights,0.);weights,_=normal_weights(weights,len(points))
    model=Model(y,cfg['locations'],cfg['scientific_configuration']);end=float(cfg['propagation']['end_time']);step=model.settings['dt']
    grid=np.arange(round(end/step)+1)*step
    daily_index=np.flatnonzero(np.isclose(grid,np.rint(grid),rtol=0,atol=1e-10))
    for j,p in enumerate(points):
        if model.beta(p).min()< -1e-12:
            raise ValueError(f'Sample row {ids[j]} violates declared training beta support. No trajectories generated and no rows discarded.')
    data={name:np.empty((len(points),len(daily_index)))for name in STATES+('beta',)}
    records=[];nll=np.empty(len(points))
    for j,p in enumerate(points):
        if j%1000==0:print(f'Calculating samples {j+1}–{min(j+1000,len(points))} of {len(points)}',flush=True)
        result=model.forward(p,end);s=result['states'];beta=result['beta'];nll[j]=-result['training_loglikelihood']
        if not np.isfinite(nll[j])or not np.isfinite(s).all()or not np.isfinite(beta).all():
            raise ValueError(f'Sample row {ids[j]} has a nonfinite likelihood/state. No row was clipped or discarded; propagation stopped.')
        mass=float(abs(s.sum(axis=1)-model.config['population']).max())
        valid=bool(beta.min()>=-1e-12 and s.min()>=-1e-8 and mass<=1e-6)
        records.append(dict(row_id=int(ids[j]),valid=valid,minimum_beta=float(beta.min()),minimum_compartment=float(s.min()),maximum_population_error=mass,summary_weight=float(weights[j])))
        for k,name in enumerate(STATES):data[name][j]=s[daily_index,k]
        data['beta'][j]=beta[daily_index]
    data.update(time=grid[daily_index],weights=weights,original_weights=original_w,original_row_index=ids,
                locations=np.asarray(cfg['locations']),parameter_names=np.asarray(cfg['sample_schema']['parameter_columns']))
    data['physical_valid']=np.array([r['valid']for r in records],dtype=bool)
    output=make_output(output);np.savez_compressed(output/'daily.npz',**data)
    with (output/'physical_validity.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    invalid=[r['row_id']for r in records if not r['valid']];weighted_invalid=[r['row_id']for r in records if not r['valid']and r['summary_weight']>0]
    status='INVALID_POSITIVE_WEIGHT_SAMPLES'if weighted_invalid else'PASS'
    info=dict(activity='deterministic forward propagation',status=status,inference_run=False,samples=len(points),original_weight_sum=total,
        weights='Explicit supplied weights normalized for summaries',conditioning=conditioning,invalid_original_rows=invalid,invalid_positive_weight_rows=weighted_invalid,
        clipping=False,automatic_exclusions=False,output=str(output),end_time=end,dt=step,
        note='Raw trajectories and all rows are retained. Summaries/plots are withheld if any positive-weight sample is physically invalid.')
    save(output/'run.json',info);save(output/'model_configuration.json',cfg)
    np.savez_compressed(output/'training_NLL.npz',row_id=ids,NLL=nll)
    fig=None
    if not weighted_invalid:
        summary=summaries(data);np.savez_compressed(output/'summaries.npz',**summary)
        plot_cfg=dict(cfg)
        condition='supplied physical-validity condition'if conditioning=='supplied'else'input weights; no historical mask'
        plot_cfg['label']=f"{cfg.get('dataset','SoS')} — forward recomputation; {int(np.sum(weights>0))}/{len(weights)} contributing samples; {condition}"
        truth=None
        if cfg['files'].get('generating_truth'):
            with np.load(path.parent/cfg['files']['generating_truth'],allow_pickle=False)as z:truth={k:z[k]for k in ['time','E','I','beta']}
        fig=plot_data(data,obs,plot_cfg,summary,truth=truth);fig.savefig(output/'fit_forecast.pdf',bbox_inches='tight')
    return info,fig,data


def calculate(config_path, output, sample_path=None, conditioning='auto'):
    """Calculate every row. Auto conditioning is only for the supplied Ottawa set.

    Explicit alternative sample files never inherit the historical mask.
    File hashes protect that specific scientific conditioning, not general use.
    """
    _, cfg = configuration(config_path)
    if conditioning == 'auto':
        conditioning = 'supplied' if sample_path is None and cfg.get('historical_conditioning') else 'none'
    return calculate_from_samples(config_path, sample_path, output, conditioning)
