"""User inputs and explicit paper defaults, kept out of scientific functions."""
from copy import deepcopy
from pathlib import Path
import math
from .io import read

MODEL_DEFAULTS=dict(dt=.1,sigma=.1,sigmoid_width=1.,gamma_e=1/15,gamma_i=1/5,gamma_r=1/11,gamma_d=1/750)
BOUND_DEFAULTS=dict(baseline_ml=[1e-6,.5],amplitudes_ml=[-.5,.5],initial_condition_upper_fraction=.001,
                    baseline_tmcmc=[0.,.5],amplitude_tmcmc_magnitude=.5)

def load(path):
    path=Path(path).resolve();c=read(path)
    allowed={'data','time_origin','training_cutoff','population','initial_guess','output_directory','model','bounds','tmcmc','procedure','seed'}
    if set(c)-allowed:raise ValueError('Unknown configuration fields:'+','.join(sorted(set(c)-allowed)))
    for key in ['data','time_origin','training_cutoff','population','initial_guess','output_directory']:
        if key not in c:raise ValueError(f'Required configuration input:{key}')
    for name,known in [('model',MODEL_DEFAULTS),('bounds',BOUND_DEFAULTS),('tmcmc',{'enabled':False,'particles':4000,'workers':8,'seed':106,'mutation_policy':'original_adaptive'})]:
        if set(c.get(name,{}))-set(known):raise ValueError('Unknown '+name+' settings')
    c['model']={**deepcopy(MODEL_DEFAULTS),**c.get('model',{})}
    c['bounds']={**deepcopy(BOUND_DEFAULTS),**c.get('bounds',{})}
    c['tmcmc']={**dict(enabled=False,particles=4000,workers=8,seed=106,mutation_policy='original_adaptive'),**c.get('tmcmc',{})}
    from .mutation_controls import describe
    describe(c['tmcmc']['mutation_policy'])
    c.setdefault('procedure','legacy');c.setdefault('seed',20261031)
    if type(c['seed']) is not int or not 0<=c['seed']<2**32:raise ValueError('ML seed must be a uint32 integer')
    if type(c['tmcmc']['seed']) is not int or not 0<=c['tmcmc']['seed']<2**32:raise ValueError('TMCMC seed must be a uint32 integer')
    if type(c['tmcmc']['enabled']) is not bool:raise ValueError('tmcmc.enabled must be a JSON boolean')
    for key in ['particles','workers']:
        if type(c['tmcmc'][key]) is not int:raise ValueError('tmcmc.'+key+' must be an integer')
    if c['procedure'] not in ['legacy','turn_optional']:raise ValueError('procedure must be legacy or turn_optional')
    if not math.isfinite(float(c['population'])) or float(c['population'])<=0:raise ValueError('Population must be positive, in persons')
    for key in ['E0','I0']:
        if key not in c['initial_guess']:raise ValueError(f'Explicit ML starting guess required:{key}')
    if c['initial_guess']['E0']<0 or c['initial_guess']['I0']<=0:raise ValueError('Starting E0>=0 and I0>0 required')
    m=c['model']
    if any(not math.isfinite(float(m[k])) or float(m[k])<=0 for k in m):raise ValueError('Model defaults/settings must be finite and positive')
    if abs(1/m['dt']-round(1/m['dt']))>1e-10:raise ValueError('dt must divide one day exactly')
    if m['dt']!=.1 or m['sigma']!=.1 or m['sigmoid_width']!=1.:
        raise ValueError('This release supports dt=.1,sigma=.1,width=1 only;scientific changes need review')
    for key in ['baseline_ml','amplitudes_ml','baseline_tmcmc']:
        if len(c['bounds'][key])!=2 or c['bounds'][key][0]>=c['bounds'][key][1]:raise ValueError('Invalid bounds:'+key)
    if c['bounds']!=BOUND_DEFAULTS:raise ValueError('This release preserves the documented bounds;review changes separately')
    if int(c['tmcmc']['particles'])<2 or int(c['tmcmc']['workers'])<1:raise ValueError('Invalid sampler counts')
    c['data']['file']=str((path.parent/c['data']['file']).resolve())
    c['output_directory']=str((path.parent/c['output_directory']).resolve())
    c['configuration_file']=str(path)
    return c
