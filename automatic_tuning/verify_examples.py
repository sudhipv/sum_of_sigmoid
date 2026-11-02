"""Frozen candidates, historical living checks and verified total-N reference.

Only saved-parameter forward evaluations are performed. No optimizer, sampler
or forecast is called. A fresh interpreter verifies the actual worker callback.
"""
from pathlib import Path
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import numpy as np
sys.dont_write_bytecode=True
from sumsigmoid.config import load
from sumsigmoid.io import load_training, save
from sumsigmoid.candidates import prepare
from sumsigmoid.model import Model, DENOMINATOR
from sumsigmoid.optimization import Problem
from sumsigmoid.sampler_provenance import load_original

ROOT = Path(__file__).resolve().parent

def historical_model():
    """Load an isolated old snapshot for historical checks only, never inference."""
    spec = importlib.util.spec_from_file_location(
        'historical_living_model', ROOT / 'fixtures/historical_living_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def check_historical():
    reports = []
    for ref in json.loads((ROOT / 'fixtures/saved_reference_cases.json').read_text()):
        config = load(ROOT / ref['config'])
        y, cutoff = load_training(config)
        candidates = prepare(y, cutoff, config['procedure'])
        assert candidates['locations'] == ref['locations']
        assert len(candidates['locations']) == ref['count']
        old_nll = -historical_model()(y, ref['locations'], config).logtarget(ref['parameters'])
        assert abs(old_nll-ref['NLL']) < 1e-7
        # Deliberately do not expect the new scientific model to match old NLLs.
        new_nll = -Model(y, ref['locations'], config).logtarget(ref['parameters'])
        reports.append(dict(case=ref['case'], candidate_count=ref['count'],
            locations=candidates['locations'], historical_denominator=ref['denominator'],
            historical_NLL=old_nll, historical_NLL_error=abs(old_nll-ref['NLL']),
            total_N_NLL_at_same_historical_parameters=new_nll,
            NLL_change_at_same_parameters=new_nll-old_nll))
    return reports


def check_total_reference():
    ref = json.loads((ROOT / 'fixtures/total_denominator_reference.json').read_text())
    config = load(ROOT / ref['config'])
    config['transmission_denominator'] = DENOMINATOR
    y, cutoff = load_training(config)
    np.testing.assert_array_equal(y, ref['training_observations'])
    assert prepare(y, cutoff, config['procedure'])['locations'] == ref['locations']
    model = Model(y, ref['locations'], config)
    p = np.asarray(ref['parameters'])
    assert model.names == ref['parameter_names']
    ll, states = model.evaluate(p, keep_states=True)
    nll = -ll
    assert abs(nll-ref['NLL']) < 1e-7
    step = round(1/config['model']['dt'])
    rmse = float(np.sqrt(np.mean((states[::step, 2]-y)**2)))
    assert abs(rmse-ref['RMSE']) < 1e-7
    errors = {}
    for key, values in [('I', states[:, 2]), ('E', states[:, 1]),
                        ('D', states[:, 4]), ('beta', model.beta(p))]:
        expected = np.asarray(ref['daily_reference'][key])
        np.testing.assert_allclose(values[::step], expected, rtol=0, atol=1e-7)
        assert abs(values[-1]-ref['last_training_grid_reference'][key]) < 1e-7
        errors[key] = float(np.max(np.abs(values[::step]-expected)))
    mass_error = float(np.max(np.abs(states.sum(axis=1)-config['population'])))
    assert mass_error <= ref['conservation_tolerance_persons']
    trajectory_nll = -sum(math.log(1/(math.sqrt(2*math.pi)*.1*float(i)))
                          -(float(obs)-float(i))**2/(2*(.1*float(i))**2)
                          for i, obs in zip(states[::step, 2], y))
    ml_nll = Problem(y, ref['locations'], config).nll(p)
    assert ml_nll == nll == trajectory_nll

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        signs = np.sign(p[1:-2])
        lower = [0.] + np.where(signs < 0, -.5, 0.).tolist() + [0., 0.]
        upper = [.5] + np.where(signs < 0, 0., .5).tolist() + [.001*config['population']]*2
        save(folder/'configuration.json', config)
        save(folder/'status.json', dict(status='accepted'))
        save(folder/'ML_result.json', dict(accepted=True, parameter_names=model.names,
            locations=ref['locations'], parameters=p.tolist(), NLL=nll))
        save(folder/'TMCMC_priors.json', dict(parameter_names=model.names,
            locations=ref['locations'], lower=lower, upper=upper))
        np.savetxt(folder/'training_observations.csv', y, delimiter=',', fmt='%.17g')
        dry = subprocess.run([sys.executable, '-B', str(ROOT/'run.py'), 'tmcmc',
            '--prepared', str(folder), '--dry-run'], cwd=ROOT, capture_output=True, text=True, check=True)
        plan = json.loads(dry.stdout)
        assert plan['denominator'] == DENOMINATOR
        assert plan['verified_ML_NLL'] == nll
        # Same target imported by sampler workers; no particles or sampler pool.
        code = '''
import os,sys,json
os.environ['SUMSIGMOID_PREPARED']=sys.argv[1]
from sumsigmoid import target
p=json.load(open(sys.argv[1]+'/ML_result.json'))['parameters']
print(json.dumps(dict(NLL=-target.logtarget(p),parameter_names=target.MODEL.names,
 locations=target.MODEL.locations,denominator=target.CONFIG['transmission_denominator'])))
'''
        child = subprocess.run([sys.executable, '-B', '-c', code, str(folder)],
            cwd=ROOT, capture_output=True, text=True, check=True)
        worker = json.loads(child.stdout)
        assert worker['NLL'] == nll and worker['locations'] == ref['locations']
        assert worker['parameter_names'] == model.names and worker['denominator'] == DENOMINATOR
        assert not (folder/'tmcmc').exists()
    return dict(case=ref['case'], denominator=DENOMINATOR, NLL=nll, saved_NLL=ref['NLL'],
        absolute_NLL_error=abs(nll-ref['NLL']), RMSE=rmse,
        daily_reference_maximum_errors=errors, maximum_population_error_persons=mass_error,
        conservation_tolerance_persons=ref['conservation_tolerance_persons'],
        ML_NLL=ml_nll, trajectory_NLL=trajectory_nll, TMCMC_callback_NLL=plan['verified_ML_NLL'],
        fresh_interpreter_worker_target=worker, observation_count=len(y),
        new_fits=0, new_sampling_runs=0, forecasts=0)


def main():
    historical = check_historical()
    total = check_total_reference()
    accepted=[]
    for ref in json.loads((ROOT/'fixtures/accepted_automatic_totals.json').read_text()):
        cfg=load(ROOT/ref['config']);y,cutoff=load_training(cfg)
        assert prepare(y,cutoff,cfg['procedure'])['locations']==ref['locations']
        model=Model(y,ref['locations'],cfg);p=np.asarray(ref['parameters'])
        assert model.names==ref['parameter_names']
        assert cfg['tmcmc']['mutation_policy']==ref['mutation_policy']
        ml=Problem(y,ref['locations'],cfg).nll(p);callback=-model.logtarget(p)
        assert ml==callback and abs(ml-ref['NLL'])<1e-7
        _,states=model.evaluate(p,True)
        rmse=float(np.sqrt(np.mean((states[::10,2]-y)**2)))
        assert abs(rmse-ref['training_RMSE'])<1e-7
        mass=float(np.max(abs(states.sum(axis=1)-cfg['population'])))
        assert mass<1e-6
        accepted.append(dict(case=ref['case'],NLL=ml,saved_NLL=ref['NLL'],RMSE=rmse,
            candidate_count=len(ref['locations']),mutation_policy=ref['mutation_policy'],
            ML_callback_equivalence=True,maximum_population_error_persons=mass))
    _, _, original = load_original()
    print(json.dumps(dict(status='PASS', publication_denominator=DENOMINATOR,
        historical_living_checks=historical, total_denominator_reference=total,
        accepted_automatic_references=accepted,
        original_source=original, new_fits=0, new_sampling_runs=0), indent=2))

if __name__ == '__main__':
    main()
