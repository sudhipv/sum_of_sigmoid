"""Validated, portable preparation -> inference -> posterior handoff.

This is output orchestration, not a change to detection, ML, or TMCMC. The
training counts and complete scientific configuration travel together. A
sampling run gets its own copy before the original sampler is invoked.
"""
from copy import deepcopy
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import numpy as np
from .io import read, save, sha, day_number
from .model import Model, DENOMINATOR, BETA_TOLERANCE

INPUT_FILES = ('configuration.json', 'status.json', 'ML_result.json',
               'TMCMC_priors.json', 'training_observations.csv')


def configuration_digest(contract):
    """Canonical full artifact hash; no machine-specific path is required."""
    value = {k: v for k, v in contract.items() if k != 'content_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def from_prepared(folder):
    """Construct the handoff from existing accepted files; never rerun ML."""
    folder = Path(folder)
    if read(folder/'status.json')['status'] != 'accepted':
        raise RuntimeError('Preparation unresolved; inference is blocked')
    config, ml, priors, status = [read(folder / f) for f in
        ('configuration.json', 'ML_result.json', 'TMCMC_priors.json', 'status.json')]
    if status['status'] != 'accepted' or not ml.get('accepted'):
        raise RuntimeError('Preparation unresolved; inference is blocked')
    if config.get('transmission_denominator') != DENOMINATOR:
        raise RuntimeError('Prepared denominator is not the accepted total-population model')
    y = np.atleast_1d(np.loadtxt(folder / 'training_observations.csv', delimiter=','))
    scientific = deepcopy(config)
    scientific['data']['file'] = Path(config['data']['file']).name
    scientific['output_directory'] = '.'
    scientific.pop('configuration_file', None)
    origin = config['time_origin']
    calendar_origin = origin if isinstance(origin, str) else config.get('calendar_origin')
    coordinates = dict(model_days=list(range(len(y))), time_origin=origin,
        calendar_origin=calendar_origin, inference_days=list(range(len(y))),
        last_observation_day=len(y)-1, forecast_start_day=len(y),
        fit_scoring_days=list(range(1, len(y))))
    if calendar_origin:
        coordinates['calendar_dates'] = [str(dt.date.fromisoformat(calendar_origin) + dt.timedelta(days=i))
                                        for i in range(len(y))]
    dataset_id = config.get('dataset_id', Path(config['data']['file']).stem)
    portable_status = deepcopy(status)
    # The executable local command already has its own text file. It is not a
    # scientific setting or a dependency of this portable data contract.
    portable_status.pop('optional_command', None)
    result = dict(schema_version=1, dataset=dict(id=dataset_id,
        observation_quantity=config.get('observation_quantity', 'infectious counts'), units='persons',
        input_reference=Path(config['data']['file']).name,
        input_record=read(folder/'input_record.json') if (folder/'input_record.json').exists() else None),
        time_coordinates=coordinates, training_observations=y.tolist(),
        scientific_configuration=scientific, locations=ml['locations'],
        parameter_names=ml['parameter_names'],
        parameter_labels=['beta0']+[f'amplitude_day_{d}' for d in ml['locations']]+['E0','I0'],
        parameter_units=['day^-1']*(len(ml['locations'])+1)+['persons','persons'],
        ML=ml, priors=priors, preparation_status=portable_status,
        procedure=dict(name=config.get('procedure','legacy'), locations_fixed=True,
                       locations_and_signs_selected_using_training_data=True,
                       selection_uncertainty_excluded=True),
        initial_conditions=dict(inferred=['E0','I0'], S0='population-E0-I0', R0=0, D0=0),
        forward_model_sha256=sha(Path(__file__).with_name('model.py')),
        beta_admissibility=dict(training_grid=True, tolerance=BETA_TOLERANCE),
        input_files_sha256={name:sha(folder/name) for name in INPUT_FILES})
    result['content_sha256'] = configuration_digest(result)
    validate(result)
    return result


def validate(value):
    """Reject mixed data, parameter order, supports, signs or scientific setup."""
    if value.get('schema_version') != 1 or configuration_digest(value) != value.get('content_sha256'):
        raise RuntimeError('Prepared artifact schema/hash mismatch')
    c, ml, prior = value['scientific_configuration'], value['ML'], value['priors']
    y = np.asarray(value['training_observations'], dtype=float)
    locations = value['locations']
    if not len(y) or not np.isfinite(y).all() or np.any(y < 0):
        raise RuntimeError('Invalid training observations')
    if c.get('transmission_denominator') != DENOMINATOR:
        raise RuntimeError('Denominator mismatch')
    if 'forward_model_sha256' in value and value['forward_model_sha256'] != sha(Path(__file__).with_name('model.py')):
        raise RuntimeError('Saved forward-model source differs from the imported module')
    if c.get('dataset_id', value['dataset']['id']) != value['dataset']['id']:
        raise RuntimeError('Dataset identity mismatch')
    if day_number(c['training_cutoff'], c['time_origin']) != len(y)-1:
        raise RuntimeError('Training cutoff mismatch')
    if value['time_coordinates']['model_days'] != list(range(len(y))):
        raise RuntimeError('Time coordinates mismatch')
    if locations != sorted(set(locations)) or any(not 0<=d<len(y) for d in locations):
        raise RuntimeError('Invalid fixed locations')
    model = Model(y, locations, c)
    for names in (value['parameter_names'], ml['parameter_names'], prior['parameter_names']):
        if names != model.names: raise RuntimeError('Parameter ordering mismatch')
    if ml['locations'] != locations or prior['locations'] != locations:
        raise RuntimeError('Location mismatch')
    if not ml.get('accepted') or value['preparation_status']['status'] != 'accepted':
        raise RuntimeError('ML preparation unresolved')
    point = np.asarray(ml['parameters'], dtype=float)
    if point.shape != (len(locations)+3,) or not np.isfinite(point).all():
        raise RuntimeError('Invalid fitted parameter vector')
    signs = np.sign(point[1:-2]).astype(int)
    if not np.all(signs): raise RuntimeError('Zero amplitude has no approved sign')
    lower = np.r_[0., np.where(signs < 0, -.5, 0.), 0., 0.]
    upper = np.r_[.5, np.where(signs < 0, 0., .5), [.001*c['population']]*2]
    if not np.array_equal(prior['lower'], lower) or not np.array_equal(prior['upper'], upper):
        raise RuntimeError('Prior supports/signs mismatch')
    if 'signs' in prior and not np.array_equal(prior['signs'], signs):
        raise RuntimeError('Recorded amplitude signs mismatch')
    if np.any(point < lower) or np.any(point > upper):
        raise RuntimeError('ML vector lies outside proposed signed support')
    if value['initial_conditions'] != dict(inferred=['E0','I0'],S0='population-E0-I0',R0=0,D0=0):
        raise RuntimeError('Initial-condition convention mismatch')
    nll = -model.logtarget(point)
    if not np.isfinite(nll) or abs(nll-ml['NLL'])>1e-7:
        raise RuntimeError('Prepared likelihood does not reproduce saved ML NLL')
    return model


def load(folder):
    """Old accepted preparations can be inspected without modifying them."""
    folder = Path(folder)
    if not (folder/'prepared_run.json').exists():
        return from_prepared(folder)
    value = read(folder/'prepared_run.json')
    validate(value)
    for filename, expected in value['input_files_sha256'].items():
        if sha(folder/filename) != expected:
            raise RuntimeError('Prepared input changed after handoff: '+filename)
    return value


def write(folder):
    value = from_prepared(folder)
    save(Path(folder)/'prepared_run.json', value)
    return value


def snapshot(folder, destination):
    """Copy small inputs before sampling; workers read this run-owned copy."""
    value = load(folder)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in INPUT_FILES:
        shutil.copy2(Path(folder)/name, destination/name)
    save(destination/'prepared_run.json', value)
    load(destination)
    return value
