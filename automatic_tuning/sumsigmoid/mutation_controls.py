"""Explicit mutation controls outside the verbatim original sampler files.

Default execution always uses the original adaptive 5–10-attempt controls.
The opt-in fixed50 experiment generates an isolated source copy changing only
the existing adaptation flag. It is not a new sampler or a convergence remedy.
"""
from pathlib import Path
import importlib
import os
import sys
import inspect
from contextlib import contextmanager
from .io import sha
from .sampler_provenance import load_original

POLICIES=('original_adaptive','fixed50_experimental')
MODULE='sos_fixed50_tmcmc'

def describe(policy):
    if policy not in POLICIES:raise ValueError('Unknown mutation policy: '+str(policy))
    fixed=policy=='fixed50_experimental'
    return dict(policy=policy,initial_attempts=50 if fixed else 5,
                adaptive_move_count=not fixed,maximum_attempts=50 if fixed else 10,
                attempts_include_rejected_proposals=True,
                original_worker_initialization=True,
                experimental=fixed)

def _expected_source():
    engine,_,_=load_original()
    raw=Path(engine.__file__).read_bytes()
    old=b"Adap_calc_Nsteps = 'yes'";new=b"Adap_calc_Nsteps = 'no'"
    if raw.count(old)!=1:raise RuntimeError('Original mutation-control source differs')
    return raw.replace(old,new)

def load_fixed50(directory,create=True):
    """Import only the verified one-flag derivative; never alter original files.

    Generated files live with run outputs, outside the publication source tree.
    Spawned workers import the same file through restore_worker_module().
    No random draws or reseeding occur in either function.
    """
    directory=Path(directory).resolve();path=directory/(MODULE+'.py');expected=_expected_source()
    if not path.exists():
        if not create:raise FileNotFoundError('Missing generated fixed50 source')
        directory.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(expected)
    if path.read_bytes()!=expected:raise RuntimeError('Generated fixed50 source mismatch; refusing overwrite')
    if MODULE in sys.modules and Path(sys.modules[MODULE].__file__).resolve()!=path:
        raise RuntimeError('Use a fresh process for a different prepared run')
    sys.path.insert(0,str(directory));engine=importlib.import_module(MODULE)
    if Path(engine.run_tmcmc.__code__.co_filename).resolve()!=path:raise RuntimeError('Unexpected sampler function source')
    return engine,dict(actual_import=str(path),sha256=sha(path),
        derived_from='e412562ac57cff503f0f6cc4c2bd9d63da04785b',
        source_difference="Only Adap_calc_Nsteps='yes' -> 'no'",
        original_files_unchanged=True)

def restore_worker_module():
    """Make the verified generated module importable under the original Pool."""
    directory=os.environ.get('SUMSIGMOID_FIXED50_DIRECTORY')
    if directory:load_fixed50(directory,create=False)

@contextmanager
def record_attempts(engine,expected=None):
    """Observe actual dispatched counts and completed returns, without RNG use.

    Original MCMC_MH executes range(Nm_steps), with no break/early return;
    completed tasks therefore attempted every supplied step, including rejects.
    These are proposal attempts, not likelihood evaluations or accepted moves.
    """
    lines,start=inspect.getsourcelines(engine.run_tmcmc)
    dispatch=next(i for i,line in enumerate(lines,start) if 'results = pool.starmap(MCMC_MH, iterables)' in line)
    accepted=next(i for i,line in enumerate(lines,start) if 'status_file.write("acceptance rate' in line)
    rows=[];pending={};previous=sys.gettrace()
    def local(frame,event,arg):
        if event!='line':return local
        v=frame.f_locals
        if frame.f_lineno==dispatch:
            counts=[a[1] for a in v['iterables']]
            if expected is not None and any(n!=expected for n in counts):raise RuntimeError('Dispatched move count differs')
            pending[v['stage_num']]=counts
        elif frame.f_lineno==accepted:
            counts=pending[v['stage_num']]
            if len(counts)!=len(v['numAcceptsS']):raise RuntimeError('Incomplete chain returns')
            rows.append(dict(stage=int(v['stage_num']),temperature=float(v['beta']),chains=len(counts),
                minimum_attempts_per_chain=int(min(counts)),maximum_attempts_per_chain=int(max(counts)),
                attempted_moves=int(sum(counts)),accepted_moves=int(v['numAccepts']),acceptance=float(v['R'])))
        return local
    def global_trace(frame,event,arg):return local if frame.f_code is engine.run_tmcmc.__code__ else None
    sys.settrace(global_trace)
    try:yield rows
    finally:sys.settrace(previous)
