"""Agreed local multistart acceptance; unchanged objective/optimizer/physics.

Full unscaled NLL, competitive window1e-4, >=2 distinct starts, all competitive
signs agree, exact zero unresolved. Fresh attempts run complete batches of4,
maximum12; the RNG continues between batches. Saved fits can be assessed without
calling the optimizer or generating further starts.
"""
import json
from pathlib import Path
import numpy as np
from .io import plain
TOLERANCE=1e-4
BATCH_SIZE=4
MAX_ATTEMPTS=12
def key(vector):return tuple(float(x) for x in vector)
def numerical_checks(call,problem):
    p=np.asarray(call.get('parameters',[]),float);nll=call.get('NLL');reasons=[]
    if not call.get('success',False):reasons.append('optimizer did not succeed')
    if not call.get('reliable',False):reasons.append('existing per-start numerical checks failed')
    if nll is None or not np.isfinite(nll):reasons.append('nonfinite full NLL')
    if p.shape!=problem.low.shape or not np.isfinite(p).all():reasons.append('invalid parameter vector')
    else:
        if np.any(p<problem.low-1e-12) or np.any(p>problem.high+1e-12):reasons.append('parameter bounds')
        if np.min(problem.beta(p)) < -1e-12:reasons.append('total beta infeasible')
    start=np.asarray(call.get('initial_parameters',[]),float)
    if start.shape!=problem.low.shape or not np.isfinite(start).all():reasons.append('invalid starting vector provenance')
    return dict(valid=not reasons,reasons=reasons)
def assess(calls,problem,max_attempts=MAX_ATTEMPTS):
    if len(calls)>max_attempts:raise ValueError('Attempt budget exceeded')
    checks=[numerical_checks(c,problem) for c in calls]
    valid=[j for j,c in enumerate(checks) if c['valid']]
    best_index=min(valid,key=lambda j:float(calls[j]['NLL'])) if valid else None
    best_nll=float(calls[best_index]['NLL']) if best_index is not None else None
    group=[j for j in valid if float(calls[j]['NLL'])<=best_nll+TOLERANCE]
    distinct={key(calls[j]['initial_parameters']) for j in group}
    signs=[np.sign(calls[j]['parameters'][1:-2]).astype(int).tolist() for j in group]
    zeros=any(0 in s for s in signs);agreement=bool(signs) and all(s==signs[0] for s in signs)
    accepted=bool(best_index is not None and len(distinct)>=2 and agreement and not zeros)
    reason='accepted'
    if best_index is None:reason='no numerically valid result'
    elif zeros:reason='exact zero amplitude in competitive group;no sign invented'
    elif not agreement:reason='conflicting amplitude signs in competitive group'
    elif len(distinct)<2:reason='best-found NLL is not reproduced by two distinct starts'
    return dict(accepted=accepted,status='accepted' if accepted else 'unresolved',reason=reason,
        attempts=len(calls),budget=max_attempts,budget_exhausted=len(calls)>=max_attempts,
        full_unscaled_NLL_tolerance=TOLERANCE,best_index=best_index,best_NLL=best_nll,
        best_parameters=calls[best_index]['parameters'] if best_index is not None else None,
        numerical_checks=checks,valid_indices=valid,competitive_indices=group,
        competitive_distinct_starts=len(distinct),competitive_amplitude_signs=signs,
        signs_agree=agreement,exact_zero_amplitude=zeros,
        poorer_valid_indices=[j for j in valid if j not in group],
        no_worse_reproduced_fallback=True,global_optimality_guarantee=False)
def run_batches(problem,fit_one,seed,record_path):
    """Run the agreed bounded-retry policy from configured feasible seeds.

    fit_one is the existing one_start(problem,start,label,index) adapter. A failed
    call consumes one attempt. No early assessment inside a batch. No scientific
    settings or initializer scales are changed by this acceptance implementation.
    """
    rng=np.random.RandomState(seed);p0=np.asarray(problem.initial_start,dtype=float).copy()
    sd=np.array([.03]+[.06]*problem.K+[150.,150.]);seen=set();calls=[];generation=[];batches=[]
    if problem.nll(p0) is None:raise RuntimeError('Existing constant starting seed infeasible;no new repair policy')
    def save_state(decision):
        data=dict(calls=calls,generation=generation,batches=batches,decision=decision,seed=seed,
            completed_batches=len(batches),best_found_parameters=decision['best_parameters'])
        # Nonfinite failed-fit diagnostics must not abort an otherwise complete
        # batch. Numeric validity/acceptance above still uses the original data.
        Path(record_path).write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
        return data
    for batch in range(MAX_ATTEMPTS//BATCH_SIZE):
        for slot in range(BATCH_SIZE):
            index=len(calls)
            if index==0:start=p0.copy();draws=0
            else:
                for draws in range(1,10001):
                    start=np.clip(p0+rng.normal(size=len(p0))*sd,problem.low,problem.high)
                    if key(start) not in seen and problem.nll(start) is not None:break
                else:raise RuntimeError('Existing10000draw feasibility budget exhausted;no optimizer or prior launched for missing start')
            assert key(start) not in seen;seen.add(key(start))
            generation.append(dict(attempt=index,batch=batch+1,slot=slot+1,draws=draws,parameters=start.tolist(),continuing_rng=True))
            try:call=fit_one(problem,start,index)
            except Exception as exc:
                call=dict(start=index,initial_parameters=start.tolist(),parameters=None,NLL=None,
                    success=False,reliable=False,status=-1,message=repr(exc))
            calls.append(call)
        decision=assess(calls,problem);batches.append(dict(batch=batch+1,attempts_after_batch=len(calls),decision=decision))
        saved=save_state(decision)
        if decision['accepted']:return saved
    return saved
