"""Joint two-sided ML preparation; optimizer/scaling preserved from frozen runs."""
import numpy as np
from .model import Model

TIE=1e-5
OPTIONS=dict(ftol=1e-9,eps=1e-8,maxiter=300)

class Problem:
    def __init__(self,observations,dates,config):
        self.model=Model(observations,dates,config);self.dates=list(dates);self.K=len(dates);self.names=self.model.names
        self.low=np.array([1e-6]+[-.5]*self.K+[0.,0.]);self.high=np.array([.5]*(self.K+1)+[.001*config['population']]*2)
        self.scale=np.array([.1]*(self.K+1)+[1000.,1000.]);self.denom=self.model.denom
        self.A=np.column_stack([np.ones(len(self.model.grid)),*(1/self.denom),np.zeros((len(self.model.grid),2))])
        self.requests=0;self.evaluations=0;self.invalid=0
        # User-supplied guesses are seeds, not fixed initial conditions.
        self.initial_start=np.array([.1]+[0.]*self.K+[config['initial_guess']['E0'],config['initial_guess']['I0']],dtype=float)

    def beta(self,p):return self.model.beta(p)

    def nll(self,p):
        self.requests+=1
        if np.any(p<self.low-1e-12) or np.any(p>self.high+1e-12) or self.beta(p).min() < -1e-12:
            self.invalid+=1;return None
        self.evaluations+=1
        with np.errstate(all='ignore'):ll=self.model.evaluate(p)[0]
        if not np.isfinite(ll):self.invalid+=1;return None
        return -float(ll)

# Preserve the original optimizer's fixed objective divisor160, including for
# shorter configured prefixes. Acceptance always uses the full unscaled NLL.

import time,warnings
from scipy.optimize import minimize

def one_start(problem,start,label,index):
    p0=np.array(start,dtype=float);begun=time.perf_counter();count=problem.evaluations;requests=problem.requests
    initial=problem.nll(p0);best=dict(NLL=initial,parameters=p0.tolist() if initial is not None else None)
    def objective(q):
        p=problem.scale*q;score=problem.nll(p)
        if score is None:return 1e10+float(np.sum(np.minimum(problem.beta(p),0)**2))*1e6
        if best['NLL'] is None or score<best['NLL']:best.update(NLL=score,parameters=p.tolist())
        return score/160.
    constraints=dict(type='ineq',fun=lambda q:problem.beta(problem.scale*q),jac=lambda q:problem.A*problem.scale[None,:])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result=minimize(objective,p0/problem.scale,method='SLSQP',bounds=list(zip(problem.low/problem.scale,problem.high/problem.scale)),constraints=constraints,options=OPTIONS)
    p=np.clip(result.x*problem.scale,problem.low,problem.high);score=problem.nll(p)
    reliable=bool(result.success and score is not None and (initial is None or score<=initial+TIE)
        and (best['NLL'] is None or score<=best['NLL']+1e-4))
    hits=[problem.names[j]+(' lower' if abs((p[j]-problem.low[j])/problem.scale[j])<=1e-6 else ' upper')
        for j in range(len(p)) if min(abs((p[j]-problem.low[j])/problem.scale[j]),abs((p[j]-problem.high[j])/problem.scale[j]))<=1e-6]
    return dict(label=label,start=index,dates=problem.dates,initial_parameters=p0.tolist(),initial_NLL=initial,
        parameters=p.tolist(),NLL=score,success=bool(result.success),status=int(result.status),message=str(result.message),
        reliable=reliable,iterations=int(result.nit),optimizer_function_evaluations=int(result.nfev),
        likelihood_evaluations=problem.evaluations-count,requests=problem.requests-requests,
        minimum_beta=float(problem.beta(p).min()),boundary_hits=hits,best_seen=best,
        warnings=[str(w.message) for w in captured],seconds=time.perf_counter()-begun)
