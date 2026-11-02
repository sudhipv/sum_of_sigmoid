"""Count-based compartment equations, normalized likelihood and Euler solver.

    S'=-beta(E+I)S/N; E'=beta(E+I)S/N-(gamma_i+gamma_e)E
    I'=gamma_i E-(gamma_r+gamma_d)I; R'=gamma_e E+gamma_r I; D'=gamma_d I

N=S+E+I+R excludes deaths, as in the preserved implementation. Rates and beta
have units day^-1; states/data are persons. No posterior/prior term is in NLL.
The arithmetic order follows the validated scalar likelihood exactly.
"""
import math
import numpy as np

BETA_TOLERANCE=1e-12

class Model:
    def __init__(self,observations,locations,config):
        self.y=np.asarray(observations,dtype=float).copy();self.locations=list(locations);self.config=config
        self.settings=config['model'];self.grid=np.arange(0,len(self.y),self.settings['dt'])
        self.denom=1+np.exp((np.array(self.locations)[:,None]-self.grid)/self.settings['sigmoid_width'])
        self.names=['A0']+[f'A{k+1}' for k in range(len(locations))]+['E0','I0']

    def beta(self,p):
        b=np.full(len(self.grid),float(p[0]))
        for j in range(len(self.locations)):b=b+float(p[j+1])/self.denom[j]
        return b

    def evaluate(self,param,keep_states=False):
        """Return log likelihood and optional training states; no forecasting.

        SD at day d is sigma*I_d; y0 is included. Positive observations do not
        set E0/I0. Both are always the last two inferred parameters.
        """
        p=[float(x) for x in param];beta=self.beta(p);Y=self.y;T=self.grid;m=self.settings
        TOTAL=float(self.config['population']);GE,GI,GR,GD=[float(m[k]) for k in ['gamma_e','gamma_i','gamma_r','gamma_d']]
        dt=float(m['dt']);sigma=float(m['sigma'])
        s,e,i,r,d,n=TOTAL-p[-2]-p[-1],p[-2],p[-1],0.,0.,TOTAL
        states=np.empty((len(T),5)) if keep_states else None
        if keep_states:states[0]=[s,e,i,r,d]
        try:
            if i<=0:return -np.inf,states
            ll=math.log(1/(math.sqrt(2*math.pi)*sigma*i))-(Y[0]-i)**2/(2*(sigma*i)**2)
            every=round(1/dt)
            for k in range(1,len(T)):
                force=float(beta[k-1])*(e+i)/n
                sn=s+dt*(-force*s);en=e+dt*(force*s-(GI+GE)*e)
                inn=i+dt*(GI*e-(GR+GD)*i);rn=r+dt*(GE*e+GR*i);dn=d+dt*(GD*i)
                s,e,i,r,d=sn,en,inn,rn,dn;n=s+e+i+r
                if keep_states:states[k]=[s,e,i,r,d]
                if k%every==0:
                    if i<=0:return -np.inf,states
                    ll+=math.log(1/(math.sqrt(2*math.pi)*sigma*i))-(Y[k//every]-i)**2/(2*(sigma*i)**2)
            return (float(ll) if math.isfinite(ll) else -np.inf),states
        except (OverflowError,ZeroDivisionError,ValueError):return -np.inf,states

    def logtarget(self,p):
        """Hard training-grid beta support, then the original likelihood."""
        if np.min(self.beta(p)) < -BETA_TOLERANCE:return -np.inf
        return self.evaluate(p)[0]
