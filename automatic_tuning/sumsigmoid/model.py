"""Count-based compartment equations, normalized likelihood and Euler solver.

    S'=-beta(E+I)S/N; E'=beta(E+I)S/N-(gamma_i+gamma_e)E
    I'=gamma_i E-(gamma_r+gamma_d)I; R'=gamma_e E+gamma_r I; D'=gamma_d I

N=S+E+I+R+D includes deaths, per the approved final inference convention.
All five compartments sum to the initial total population (up to roundoff).
Rates and beta
have units day^-1; states/data are persons. No posterior/prior term is in NLL.
The arithmetic order follows the validated scalar likelihood exactly.
"""
import math
import numpy as np

BETA_TOLERANCE=1e-12
DENOMINATOR='total S+E+I+R+D'

class Model:
    def __init__(self,observations,locations,config):
        self.y=np.asarray(observations,dtype=float).copy();self.locations=list(locations);self.config=config
        self.settings=config['model'];self.grid=np.arange(0,len(self.y),self.settings['dt'])
        self.denom=1+np.exp((np.array(self.locations)[:,None]-self.grid)/self.settings['sigmoid_width'])
        self.names=['A0']+[f'A{k+1}' for k in range(len(locations))]+['E0','I0']

    def beta(self,p,grid=None):
        """Transmission on the training grid or an explicit propagation grid."""
        denom=self.denom if grid is None else 1+np.exp((np.array(self.locations)[:,None]-grid)/self.settings['sigmoid_width'])
        b=np.full(denom.shape[1],float(p[0]))
        for j in range(len(self.locations)):b=b+float(p[j+1])/denom[j]
        return b

    def evaluate(self,param,keep_states=False):
        """Return log likelihood and optional training states; no forecasting.

        SD at day d is sigma*I_d; y0 is included. Positive observations do not
        set E0/I0. Both are always the last two inferred parameters.
        """
        return self._integrate(param,self.grid,self.beta(param),keep_states)

    def forward(self,param,end_time=199.9):
        """Forward-only fit/forecast from the inferred initial state, in days.

        No future observations are accepted. The same Euler kernel continues
        through the cutoff without resets; the returned likelihood still uses
        only self.y, including day zero. Forecast-domain validity is a separate
        diagnostic, not a filter that removes or repairs posterior particles.
        end_time must lie on the configured grid and include the training grid.
        """
        dt=float(self.settings['dt']);steps=round(float(end_time)/dt)
        if not math.isfinite(float(end_time)) or abs(steps*dt-end_time)>1e-10 or end_time<self.grid[-1]:
            raise ValueError('Forward endpoint must include training and lie on the Euler grid')
        grid=np.arange(steps+1)*dt;beta=self.beta(param,grid)
        ll,states=self._integrate(param,grid,beta,True)
        return dict(time=grid,states=states,beta=beta,training_loglikelihood=ll,
                    training_observation_count=len(self.y),denominator=DENOMINATOR)

    def _integrate(self,param,grid,beta,keep_states):
        """Single scalar Euler implementation shared by inference and export."""
        p=[float(x) for x in param];Y=self.y;T=grid;m=self.settings
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
                s,e,i,r,d=sn,en,inn,rn,dn;n=s+e+i+r+d
                if keep_states:states[k]=[s,e,i,r,d]
                if k%every==0 and k//every<len(Y):
                    if i<=0:return -np.inf,states
                    ll+=math.log(1/(math.sqrt(2*math.pi)*sigma*i))-(Y[k//every]-i)**2/(2*(sigma*i)**2)
            return (float(ll) if math.isfinite(ll) else -np.inf),states
        except (OverflowError,ZeroDivisionError,ValueError):return -np.inf,states

    def logtarget(self,p):
        """Hard training-grid beta support, then the original likelihood."""
        if np.min(self.beta(p)) < -BETA_TOLERANCE:return -np.inf
        return self.evaluate(p)[0]
