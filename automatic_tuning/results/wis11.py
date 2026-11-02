"""Published central-interval WIS, implemented here; not Brandon-verified code.

Bracher et al., doi:10.1371/journal.pcbi.1008618, equations 1–2.
The 2022 correction concerns examples/figures, not these score definitions.
Quantiles are (quantile,time); observations are (time,). No time weighting.
"""
import numpy as np

COVERAGES=np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98])
ALPHA=1-COVERAGES
QUANTILES=np.sort(np.r_[ALPHA/2,.5,1-ALPHA/2])

def at(q,level):
    return q[np.argmin(abs(QUANTILES-level))]

def score(y,q):
    """Per-time WIS; interval weights alpha/2; median weight .5; divisor11.5."""
    y=np.asarray(y);q=np.asarray(q)
    if q.shape!=(23,len(y)) or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError('Expected finite 23-by-time quantiles and time observations')
    if np.min(np.diff(q,axis=0))< -1e-8:raise ValueError('Crossing quantiles')
    total=.5*abs(y-at(q,.5))
    for a in ALPHA:
        lo,hi=at(q,a/2),at(q,1-a/2)
        interval=(hi-lo)+2/a*np.maximum(lo-y,0)+2/a*np.maximum(y-hi,0)
        total+=a/2*interval
    return total/(len(ALPHA)+.5)

def metrics(y,q,mean,days):
    days=np.asarray(days);target=np.asarray(y)[days];selected=q[:,days]
    lo,hi=at(selected,.025),at(selected,.975);error=np.asarray(mean)[days]-target
    return dict(n_days=len(days),first_day=int(days[0]),last_day=int(days[-1]),
        WIS11=float(score(target,selected).mean()),RMSE=float(np.sqrt(np.mean(error**2))),
        signed_bias=float(error.mean()),coverage95=float(np.mean((lo<=target)&(target<=hi))),
        mean_width95=float(np.mean(hi-lo)))

def independent_checks():
    """Exact rational hand example, plus the equivalent quantile-loss identity."""
    from fractions import Fraction as F
    cover=[F(k,10)for k in range(1,10)]+[F(95,100),F(98,100)]
    # Uniform[-1,1]: m=0 and central c interval[-c,c]. For y=0,
    # weighted interval term=c(1-c); for y=2 it is 2-c*c, median term=1.
    inside=sum(c*(1-c)for c in cover)/F(23,2)
    outside=(1+sum(2-c*c for c in cover))/F(23,2)
    q=np.repeat((2*QUANTILES-1)[:,None],2,axis=1);y=np.array([0.,2.])
    got=score(y,q);np.testing.assert_allclose(got,[float(inside),float(outside)],rtol=0,atol=1e-14)
    # Algebraically separate pinball expression, equal weights over23 quantiles.
    errors=y[None,:]-q
    equivalent=2*np.mean(np.maximum(QUANTILES[:,None]*errors,(QUANTILES[:,None]-1)*errors),axis=0)
    np.testing.assert_allclose(got,equivalent,rtol=0,atol=1e-14)
    return dict(status='PASS',distribution='Uniform[-1,1]',observations=y.tolist(),
        hand_calculated_rational=[str(inside),str(outside)],expected=[float(inside),float(outside)],computed=got.tolist(),
        weights='median .5, intervals alpha/2',denominator=11.5,
        source='https://doi.org/10.1371/journal.pcbi.1008618',
        correction='https://doi.org/10.1371/journal.pcbi.1010592',
        independent_Brandon_implementation_verified=False)

if __name__=='__main__':
    import json
    print(json.dumps(independent_checks(),indent=2))
