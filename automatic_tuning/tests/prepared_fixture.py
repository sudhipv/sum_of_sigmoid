"""A fixed two-observation model for entrypoint unit checks;no optimization."""
from pathlib import Path
import numpy as np
from sumsigmoid.config import MODEL_DEFAULTS, BOUND_DEFAULTS
from sumsigmoid.model import Model
from sumsigmoid.io import save

def make(folder):
    folder = Path(folder)
    config = dict(model=MODEL_DEFAULTS,bounds=BOUND_DEFAULTS,population=2794356,
                  tmcmc=dict(particles=4000,seed=106,workers=8,enabled=False))
    y = np.array([100.,110.]);point=[.1,-.04,120.,100.];model=Model(y,[1],config)
    priors=dict(locations=[1],parameter_names=model.names,lower=[0,-.5,0,0],upper=[.5,0,2794.356,2794.356])
    ml=dict(accepted=True,locations=[1],parameter_names=model.names,parameters=point,NLL=-model.logtarget(point))
    save(folder/'configuration.json',config);save(folder/'TMCMC_priors.json',priors)
    save(folder/'ML_result.json',ml);save(folder/'status.json',dict(status='accepted'))
    np.savetxt(folder/'training_observations.csv',y,delimiter=',',fmt='%.17g')
    return config,model,priors,ml
