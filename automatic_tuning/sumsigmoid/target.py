"""One immutable prepared likelihood per spawned worker; no old particles loaded."""
import os
from pathlib import Path
import numpy as np
from .io import read
from .model import Model

FOLDER=Path(os.environ['SUMSIGMOID_PREPARED'])
CONFIG=read(FOLDER/'configuration.json')
PRIORS=read(FOLDER/'TMCMC_priors.json')
MODEL=Model(np.loadtxt(FOLDER/'training_observations.csv',delimiter=','),PRIORS['locations'],CONFIG)

def logtarget(parameters):return MODEL.logtarget(parameters)
