"""One immutable prepared likelihood per spawned worker; no old particles loaded."""
import os
from pathlib import Path
from .model import Model
from .prepared import load

def bind(folder):
    """Bind one validated preparation, or its immutable sampling snapshot."""
    global FOLDER, CONTRACT, CONFIG, PRIORS, MODEL
    FOLDER=Path(folder).resolve()
    CONTRACT=load(FOLDER)
    CONFIG=CONTRACT['scientific_configuration']
    PRIORS=CONTRACT['priors']
    MODEL=Model(CONTRACT['training_observations'],CONTRACT['locations'],CONFIG)

bind(os.environ['SUMSIGMOID_PREPARED'])

def logtarget(parameters):return MODEL.logtarget(parameters)
