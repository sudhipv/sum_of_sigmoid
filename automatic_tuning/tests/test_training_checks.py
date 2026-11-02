"""Exercise external diagnostics on fixed vectors only;no sampler or fits."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
import numpy as np
from prepared_fixture import make
from sumsigmoid.training_checks import check_saved_samples

class TrainingCheckTests(unittest.TestCase):
    def test_direct_offset_and_training_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            _,model,priors,ml=make(folder)
            p=np.array([ml['parameters'],ml['parameters']])
            C=float(np.sum(-np.log(np.array(priors['upper'])-priors['lower'])))
            values=np.array([-ml['NLL']-C]*2)
            result=check_saved_samples(Path(folder),p,values,priors,ml,SimpleNamespace(MODEL=model,logtarget=model.logtarget))
            self.assertLess(result['max_direct_likelihood_error'],1e-12)
            self.assertTrue(result['training_review_required'])
            self.assertAlmostEqual(result['sampled_mean_training_RMSE'],result['ML_training_RMSE'])

    def test_inconsistent_cache_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            _,model,priors,ml=make(folder)
            with self.assertRaisesRegex(RuntimeError,'Direct likelihood check failed'):
                check_saved_samples(Path(folder),np.array([ml['parameters']]),np.array([0.]),priors,ml,SimpleNamespace(MODEL=model,logtarget=model.logtarget))
