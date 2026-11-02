"""Total-N arithmetic, conservation, saved likelihood and stale-preparation guards."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sumsigmoid.config import MODEL_DEFAULTS
from sumsigmoid.model import Model
from sumsigmoid.optional_tmcmc import inspect_prepared
from sumsigmoid.io import save
from prepared_fixture import make


class TotalDenominatorTests(unittest.TestCase):
    def test_total_population_drives_force_after_deaths_accumulate(self):
        config = dict(model=MODEL_DEFAULTS, population=1000.)
        model = Model([100., 110.], [], config)
        point = [.2, 120., 100.]
        _, states = model.evaluate(point, keep_states=True)
        s, e, i, r, d = states[1]
        self.assertGreater(d, 0)
        total_step_s = s - .1 * (.2 * (e+i) / (s+e+i+r+d) * s)
        living_step_s = s - .1 * (.2 * (e+i) / (s+e+i+r) * s)
        self.assertEqual(states[2, 0], total_step_s)
        self.assertGreater(abs(states[2, 0]-living_step_s), 1e-8)
        np.testing.assert_allclose(states.sum(axis=1), 1000., rtol=0, atol=1e-10)

    def test_verified_total_reference_and_worker_equivalence(self):
        from verify_examples import check_total_reference
        result = check_total_reference()
        self.assertEqual(result['NLL'], result['saved_NLL'])
        self.assertLessEqual(result['maximum_population_error_persons'], 1e-6)

    def test_historical_preparation_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _, _ = make(directory)
            config.pop('transmission_denominator')
            save(Path(directory)/'configuration.json', config)
            with self.assertRaisesRegex(RuntimeError, 'denominator'):
                inspect_prepared(directory)
