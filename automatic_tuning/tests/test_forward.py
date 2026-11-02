"""Forecast horizon is independent of likelihood observations; no sampling."""
from pathlib import Path
import json,unittest
import numpy as np
from sumsigmoid.model import Model
from sumsigmoid.config import load
from sumsigmoid.io import load_training

ROOT=Path(__file__).resolve().parents[1]

class ForwardTests(unittest.TestCase):
    def reference(self):
        ref=json.loads((ROOT/'fixtures/total_denominator_reference.json').read_text())
        config=load(ROOT/ref['config']);y,_=load_training(config)
        return Model(y,ref['locations'],config),ref

    def test_extended_training_prefix_and_likelihood_are_identical(self):
        model,ref=self.reference();ll,states=model.evaluate(ref['parameters'],True)
        result=model.forward(ref['parameters'])
        np.testing.assert_array_equal(result['time'],np.arange(0,200,.1))
        np.testing.assert_array_equal(result['states'][:1600],states)
        np.testing.assert_array_equal(result['beta'][:1600],model.beta(ref['parameters']))
        self.assertEqual(result['training_loglikelihood'],ll)
        self.assertEqual(-ll,ref['NLL']);self.assertEqual(result['training_observation_count'],160)
        self.assertLess(np.max(abs(result['states'].sum(axis=1)-model.config['population'])),1e-6)

    def test_cutoff_continuity_uses_existing_state(self):
        model,ref=self.reference();a=model.forward(ref['parameters']);s,e,i,r,d=a['states'][1599]
        m=model.settings;force=a['beta'][1599]*(e+i)/(s+e+i+r+d)
        expected_i=i+.1*(m['gamma_i']*e-(m['gamma_r']+m['gamma_d'])*i)
        self.assertEqual(a['states'][1600,2],expected_i)

    def test_endpoint_checks(self):
        model,ref=self.reference()
        for endpoint in [159.,199.95]:
            with self.assertRaises(ValueError):model.forward(ref['parameters'],endpoint)

    def test_no_future_likelihood_terms(self):
        model,ref=self.reference()
        first=model.forward(ref['parameters'],160.)
        later=model.forward(ref['parameters'],199.9)
        self.assertEqual(first['training_loglikelihood'],later['training_loglikelihood'])
        np.testing.assert_array_equal(first['states'],later['states'][:1601])
