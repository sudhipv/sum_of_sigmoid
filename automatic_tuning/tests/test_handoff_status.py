"""Failed/missing cases are not silently presented as posterior exports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff_test_tools',ROOT/'handoff/handoff_tools.py')
tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)

class HandoffStatusTests(unittest.TestCase):
    def test_available_cases_requires_both_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for case,training,physical in [('case1',True,True),('case2',False,True),('toronto',True,False),('case2_seed107',True,True)]:
                folder=root/'cases'/case;folder.mkdir(parents=True)
                (folder/'posterior_samples.csv').write_text('not loaded by availability check')
                (folder/'validation_status.json').write_text(json.dumps(dict(training_checks_passed=training,forecast_domain_passed=physical)))
            self.assertEqual(tools.available_cases(root),['case1','case2_seed107'])
            self.assertEqual(tools.base_case('case2_seed107'),'case2')
            self.assertEqual(tools.base_case('toronto'),'toronto')

    def test_notebook_has_no_sampler_or_optimizer_call(self):
        nb=json.loads((ROOT/'handoff/forward_model.ipynb').read_text())
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertNotIn('run_tmcmc(',code);self.assertNotIn('minimize(',code)
        self.assertIn('available_cases(DATA_ROOT)',code)
        self.assertNotIn('c in CASES',code)
