"""Public case policies and self-contained compact score reproduction."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
from sumsigmoid.config import load
ROOT=Path(__file__).resolve().parents[1]

class PublicResultsTests(unittest.TestCase):
    def test_three_examples_declare_accepted_mutation_policies(self):
        for case in ['case1','case2','toronto']:
            config=load(ROOT/'examples'/(case+'.json'))
            expected='fixed50_experimental'if case=='case2'else'original_adaptive'
            self.assertEqual(config['tmcmc']['mutation_policy'],expected)
            self.assertEqual(config['tmcmc']['particles'],4000)
            self.assertEqual(config['tmcmc']['seed'],106)
            self.assertEqual(config['tmcmc']['workers'],8)
            self.assertFalse(config['tmcmc']['enabled'])

    def test_score_reproduction_from_arbitrary_working_directory(self):
        with tempfile.TemporaryDirectory()as cwd:
            p=subprocess.run([sys.executable,'-B',str(ROOT/'results/reproduce.py'),'--check-only'],
                cwd=cwd,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        report=json.loads(p.stdout)
        self.assertEqual(report['configurations'],27)
        self.assertLess(report['maximum_metric_error'],1e-7)
        self.assertEqual(report['new_sampling'],0)
        self.assertEqual(report['new_forecast_propagation'],0)
        self.assertEqual(report['WIS_independent_check']['status'],'PASS')
