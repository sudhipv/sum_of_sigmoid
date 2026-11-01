"""Packaging guard checks; no optimizer, sampler or forecast calls."""
from pathlib import Path
import json,tempfile,unittest,subprocess,sys
import numpy as np
from sumsigmoid.io import load_training,day_number,save
from sumsigmoid.config import load
from sumsigmoid.optional_tmcmc import run as optional

class InputTests(unittest.TestCase):
    def test_future_count_not_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.csv';path.write_text('day,I\n0,100\n1,120\n2,not-a-number\n')
            config=dict(data=dict(file=str(path),header=True,date_column='day',count_column='I'),time_origin=0,training_cutoff=1)
            y,c=load_training(config);self.assertEqual(c,1);np.testing.assert_array_equal(y,[100,120])
    def test_calendar_origin(self):
        self.assertEqual(day_number('2020-09-07','2020-04-01'),159)
    def test_missing_days_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.csv';path.write_text('day,I\n0,100\n2,120\n')
            with self.assertRaises(ValueError):load_training(dict(data=dict(file=str(path),header=True,date_column='day',count_column='I'),time_origin=0,training_cutoff=2))
    def test_explicit_initial_guesses_required(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';save(path,dict(data={},time_origin=0,training_cutoff=159,population=1000,output_directory='new'))
            with self.assertRaisesRegex(ValueError,'initial_guess'):load(path)
    def test_cutoff_extension_is_not_silent(self):
        with self.assertRaisesRegex(ValueError,'1..159'):load_training(dict(data={},time_origin=0,training_cutoff=160))
    def test_optional_dry_run_and_unresolved_gate(self):
        from prepared_fixture import make
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);save(path/'status.json',dict(status='unresolved'))
            with self.assertRaises(RuntimeError):optional(path,dry_run=True)
            _,_,_,ml=make(path)
            cli=Path(__file__).resolve().parents[1]/'run.py'
            result=subprocess.run([sys.executable,'-B',str(cli),'tmcmc','--prepared',str(path),'--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            plan=json.loads(result.stdout)
            self.assertEqual(plan['particles'],4000)
            self.assertEqual(plan['verified_ML_NLL'],ml['NLL'])
            self.assertEqual(plan['denominator'],'living S+E+I+R')
            self.assertIsNone(plan['pool_initializer'])
            self.assertFalse((path/'tmcmc').exists())
