"""Check explicit controls, immutable originals and spawn imports; no sampling."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
from sumsigmoid.mutation_controls import describe
from sumsigmoid.optional_tmcmc import _sample_once
ROOT=Path(__file__).resolve().parents[1]

class MutationControlTests(unittest.TestCase):
    def test_default_and_invalid_policy(self):
        default=describe('original_adaptive')
        self.assertEqual(default['initial_attempts'],5);self.assertEqual(default['maximum_attempts'],10)
        self.assertTrue(default['adaptive_move_count']);self.assertFalse(default['experimental'])
        with self.assertRaises(ValueError):describe('fixed_100')

    def test_existing_arguments_only_for_explicit_fixed50(self):
        original=Mock();original.__name__='sumsigmoid.original.tmcmc_mod'
        _sample_once(original,4000,[],None,'status')
        original.run_tmcmc.assert_called_once_with(4000,[],None,'multiprocessing','status')
        fixed=Mock();fixed.__name__='sos_fixed50_tmcmc'
        _sample_once(fixed,4000,[],None,'status')
        fixed.run_tmcmc.assert_called_once_with(4000,[],None,'multiprocessing','status',Nm_steps_max=50,Nm_steps_maxmax=50)

    def test_derivative_and_worker_restore_preserve_rng_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            code=r'''
import ast,hashlib,os,sys,numpy as np
from pathlib import Path
from sumsigmoid.mutation_controls import load_fixed50,restore_worker_module
from sumsigmoid.sampler_provenance import load_original
original,pdfs,_=load_original();raw=Path(original.__file__).read_bytes()
np.random.seed(107);before=np.random.get_state()
engine,evidence=load_fixed50(sys.argv[1])
assert evidence['sha256']=='5f38aad7fce446037f43f16db24c582b310c038fc7f28babdfbef12da08a2142'
assert Path(original.__file__).read_bytes()==raw
assert np.array_equal(before[1],np.random.get_state()[1]) and before[2:]==np.random.get_state()[2:]
os.environ['SUMSIGMOID_FIXED50_DIRECTORY']=sys.argv[1]
restore_worker_module()
assert np.array_equal(np.random.random_sample(4),np.random.RandomState(107).random_sample(4))
source=Path(engine.__file__).read_bytes()
assert source==raw.replace(b"Adap_calc_Nsteps = 'yes'",b"Adap_calc_Nsteps = 'no'")
assert b'initializer=' not in source
assert original.run_tmcmc.__defaults__==(5,10)
print('FIXED50_NO_SAMPLING_PASS')
'''
            result=subprocess.run([sys.executable,'-B','-c',code,directory],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('FIXED50_NO_SAMPLING_PASS',result.stdout)

    def test_fixed50_prepared_entrypoint_is_explicit(self):
        from prepared_fixture import make
        with tempfile.TemporaryDirectory() as directory:
            make(directory);path=Path(directory)/'configuration.json';cfg=json.loads(path.read_text())
            cfg['tmcmc']['mutation_policy']='fixed50_experimental';path.write_text(json.dumps(cfg))
            code='''
import sys
from sumsigmoid.optional_tmcmc import inspect_prepared
_,_,_,_,_,engine,_,plan=inspect_prepared(sys.argv[1])
assert engine.__name__=='sos_fixed50_tmcmc'
assert plan['mutation_controls']['initial_attempts']==50
assert not plan['mutation_controls']['adaptive_move_count']
assert plan['pool_initializer'] is None
assert plan['executed_sampler']['original_files_unchanged']
print('INSPECT_ONLY_PASS')
'''
            result=subprocess.run([sys.executable,'-B','-c',code,directory],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('INSPECT_ONLY_PASS',result.stdout)
