"""Verify source, Uniform MH and entrypoint without a full sampler call."""
from pathlib import Path
from unittest.mock import patch
import json,os,subprocess,sys,tempfile,unittest
import numpy as np
from sumsigmoid.sampler_provenance import load_original
from prepared_fixture import make

ROOT=Path(__file__).resolve().parents[1]

class OriginalSamplerTests(unittest.TestCase):
    def test_original_imports_and_no_initializer(self):
        engine,pdfs,source=load_original()
        self.assertEqual(source['tmcmc_mod.py']['commit'],'e412562ac57cff503f0f6cc4c2bd9d63da04785b')
        self.assertEqual(Path(engine.__file__).resolve(),ROOT/'sumsigmoid/original/tmcmc_mod.py')
        self.assertEqual(Path(pdfs.__file__).resolve(),ROOT/'sumsigmoid/original/pdfs.py')
        self.assertNotIn('initializer=',Path(engine.__file__).read_text())
        self.assertEqual(engine.run_tmcmc.__defaults__,(5,10))

    def test_uniform_mh_acceptance_and_rejection_cache(self):
        engine,pdfs,_=load_original();pars=[pdfs.Uniform(-1,1)]
        x=np.array([.2]);callback=lambda p:-float(p[0]**2)/2
        stored=callback(x)-engine.log_prior(x,pars)
        with patch.object(engine,'propose',return_value=np.array([[2.]])):
            p,ll,n=engine.MCMC_MH(np.eye(1),1,x,stored,.5,0,pars,callback)
        self.assertTrue(np.array_equal(p,x));self.assertEqual(ll,stored);self.assertEqual(n,0)
        with patch.object(engine,'propose',return_value=np.array([[-.2]])),patch.object(engine.np.random,'uniform',return_value=.5):
            p,ll,n=engine.MCMC_MH(np.eye(1),1,x,stored,.5,0,pars,callback)
        self.assertEqual(n,1);self.assertEqual(p[0],0.)
        self.assertEqual(ll,callback(p)-engine.log_prior(p,pars))

    def test_entrypoint_arguments_intercepted_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            make(directory)
            code='''
import sys
from unittest.mock import patch
from sumsigmoid import optional_tmcmc
class Intercepted(Exception):pass
def intercept(engine,count,pars,callback,status):
    assert count==4000 and len(pars)==4
    assert engine.__name__=='sumsigmoid.original.tmcmc_mod'
    assert all(type(p).__module__=='sumsigmoid.original.pdfs' for p in pars)
    assert [(p.lower,p.upper) for p in pars]==[(0,.5),(-.5,0),(0,2794.356),(0,2794.356)]
    assert callback.__module__=='sumsigmoid.target'
    assert str(status).endswith('tmcmc/sampler_status.txt')
    raise Intercepted()
with patch.object(optional_tmcmc,'_sample_once',side_effect=intercept):
    try:optional_tmcmc.run(sys.argv[1])
    except Intercepted:print('INTERCEPTED_NO_SAMPLING')
    else:raise AssertionError('Sampler bridge was not reached')
'''
            result=subprocess.run([sys.executable,'-B','-c',code,directory],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('INTERCEPTED_NO_SAMPLING',result.stdout)

    def test_cli_spawn_import_preserves_shared_seed_without_workers(self):
        code='''
import os,runpy,numpy as np
os.environ['SUMSIGMOID_ORIGINAL_TMCMC_SEED']='106'
runpy.run_path('run.py',run_name='__mp_main__')
assert np.array_equal(np.random.random_sample(4),np.random.RandomState(106).random_sample(4))
print('SEED_IMPORT_CHECK_ONLY')
'''
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
