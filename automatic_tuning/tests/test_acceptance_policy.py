"""Lightweight acceptance tests using synthetic records/fake callbacks;no ML."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
from sumsigmoid.acceptance import assess,run_batches
OUT=Path(__file__).resolve().parent
class FakeProblem:
    K=1
    initial_start=np.array([.1,0.,430.,430.])
    low=np.array([1e-6,-.5,0.,0.]);high=np.array([.5,.5,2794.356,2794.356])
    def beta(self,p):return np.array([p[0],p[0]+p[1]])
    def nll(self,p):return 0. if np.all(p>=self.low) and np.all(p<=self.high) and self.beta(p).min()>=-1e-12 else None
def call(index,nll,sign=1,success=True):
    return dict(start=index,initial_parameters=[.1+index*.001,0.,430.,430.],parameters=[.2,.04*sign,450.,420.],NLL=nll,success=success,reliable=success)
def main():
    problem=FakeProblem();checks=[]
    result=assess([call(0,100),call(1,100.00005),call(2,5000,-1),call(3,120)],problem)
    assert result['accepted'] and result['competitive_indices']==[0,1] and result['poorer_valid_indices']==[2,3]
    checks.append('reproduced best accepted despite much poorer conflicting sign')
    result=assess([call(0,100),call(1,100.00005),call(2,99)],problem)
    assert not result['accepted'] and result['best_NLL']==99 and result['competitive_indices']==[2]
    checks.append('unconfirmed new best unresolved;no fallback to older reproduced result')
    result=assess([call(0,100),call(1,100.00001,-1),call(2,100.00002)],problem)
    assert not result['accepted'] and not result['signs_agree']
    checks.append('all equally-good fits checked;two matching signs cannot outvote contradiction')
    result=assess([call(0,100,0),call(1,100.00001,0)],problem)
    assert not result['accepted'] and result['exact_zero_amplitude']
    checks.append('exact zero amplitude unresolved;no sign invented')
    duplicate=call(0,100);second=copy.deepcopy(duplicate);second['NLL']=100.00001
    assert not assess([duplicate,second],problem)['accepted']
    checks.append('duplicate starts do not provide independent start confirmation')
    with tempfile.TemporaryDirectory() as d:
        def losing(p,s,j):
            c=call(j,100-j,success=j not in [1,6,10]);c['initial_parameters']=s.tolist();return c
        result=run_batches(problem,losing,20261031,Path(d)/'exhaustion.json')
        assert len(result['calls'])==12 and not result['decision']['accepted'] and result['decision']['budget_exhausted'] and result['decision']['best_NLL']==89
        assert result['best_found_parameters'] is not None and result['completed_batches']==3
        checks.append('budget12includesfailed attempts;unconfirmed best saved and unresolved')
        scores=[100,100.00001,99,120,98,110,98.00005,111]
        def confirming(p,s,j):
            c=call(j,scores[j]);c['initial_parameters']=s.tolist();return c
        result=run_batches(problem,confirming,20261031,Path(d)/'batches.json')
        assert len(result['calls'])==8 and not result['batches'][0]['decision']['accepted'] and result['decision']['accepted']
        keys=[tuple(x['parameters']) for x in result['generation']];assert len(set(keys))==8
        rng=np.random.RandomState(20261031);p0=np.array([.1,0.,430.,430.]);sd=np.array([.03,.06,150.,150.]);expected=[p0]
        for _ in range(7):
            for attempt in range(10000):
                s=np.clip(p0+rng.normal(size=4)*sd,problem.low,problem.high)
                if tuple(s) not in [tuple(x) for x in expected] and problem.nll(s) is not None:expected.append(s);break
        assert np.array_equal(expected,[x['parameters'] for x in result['generation']])
        checks.append('complete batches,continuing reproducible stream,no duplicate starts,no premature acceptance')
    assert len(checks)==7
    print('PASS',len(checks),'acceptance checks;no scientific fitting')
if __name__=='__main__':main()

import unittest
class AcceptancePolicyTests(unittest.TestCase):
    def test_original_seven_checks(self):
        main()
    def test_failed_nonfinite_diagnostics_remain_recordable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempts.json'
            def recorded_failure(problem,start,index):
                result=call(index,100.+index*1e-5)
                result['initial_parameters']=start.tolist()
                if index in [1,3]:
                    result.update(success=False,reliable=False,NLL=float('nan'),minimum_beta=float('nan'))
                return result
            result=run_batches(FakeProblem(),recorded_failure,20261031,path)
            self.assertEqual(len(result['calls']),4)
            self.assertTrue(result['decision']['accepted'])
            saved=json.loads(path.read_text())
            self.assertIsNone(saved['calls'][1]['NLL'])
            self.assertEqual(saved['decision']['valid_indices'],[0,2])
