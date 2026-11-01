"""Reproduce frozen candidate lists and saved NLLs;no fitting or sampling."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
from sumsigmoid.config import load
from sumsigmoid.io import load_training
from sumsigmoid.candidates import prepare
from sumsigmoid.model import Model
from sumsigmoid.sampler_provenance import load_original

ROOT = Path(__file__).resolve().parent

def main():
    reports = []
    for reference in json.loads((ROOT / 'fixtures/saved_reference_cases.json').read_text()):
        config = load(ROOT / reference['config'])
        y, cutoff = load_training(config)
        candidates = prepare(y, cutoff, config['procedure'])
        assert candidates['locations'] == reference['locations']
        assert len(candidates['locations']) == reference['count']
        model = Model(y, candidates['locations'], config)
        actual = -model.logtarget(reference['parameters'])
        assert abs(actual-reference['NLL']) < 1e-7
        reports.append(dict(case=reference['case'],candidate_count=len(candidates['locations']),
                            locations=candidates['locations'],NLL=actual,
                            absolute_NLL_error=abs(actual-reference['NLL'])))
    _, _, original = load_original()
    print(json.dumps(dict(status='PASS',cases=reports,original_source=original,
                          new_fits=0,new_sampling_runs=0),indent=2))

if __name__ == '__main__':
    main()
