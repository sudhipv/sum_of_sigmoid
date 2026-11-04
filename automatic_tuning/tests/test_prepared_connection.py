"""Deterministic integration checks: no optimization or sampling."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from prepared_fixture import make
from sumsigmoid import prepared
from sumsigmoid.io import save


class PreparationConnectionTests(unittest.TestCase):
    def test_snapshot_survives_later_preparation_edit(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/'preparation';folder.mkdir()
            config, model, priors, ml=make(folder)
            contract=prepared.write(folder)
            snap=Path(root)/'sample_inputs'
            prepared.snapshot(folder,snap)
            np.testing.assert_array_equal(contract['training_observations'],model.y)
            # An unrelated future edit is rejected in preparation, not applied
            # retroactively to the run-owned snapshot.
            priors['locations']=[0];save(folder/'TMCMC_priors.json',priors)
            with self.assertRaisesRegex(RuntimeError,'input changed'):
                prepared.load(folder)
            frozen=prepared.load(snap)
            self.assertEqual(frozen['locations'],[1])
            self.assertEqual(prepared.validate(frozen).logtarget(ml['parameters']),-ml['NLL'])

    def test_mixed_order_sign_dataset_and_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            make(root);value=prepared.write(root)
            changes=[('parameter_names',['A0','A1','I0','E0']),
                     ('dataset',dict(id='wrong'))]
            for key,changed in changes:
                bad=json.loads(json.dumps(value));bad[key]=changed
                bad['content_sha256']=prepared.configuration_digest(bad)
                with self.assertRaises(RuntimeError):prepared.validate(bad)
            bad=json.loads(json.dumps(value));bad['priors']['lower'][1]=0
            bad['content_sha256']=prepared.configuration_digest(bad)
            with self.assertRaisesRegex(RuntimeError,'supports/signs'):prepared.validate(bad)
            bad=json.loads(json.dumps(value));bad['training_observations'][1]+=1
            with self.assertRaisesRegex(RuntimeError,'hash'):prepared.validate(bad)

    def test_unresolved_preparation_is_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            make(root);save(Path(root)/'status.json',dict(status='unresolved'))
            with self.assertRaisesRegex(RuntimeError,'unresolved'):prepared.write(root)


if __name__=='__main__':unittest.main()
