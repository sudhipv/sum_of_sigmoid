# Verification — approved total-population update, 2026-11-01

The final publication inference model now uses **S+E+I+R+D**. This is an explicit
author-approved scientific change, not a sampler modification. All other
scientific settings and the predeclared candidate procedure remain unchanged.

- **17 deterministic unittest methods passed**, including the seven original
  acceptance scenarios, exact original-source/import checks, total-denominator
  arithmetic, conservation, stale-preparation rejection and external-likelihood
  checks. Sampler calls in interface tests are intercepted before sampling.
- The primary detector still returns case one11, case two10 and Toronto16.
  Its source, both consolidation modules, optimizer, acceptance policy and
  numeric configuration defaults are byte-identical to the preceding package.
- **Verified total-denominator reference:** Toronto's saved full-precision
  19-parameter vector reproduces NLL846.1525253786608 and
  training RMSE69.7328983722431. Bundled daily E/I/D/beta reference values and
  the last training-grid point agree exactly in the tested environment. Local
  full-grid checks against saved E/I/D/beta arrays also have zero error.
- ML objective, saved-trajectory likelihood reconstruction, the actual TMCMC
  callback and that callback in a fresh interpreter agree exactly at this
  vector. The optional entrypoint dry-run verifies original imports, dates,
  parameter order and signed prior supports without drawing particles.
- Maximum training-grid population error is4.6566128730773926e-9 persons at
  N0=2794356, below the existing1e-6person reference tolerance. A deterministic
  positive-death test distinguishes total from living denominators after the
  first step, rather than testing conservation alone.
- **Exactly one new preparation/ML smoke test:** Toronto,160training counts,
  all16fixed candidates, free beta0/16amplitudes/E0/I0. All4starts reported
  optimizer success;3distinct starts agree at the best NLL with matching signs.
  The fourth poor local solution (NLL5728.50643657964) remains recorded outside
  the competitive group and does not veto acceptance. No manual start editing.
  Best parameters/NLL exactly match the saved total-denominator reference;
  NLL846.1525253786608, RMSE69.7328983722431,14.0984s preparation including
  detection/output. The accepted fit hits A5's -0.5 bound (day35), and the beta
  constraint is active within the existing1e-12tolerance. These are preserved
  reference properties, not new thresholds or evidence of nonrestrictive priors.
- **Historical checks are separate:** the isolated snapshot in
  `fixtures/historical_living_model.py` reproduces the old living-model NLLs
  (case one899.0524518687273; case two861.2059506985091;
  Toronto846.1528567195487). It is never imported by ML or TMCMC. These values
  are not asserted for the new total model. `verify_examples.py` reports the
  new NLLs at the same historical vectors separately, without refitting them.
- Both original sampler files still match commit
  `e412562ac57cff503f0f6cc4c2bd9d63da04785b` byte-for-byte. The pre-existing
  Toronto script and notebook edits were backed up and remain unchanged.

Run from `automatic_tuning`:

```bash
python -B -m unittest discover -s tests -v
python -B verify_examples.py
```

To independently repeat the **ML-only** smoke, choose a fresh output path:

```bash
python -B run.py prepare --config examples/toronto.json --output outputs/toronto_total_smoke
python -B run.py tmcmc --prepared outputs/toronto_total_smoke --dry-run
```

No new sampling, forecasts, timing sensitivity, other scientific experiments,
commits or pushes were performed for this update. No complete new optional
TMCMC run is claimed. Previously generated results keep their original labels.
Historical prepared outputs lack the new denominator metadata and are rejected
by optional inference; editing that label is not a valid scientific migration.

Portability remains limited to the tested Python3.11.5/pinned NumPy/SciPy/
Matplotlib environment. Native CPU count must match the requested sampler worker
count; no hidden override or initializer is introduced. Original worker RNG
states can duplicate and tempering completion does not establish adequate
exploration. Detection cutoffs remain1..159 and dt/noise/width remain frozen.
Deterministic numerical tolerances do not promise bitwise cross-platform ML
agreement. Plot tests emit nonfatal Matplotlib/pyparsing deprecation warnings.
Human scientific and code review is still pending.


## Current publication reconciliation

The scientific model, candidate rules, optimization/acceptance implementation,
mutation-control adapter and likelihood callback were compared byte-for-byte
with the source snapshot used by the accepted 27-configuration study. They
match. The original sampler and distributions independently match Git commit
`e412562ac57cff503f0f6cc4c2bd9d63da04785b`. Only explanatory input-error text,
documentation, explicit example settings and compact result-reproduction files
were updated. No inference algorithm or automatic selection rule was changed.

The primary case-two example now explicitly chooses `fixed50_experimental`;
case one and Toronto explicitly choose `original_adaptive`. Saved full-precision
ML vectors for all three total-denominator automatic baselines are bundled in
`fixtures/accepted_automatic_totals.json` for forward/likelihood checks.
The existing historical living checks remain separately labeled.

The three preparation/ML examples, optional sampling dry-runs, deterministic
tests and compact 27-result score reproduction are verified from an isolated
copy containing only publication files, without local experiment directories.
No new sampling or forecast propagation is required for this verification.
Reproducing compact metrics checks WIS, RMSE, coverage, interval width and bias
against the saved tables to absolute tolerance 1e-7. These tests assess numerical
consistency and packaging, not statistical convergence.
