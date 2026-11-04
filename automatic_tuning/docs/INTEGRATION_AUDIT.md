# Integration audit and historical code comments

Human review pending. The locally extracted email thread was available at
`reproduction/2026-11-01_final_comparison_specification/email_compact.txt`.
These are supporting comments, not instructions to overwrite scientific settings.
No email was sent and no manuscript or historical notebook was edited.

| David's comment | Evidence and disposition |
|---|---|
| October25: `Npar=8` but the real-data script reads `param[8]`, loads synthetic case two with the Toronto line commented, and writes synthetic filenames | Verified in the preserved October22 snapshot of `inference/mcmc/singlephu_mult_mcmc_real.py`, including data line141, I0 line165 and output lines280–292. It is not a reliable Toronto reproduction entrypoint. The current local historical edits already address parts of this and were preserved byte-for-byte. The supported route now uses an explicit Toronto configuration, named parameter order, real-data identity/hash and observed active counts, checked in both parent and worker. Historical scripts are not silently substituted for the new route. |
| October25: manuscript amplitude magnitude0.5 versus magnitude0.2 in the three original scripts | Verified as a historical discrepancy, not a reason to relabel old runs. The current **approved** primary procedures use magnitude0.5 signed priors, recorded alongside each posterior. ML two-sided bounds and signed TMCMC priors remain separately identified. No claim is made that historical or current bounds are nonrestrictive. Manuscript reconciliation is still a later editorial task. |
| October25: Fig11 and Fig17 are identical although the captions describe different cases | Earlier local figure inventory records byte-identical files (SHA256 starts882daad6). New case-specific notebooks identify their sample/data/configuration hashes and reconstruct their own figures. The duplicated manuscript image itself remains unchanged and unresolved until manuscript work is authorized. |
| October16: old plots forward-simulated only500 of about2000 available particles; downstream scoring needs full predictive ensembles, not just parameter samples | Supported by the historical plotting source and saved email. Current notebooks use all4000 accepted particles with equal weights, including duplicate rows, and regenerate E/I/beta plus separate observation-predictive outputs. Single-row mode is explicit and does not report ensemble scores. |
| October16: what do noise10/noise20 filenames mean if sigma is hardcoded0.1? | Filenames alone do not establish a noise-sensitivity experiment. The supported run contract records sigma0.1 as relative Gaussian SD and hash-checks the shared model; it does not infer scientific settings from filenames. Historical suffix provenance is not fully reconstructed here. |

The full thread also describes proposed likelihood comparisons and model
complexity questions. This task does not adopt those proposals, infer locations
in TMCMC, alter the approved priors, or make changes to the spline model.
Brandon's exact metric function is still unavailable; the accepted eleven-level
WIS implementation is labelled as ours, not asserted identical to his code.

## Actual execution paths and duplicated models

| Path | Status |
|---|---|
| `run.py prepare` → workflow → candidates → optimization/acceptance → `Model.evaluate` | Supported automatic preparation; legacy primary remains default. |
| `run.py tmcmc` → optional_tmcmc → target → `Model.logtarget/evaluate` | Supported inference; validated run-owned snapshot, original sampler/workers; isolated case-two fixed50 controls. |
| `notebooks/*.ipynb` → `sumsigmoid.posterior` → `Model.forward` | Supported saved-posterior path; no optimizer/sampler imports or invocation. |
| `handoff/forward_model.ipynb` → `handoff_tools` → packaged `Model` | Earlier runnable all-case interface. The existing large package contains its own archived copy of the same verified model and remains untouched. |
| `inference/mcmc/singlephu_mult_mcmc_{synth_1,synth_2,real}.py` | Historical independent Euler kernels with hard-coded settings, indices and locations. Preserved, not current defaults. |
| `inference/mle/*.ipynb`, `inference/manual_tuning/*.ipynb` | Historical manual/ML kernels and demonstrations; not imported by the supported workflow. |
| `inference/plotting/plot_synth_case1.ipynb`, `plot_synth_case2.ipynb`, `plot_real_toronto.ipynb`, prediction notebooks | Historical trajectory/prediction implementations and sample-loading conventions. Preserved; not used by the new notebooks. |
| `review_package/` and experiment `src/` snapshots | Earlier package/saved execution copies. They remain available as provenance and are not imported by the current CLI or new notebooks. |

The new route does not deduplicate or delete historical source. It establishes
one shared forward kernel for new preparation, inference and posterior
reconstruction, with source-hash checks. The old kernel copies differ in
historical denominator/settings and must not be assumed interchangeable.

## What was and was not verified

Verified deterministically: accepted preparation/priors/parameter mapping,
saved ML and representative-particle likelihoods/curves through the real parent
and fresh-worker callbacks, immutable snapshot behavior, all three notebooks
in a clean dependency-only copy, compact accepted trajectories/summaries/scores,
and single-row versus ensemble indexing including predictive noise.

No new fitting or TMCMC was run. Thus this audit adds no sampling-convergence
evidence and does not repair inherited sampler limitations. Original sampler
and distribution files, scientific kernels, detection/consolidation rules,
historical local edits, accepted posterior samples and the earlier large
collaborator package remain preserved.
