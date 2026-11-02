# Saved-result validation and collaborator handoff

The inference model uses **S+E+I+R+D**, with the predeclared primary automatic dates,
jointly inferred baseline/amplitudes/E0/I0, prescribed signed magnitude-0.5
Uniform priors, sigma=0.1, one-day sigmoid width and Euler dt=0.1 day.
Training observations are days 0–159. Neither held-out values nor forecasts
choose signs, locations, priors or a preferred repeat.

| Dataset | K / inferred parameters | Sampling policy | Reference |
|---|---:|---|---|
| Synthetic case one | 11 / 14 | Original adaptive 5–10, seed 106 | Completed final run reused |
| Synthetic case two | 10 / 13 | Experimental fixed 50, seeds 106/107/108 | Seed 106 predesignated; other seeds are validation repeats |
| Toronto | 16 / 19 | Original adaptive 5–10, seed 106 | Compatible completed original-sampler run verified and reused |

Every run uses 4,000 fresh-prior particles and the original eight-worker policy.
Original spawn-import seeds can duplicate initial worker RNG states; no distinct
worker initializer is applied. Original sampler and distribution source files
remain byte-for-byte pinned to `e412562ac57cff503f0f6cc4c2bd9d63da04785b`.
The case-two derivative changes only the existing move-count adaptation flag
to `no` and supplies `Nm_steps_max=50, Nm_steps_maxmax=50`. Attempts include
prior/beta rejections. This is explicit in `examples/case2.json`; the other examples retain adaptive
controls. It is not a new sampler or a guarantee of successful inference.

The Toronto reuse audit verified source hashes, default move settings, priors,
seed, initial particles, data/likelihood, shared-model trajectories and all final
likelihoods. Its earlier sign-selection provenance is retained; the resulting
signed supports exactly match the accepted total-denominator joint-ML preparation.
It is labelled as a verified reuse, not newly sampled or an unchanged paper run.

Lightweight outcomes and source hashes are recorded in
`../fixtures/final_validation_controls.json`. Full local results are in
`reproduction/2026-11-01_remaining_validation_handoff/`; research arrays and
archives are intentionally outside Git. Prior failed populations remain intact
in their original diagnostic folders and are not distributed as validated posteriors.

## Forward-only notebook

Use the separate collaborator ZIP and [notebook guide](../handoff/README.md).
The notebook retains every particle, including duplicates, and reconstructs
latent E/I/beta and observation-predictive I without optimization, sampling,
state resets or future-observation assimilation. Native time is 0–199.9 by 0.1;
daily time is 0–199. Proposed scoring masks are days 1–159 and 160–199, distinct
from the inference mask 0–159. Toronto day 0 is 2020-04-01.

Reference comparisons use atol=1e-7 and rtol=1e-12. Uncertainty bands are
pointwise equal-tail intervals. Analytic observation-predictive quantiles use
the Gaussian mixture over all particles; finite seeded predictive draws are
separate outputs. No clipping, particle filtering or future assimilation occurs.

## Interpretation and missing inputs

Training checks, physical forecast checks and numerical repeatability are
distinct from posterior convergence or scientific validation. Importance-weight
ESS is not independent-chain ESS. Three repeats cannot establish complete
exploration or general placement robustness. Report any failures or material
repeat disagreements in the package status; do not silently adopt the reference
as validated. Posteriors condition on selected fixed dates/signs and prior bounds;
selection and location uncertainty are excluded.

Synthetic generating curves remain their original saved research artifacts,
including their historical denominator; they are not regenerated to match new
inference. Case two has no true fitted sigmoid coefficients or locations.
Toronto E/beta have no observed truth. Procedure development and previous
forecast inspection on these cases preclude claims of untouched prospective
validation.

Brandon's exact WIS scorer and expected-result fixture remain required for final
scoring equivalence. The handoff exports the 23 quantiles for the 11 agreed
central levels (10–90%,95%,98%) and median. Local RMSE/coverage/width are labelled
diagnostics; earlier three-interval WIS is not substituted. Four-model figures
also need verified Wiener/spline outputs and David's Toronto effective-complexity
alignment. No missing comparator result is invented.

OpenAI Codex assisted with implementation and verification. No unverifiable
backend model identifier is asserted. Human scientific/code review remains pending.

## Compact published summaries

See [results](RESULTS.md) for the completed total-population 27-configuration
timing study and the new, independently checked 11-interval WIS implementation.
Those scores use the same documented daily targets. Equivalence to Brandon’s
implementation remains unverified. Public summary reproduction uses only bundled
CSV/JSON files; no collaborator archive or research folder is imported.
