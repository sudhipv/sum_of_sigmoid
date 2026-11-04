# Preparation, inference and saved-posterior reconstruction

Human review pending. This integration changes interfaces and packaging, not
scientific rules. No new fitting, TMCMC or scientific experiment was performed.

## Supported paths

1. `run.py prepare` → `sumsigmoid.workflow` → unchanged `candidates.prepare`
   (`legacy`, default; optional `turn_optional`) → unchanged joint ML/acceptance.
2. `run.py tmcmc --prepared DIR` → validated preparation → run-owned snapshot →
   `sumsigmoid.target.logtarget` → shared `sumsigmoid.model.Model` → original
   sampler. `--dry-run` checks the actual callback without sampling.
3. `run.py posterior` or one of `notebooks/*.ipynb` → `sumsigmoid.posterior` →
   the **same** `sumsigmoid.model.Model.forward`, applied to saved particles.

The accepted primary settings are total `S+E+I+R+D`, inferred beta0/all
amplitudes/E0/I0, 11/10/16 fixed candidate dates (14/13/19 parameters), no
provisional day156, broad ML-derived signed Uniform supports of magnitude0.5,
initial conditions in [0,.001 population], original worker behavior and original
sampler sources at e412562ac57cff503f0f6cc4c2bd9d63da04785b. Case two uses the
isolated fixed50 control copy (no move-count adaptation); case one and Toronto
retain original adaptive controls. See [scientific method](SCIENTIFIC_METHOD.md)
and [sampler limitations](SAMPLER_LIMITATIONS.md). Completion of tempering is
not a convergence test, and initial parallel RNG states can duplicate.

The optional turn consolidation already exists: set `procedure` to
`turn_optional` in a separate input configuration. It applies the existing
legacy reduction first and then the existing turn association, not BIC or
location refinement. Primary default `legacy` yields 11/10/16; the accepted
optional alternative yields 10/9/11. No thresholds or rules were altered.

## Commands from the repository root

```bash
python -m pip install -r automatic_tuning/requirements.txt
python -m pip install -r automatic_tuning/handoff/requirements-notebook.txt
python automatic_tuning/run.py prepare --config automatic_tuning/examples/case1.json --output automatic_tuning/outputs/my_case1
python automatic_tuning/run.py prepare --config automatic_tuning/examples/case2.json --output automatic_tuning/outputs/my_case2
python automatic_tuning/run.py prepare --config automatic_tuning/examples/toronto.json --output automatic_tuning/outputs/my_toronto
```

For **each** selected preparation (choose one, do not submit all accidentally):

```bash
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case1 --dry-run
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case1
python automatic_tuning/run.py posterior --inference automatic_tuning/outputs/my_case1/tmcmc --data automatic_tuning/saved_baselines/case1/observed_daily.csv --truth automatic_tuning/saved_baselines/case1/generating_curves.npz --output automatic_tuning/regenerated/my_case1

python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case2 --dry-run
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case2
python automatic_tuning/run.py posterior --inference automatic_tuning/outputs/my_case2/tmcmc --data automatic_tuning/saved_baselines/case2/observed_daily.csv --truth automatic_tuning/saved_baselines/case2/generating_curves.npz --output automatic_tuning/regenerated/my_case2

python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_toronto --dry-run
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_toronto
python automatic_tuning/run.py posterior --inference automatic_tuning/outputs/my_toronto/tmcmc --data automatic_tuning/saved_baselines/toronto/observed_daily.csv --output automatic_tuning/regenerated/my_toronto
```

These commands describe future user execution; no sampling was launched for
integration verification. The original sampler uses native `cpu_count()`;
the runner blocks when it differs from configured workers (accepted count8).
Changing that setting is an explicit configuration decision, not an independent
worker-stream correction. Windows and other CPU environments are not validated.

Alternatively open `notebooks/synthetic_case_one.ipynb`,
`notebooks/synthetic_case_two.ipynb` or `notebooks/toronto.ipynb`. All use one
short path/configuration cell. Relative path resolution supports the notebook
directory, standalone package root and repository-root VS Code kernels. Change
only input/output paths and `DRAW_ID` to use another verified run; fixed dates,
signs and priors always come from its saved configuration, never notebook cells.

## Input and artifact contract

The preparation input fields are documented in [the README](../README.md).
`dataset_id` and `observation_quantity` explicitly identify these examples;
the dataset is also tied to embedded training counts and SHA256 checks.
Preparation seed controls feasible ML starts; `tmcmc.seed` controls original
prior draws/worker behavior. Notebook observation-noise seed is separate.

`prepared_run.json` (schema1) includes:

- Dataset ID, quantity/units, source reference and training input hashes.
- Training counts, model-day/calendar mapping, last observed day and forecast
  boundary, model settings, population and initial-state construction.
- Fixed dates, internal parameter order, date-labelled amplitudes and units.
- Joint ML vector/NLL, acceptance record, signs and proposed Uniform supports.
- Procedure choice, beta admissibility and canonical content hash.

Inference rejects unresolved ML, changed input hashes, parameter/order/date
mismatches, changed denominator/supports/signs and a mismatched ML likelihood.
It writes `input_snapshot/` plus `run_configuration.json` **before sampling**.
Parent and workers bind that snapshot. Later edits to preparation cannot alter
the sampling callback or plotting inputs. Snapshot copies are validated by hash;
this is an integrity convention, not filesystem access control.

Alongside raw `particles.npz`, inference writes `posterior_samples.csv` and
`parameter_order.json`. Columns are `draw_id,weight,A0,A_day23,...,E0,I0` for
the actual dates, retaining all final particles and equal weights. Raw engine
records remain untouched; reporting checks decode its documented constant
Uniform-prior recording offset. The likelihood itself remains unchanged.

Posterior evaluation takes a **separate** daily `day,I,is_observed` CSV on
0–199 and optional synthetic `generating_curves.npz`. Training entries must
exactly match the saved inference input. For Toronto omit truth: it has real
active-case observations and no known latent E/I/beta. For synthetic case one,
post-cutoff zeros are placeholders, not observations. Supplied original values
are preserved and their mask is explicit. A synthetic truth NPZ has time and
S/E/I/R/D/beta on0–199.9 by0.1. No coefficient/location recovery claim is made
for case two. Held-out values are used only for final plots/scores.

`run.py posterior --inference` converts the run-owned named samples/configuration
into `OUTPUT/inputs/`, validates them, then calls the same notebook helper.
An existing exported run can instead use `--configuration`, `--samples`,
`--data`, optional `--truth`, `--output` and optional `--reference`.
`--draw-id N` selects one saved row; all-sample mode is the default. A run
failing the existing near-ML training screen is blocked before forecast export;
passing that screen still does not prove posterior convergence.

## Regenerated outputs and definitions

- `latent_predictive_native.npz`: E/I/beta and separate `I_obs`, axes(draw,time),
  native time0–199.9, weights, draw IDs and training log likelihoods.
- `daily.npz`: integer days0–199 and, for ensembles, analytic predictive quantiles.
- `summaries.npz`: means/medians and linear empirical2.5/97.5% latent quantiles.
- `fit_forecast_beta.pdf` and inline vector SVG; no new samples are generated.
- `metrics.json`, `diagnostics.json`, `verification.json` and exact run config.

Propagation starts from each sample's inferred E0/I0 and continues without a
cutoff reset or assimilation. Inference includes days0–159; proposed fit scoring
uses1–159 and forecasts160–199. All4000particles contribute. Latent bands are
pointwise, not simultaneous; `I_obs=I*(1+epsilon)`, epsilon~Normal(0,.1²), is
separate. Analytic observation quantiles invert the all-particle Gaussian
mixture; they are not finite-noise-draw empirical quantiles.

Synthetic scores target generating latent I with latent intervals. Toronto
scores observed active I using predictive intervals. WIS uses eleven central
coverages10,20,…,90,95,98%, median weight.5, interval weightsalpha/2, divisor11.5,
and an arithmetic mean over time. `scoring.py` is the identical accepted
`results/wis11.py` implementation; Brandon's function remains unavailable.

## Verification and historical alternatives

All three final notebooks were executed in a clean copy with only declared
dependencies and existing accepted samples. Compact references check native
summaries, direct likelihoods, row0 E/I/beta/I_obs, predictive quantiles and
daily scores with atol1e-7/rtol1e-12. Deterministic checks also compare saved ML
and representative posterior vectors through preparation, actual inference
callback, fresh worker and notebook Model, and selected row1773 against the
ensemble including its predictive noise. These are not new sampling tests.

Historical inference scripts, MLE notebooks and plotting notebooks retain
their independent Euler implementations and earlier scientific settings.
`handoff/forward_model.ipynb` and its helpers remain the older all-case bundle
interface. New supported notebooks are in `notebooks/`; the earlier large
Brandon package and archived samples/figures were not changed or relabelled.
See the local integration audit for David's verified comments and limitations.

Original scientific code is by the research authors; interface, notebook and
documentation preparation was assisted by OpenAI Codex. No unverifiable model
identifier is asserted. Human review remains pending; no commit or push is part
of this integration task.
