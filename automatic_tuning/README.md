# Automatic location selection

A training-only procedure for proposing sigmoid locations from an observed
active-case curve, fitting a sum-of-sigmoids transmission model, and preparing
signed priors for optional TMCMC. The steps and thresholds below were specified
before the reported comparisons and are kept unchanged between datasets.

The primary procedure is **feature detection → identical-date merging →
consolidation of associated flags → joint maximum likelihood (ML) → signed
Uniform-prior configuration**. It uses a seven-day smoother only for detection;
raw counts enter the likelihood. Locations are heuristic candidates inferred
from observed features, not known transmission-change times. BIC pruning,
location refinement, the provisional day-156 override and ML-centered restricted
priors are outside this default workflow.

Start with the [scientific method](docs/SCIENTIFIC_METHOD.md),
[accepted results](docs/RESULTS.md), [source provenance](docs/PROVENANCE.md) and
[sampler limitations](docs/SAMPLER_LIMITATIONS.md). This is research code;
numerical checks do not establish posterior convergence or scientific validity.

## Installation and three examples

From the repository root, with Python 3.11 (tested with 3.11.5):

```bash
cd automatic_tuning
python -m pip install -r requirements.txt

# Each command performs automatic location selection and joint ML only.
# Use a new output directory for each execution.
python -B run.py prepare --config examples/case1.json --output outputs/case1
python -B run.py prepare --config examples/case2.json --output outputs/case2
python -B run.py prepare --config examples/toronto.json --output outputs/toronto
```

These examples are self-contained. They import only this package and its pinned
Python dependencies, and load the bundled 160-observation training CSVs. No
excluded experiment folder, posterior array or collaborator archive is required.
Generating curves and held-out observations are not inference inputs.

| Example | Primary sigmoid count / inferred parameters | Optional sampling controls |
|---|---:|---|
| Synthetic case one | 11 / 14 | Original adaptive controls: initial 5 attempts, cap increasing to 10 |
| Synthetic case two | 10 / 13 | Fixed 50 attempted moves per chain at every stage; adaptation disabled |
| Toronto | 16 / 19 | Original adaptive controls: initial 5 attempts, cap increasing to 10 |

The accepted configurations use 4,000 prior particles, master seed 106 and eight
workers. **`examples/case2.json` explicitly selects `fixed50_experimental`.**
The older `case2_fixed50_experimental.json` remains an equivalent explicit
control example with a separate output directory. The general configuration
fallback is adaptive; case two's named example does not use that fallback.

The terminal reports each optimizer attempt, the final full NLL, fitted-parameter
CSV, training metrics/figure, optimizer diagnostics, proposed TMCMC configuration
and the optional sampling command. Existing outputs are never overwritten.

## Input fields and assumptions

Paths in JSON are relative to the JSON file. A CLI `--output` path is relative
to the current working directory. The three examples explicitly record the
model constants, bounds and case-specific mutation policy.

| Field | Meaning |
|---|---|
| `data.file` | Required daily active/infectious-count CSV, in persons; not incidence or population fractions. |
| `data.header`, `delimiter` | Header presence (default true) and delimiter (default comma). |
| `data.date_column`, `count_column` | Required header names, or zero-based column indices without a header. |
| `time_origin` | Required numeric origin or ISO calendar date defining model day zero. |
| `training_cutoff` | Required inclusive model day/date. Consecutive days 0–cutoff must exist. Supported cutoff: 1–159. |
| `population` | Required initial total population N0, in persons. |
| `initial_guess.E0`, `I0` | Required feasible optimizer guesses in persons. Both remain free in every ML fit and TMCMC. The examples' 430-person guesses are not generic fixed values. |
| `procedure` | `legacy`: primary associated-flag consolidation; `turn_optional`: separately documented additional turn consolidation. |
| `seed` | ML-start generation seed (examples: 20261031). One reproducible stream continues across retry batches; this does not set TMCMC's seed. |
| `output_directory` | Required new results directory; `--output` overrides it. |
| `model` | Euler step, observation SD multiplier, sigmoid width and four fixed rates; values below. |
| `bounds` | Explicit restatement of supported ML/prior bounds; alternative supports are rejected for separate scientific review. |
| `tmcmc.enabled` | Default false. True explicitly requests sampling after accepted preparation. |
| `tmcmc.particles` | Fresh prior-particle count; accepted examples use 4000. |
| `tmcmc.seed` | Parent prior/resampling seed and original CLI spawn-import seed; accepted examples use 106. Does not guarantee identical parallel mutation draws. |
| `tmcmc.workers` | Expected native CPU count; accepted runs used 8. The original sampler uses `multiprocessing.cpu_count()` and is not overridden. |
| `tmcmc.mutation_policy` | `original_adaptive` or the explicit case-two `fixed50_experimental` controls. |

Missing or duplicate dates and all-zero curves are rejected. There is no
interpolation or rescaling. The detector receives only observations up to the
cutoff; a candidate's full-window availability must also be satisfied.

The shared forward model uses **N(t) = S+E+I+R+D** in the infection-flow
denominator for ML, TMCMC and propagation. It conserves N0 up to roundoff.
Initial states are S0=N0−E0−I0 and R0=D0=0. Defaults are Euler step 0.1 day,
sigmoid width 1 day, rates gamma_e=1/15, gamma_i=1/5, gamma_r=1/11 and
 gamma_d=1/750 per day. The observation distribution is Gaussian with mean I
and **SD 0.1×I**, including its normalization and day-zero observation. Smoothing
never replaces likelihood observations. Total beta must be ≥−1e−12 on the
training Euler grid. See the [equations and exact rules](docs/SCIENTIFIC_METHOD.md).

ML jointly fits beta0, every amplitude, E0 and I0: K+3 parameters. Bounds are
beta0 [1e−6,0.5], each amplitude [−0.5,0.5], E0/I0 [0,0.001×N0]. The existing
SLSQP scaling and settings are preserved. Complete batches of four distinct
feasible starts, at most 12 attempts. Acceptance requires two starts within
1e−4 of the best valid full NLL and matching nonzero signs throughout that
competitive group. Poorer fits do not veto agreement; an unconfirmed better
fit cannot be replaced by a worse reproduced one. Unresolved preparation saves
its diagnostics and blocks signed priors/TMCMC. This is not global optimization.

Proposed TMCMC priors are beta0 Uniform[0,0.5], amplitudes Uniform[0,0.5] or
Uniform[−0.5,0] according to accepted joint-ML signs, and E0/I0 Uniform[0,0.001×N0],
with the same beta-admissibility restriction. **Bounds are configured assumptions,
not ML confidence intervals.** Inference is conditional on selected signs and
fixed locations; their selection uncertainty is excluded.

## Optional original TMCMC

Inspect configuration, actual imports, source hashes, parameter order, priors,
callback NLL and worker policy without sampling:

```bash
python -B run.py tmcmc --prepared outputs/case1 --dry-run
python -B run.py tmcmc --prepared outputs/case2 --dry-run
python -B run.py tmcmc --prepared outputs/toronto --dry-run
```

Then explicitly request one fresh run, for example:

```bash
python -B run.py tmcmc --prepared outputs/case2
```

The two files in `sumsigmoid/original/` are **byte-for-byte** from commit
`e412562ac57cff503f0f6cc4c2bd9d63da04785b`. There are no changes to tempering,
resampling, covariance, acceptance or distributions. No added worker initializer
is used. Case one and Toronto retain the original adaptive move controls.
Case two creates an isolated output-local copy changing only the existing
`Adap_calc_Nsteps` flag from `yes` to `no`, and supplies both existing move caps
as 50. The original files remain unchanged. Merely raising the caps would not
force 50 moves. Actual dispatched and completed attempts are recorded; rejected
proposals count as attempts. This is an accepted case-specific setting, not a
new sampling algorithm or a convergence guarantee.

Original parallel initialization can duplicate worker RNG states. Dynamic
scheduling prevents a promise of bit-for-bit mutation repetition. The native
start method and CPU count are recorded. On a machine with a different native
count, preparation still works, but optional sampling stops until the user
explicitly acknowledges that count in configuration; that is not an exact
reproduction of the accepted eight-worker environment.

Every final likelihood is recalculated outside the sampler, including decoding
its constant Uniform recording offset for reporting. Training fits and physical
validity require review. Reaching tempering exponent 1, stage importance ESS or
many distinct particles does not prove adequate exploration. Stage importance
ESS is not independent-chain ESS. There are no automatic repeats or forecasts
in the preparation CLI. See [limitations](docs/SAMPLER_LIMITATIONS.md).

## Saved results and portable figure reproduction

[Results and interpretation](docs/RESULTS.md) contain the accepted baseline table
and all 27 timing comparisons. Only compact daily summaries, configurations,
metrics and selected figures are bundled—no posterior particles or trajectory
ensembles. Reproduce the saved scores and figures without any sampling:

```bash
python -B results/reproduce.py --check-only
python -B results/reproduce.py --output outputs/summary_figures
```

Figures use saved **daily** posterior summaries. Their native 0.1-day archives
remain unchanged outside Git. Synthetic intervals score latent I against saved
generating I; Toronto intervals include observation noise and score reported
active cases. The 11-interval WIS implementation follows the documented published
convention and passed an independent numerical check; equivalence to Brandon's
code remains unverified. Historical three-interval scores are not mixed in.

Large saved-particle reconstruction is separately documented in the
[collaborator notebook guide](handoff/README.md); that optional archive is not
required by the examples or the compact result scripts.

## Verification and output files

```bash
python -B -m unittest discover -s tests -v
python -B verify_examples.py
python -B results/reproduce.py --check-only
```

Tests do not launch full sampling. Checks cover original source hashes, candidate
counts 11/10/16, the verified total-denominator reference, population conservation,
ML/worker-callback likelihood agreement, retry acceptance and fixed-50 dispatch.
Historical living-denominator checks are explicitly separate from the final
inference model. Public examples are also tested from an isolated copy without
excluded research folders. See [verification](docs/VERIFICATION.md).

Each preparation writes `configuration.json`, `input_record.json`, `detection.json`,
`identical_date_groups.json`, `legacy_consolidation.json`, `candidates.json`,
`ML_bounds.json`, `ML_result.json`, `ML_parameters.csv`, `optimizer_attempts.json`,
`training_metrics.json`, `fitted_values.csv`, `training_fit.pdf`, `status.json`,
and—only when accepted—`TMCMC_priors.json`, `TMCMC_configuration.json` and
`optional_TMCMC_command.txt`. Paths are printed to the terminal.

[PUBLICATION_FILES.txt](PUBLICATION_FILES.txt) lists the reviewed package files.
Original scientific notebooks and historical outputs remain preserved elsewhere.
No sampler result is relabeled as an unchanged historical paper posterior.
[Provenance](docs/PROVENANCE.md) records original authorship and OpenAI Codex
assistance without inventing a backend model identifier. GPL-3.0 and the existing
citation are retained.
