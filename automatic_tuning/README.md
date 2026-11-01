# Automatic sum-of-sigmoids preparation

**Human scientific and code review remains pending.** This folder is the
publication execution path. Historical notebooks and experiments are preserved
elsewhere; their results have not been relabeled or rerun.

The primary pipeline is training-only feature detection -> identical-date merge
-> frozen legacy consolidation -> joint ML -> signed Uniform prior configuration
for optional TMCMC. ML jointly fits beta0, every amplitude, E0 and I0. The
default transmission denominator is **living population S+E+I+R**. The separate
total-population experiment is not a publication default or configuration option.

TMCMC uses `sumsigmoid/original/tmcmc_mod.py` and `pdfs.py` **byte-for-byte** from
commit `e412562ac57cff503f0f6cc4c2bd9d63da04785b`. They contain no added worker
initializer or fixes. See [source provenance](docs/PROVENANCE.md) and
[saved-result provenance](docs/SAVED_RESULTS.md).

## Install and run

From the repository root, using Python 3.11 (tested 3.11.5):

```bash
cd automatic_tuning
python -m pip install -r requirements.txt

# Preparation + ML only. Each output directory must be new.
python -B run.py prepare --config examples/case1.json --output outputs/case1
python -B run.py prepare --config examples/case2.json --output outputs/case2
python -B run.py prepare --config examples/toronto.json --output outputs/toronto
```

These are independent examples, not a command to launch a research campaign.
All three small inputs contain the original 160 training counts with explicit
day/date labels. Toronto is observed active cases; the synthetic inputs are
noisy infectious counts. No held-out values, generating trajectories, fitted
parameters or posterior samples are bundled as example input. The explicit
430-person E0/I0 guesses are example optimizer seeds, never generic fixed values.

The terminal prints each ML attempt, final NLL, parameter file, metrics, figure,
optimizer status, proposed TMCMC configuration/priors and optional command.
Unresolved ML retains its diagnostics and blocks signed-prior preparation and
sampling. No manual starting-point editing, pruning or location search follows.

## Configuration inputs

Paths in JSON are relative to that JSON file. `--output` is relative to the
current directory. Existing result directories are never overwritten.

| Input | Meaning/default |
|---|---|
| `data.file` | Required CSV of daily active/infectious counts in persons; not incidence or proportions. |
| `data.header` | Whether the first row contains column names; default true. |
| `data.delimiter` | CSV delimiter; default comma. |
| `data.date_column`, `count_column` | Required header names, or zero-based indices when `header:false`. |
| `time_origin` | Required numeric origin or ISO calendar date; defines model day zero. |
| `training_cutoff` | Required inclusive model day/date. Exact daily days 0..cutoff must be present. |
| `population` | Required initial population N0, in persons. No Toronto value is assumed generically. |
| `initial_guess.E0`, `.I0` | Required feasible initial guesses in persons. Both remain free throughout ML and TMCMC. |
| `procedure` | `legacy` (primary default), or separately named `turn_optional`. |
| `seed` | ML multistart RandomState seed; example/default 20261031. Does not set the TMCMC seed. |
| `output_directory` | Required new results directory; overridden by `--output`. |
| `model` | Explicit fixed-rate assumptions; defaults below. Step, noise and width are frozen in this release. |
| `bounds` | Optional explicit restatement of frozen supports; different bounds are rejected, not silently substituted. |
| `tmcmc.enabled` | Default false. True explicitly requests sampling after accepted ML. |
| `tmcmc.particles` | Default 4000 fresh base-prior particles. |
| `tmcmc.seed` | Default 106; controls parent prior/resampling stream and the CLI's original shared spawn-import seed behavior. |
| `tmcmc.workers` | Default 8; must match the native `multiprocessing.cpu_count()` before sampling. The unchanged engine determines count; this field does not override it. |

No interpolation, rescaling or smoothing enters the likelihood. Counts after
the cutoff are ignored. Missing/duplicate daily dates are rejected. The frozen
detector supports **cutoffs 1..159**; longer windows are rejected for review.
Leading-zero/degenerate curves may require review; all-zero curves are rejected.

Model defaults: explicit Euler step 0.1 day, sigmoid width 1 day, Gaussian
observation SD 0.1*I, and rates gamma_e=1/15, gamma_i=1/5, gamma_r=1/11,
gamma_d=1/750 per day. Fixed rates are configurable assumptions; only the stated
defaults were checked against saved results. Model.py documents the equations.
S0=N0-E0-I0, R0=D0=0. Total beta is constrained >=-1e-12 on the training grid.

ML supports: beta0 [1e-6,0.5], every amplitude [-0.5,0.5], E0/I0 [0,0.001*N0].
Proposed TMCMC supports: beta0 [0,0.5], amplitudes [0,0.5] or [-0.5,0] according
to accepted joint ML signs, and the same E0/I0 bounds. These are the frozen
matched-procedure supports, not every historical paper run's priors.
**Bounds are assumptions, not ML confidence intervals.** Data-selected signs
and fixed locations exclude their selection uncertainty from optional inference.

## Optional original TMCMC

Inspect the actual imports, hashes, parameter ordering, priors, callback NLL,
native worker count and start method without drawing particles or starting workers:

```bash
python -B run.py tmcmc --prepared outputs/case1 --dry-run
python -B run.py tmcmc --prepared outputs/case2 --dry-run
python -B run.py tmcmc --prepared outputs/toronto --dry-run
```

Only when deliberately requesting a fresh sampling run:

```bash
python -B run.py tmcmc --prepared outputs/toronto
```

The package calls the original five-argument interface with original mutation
defaults. It adds no Pool initializer and does not override CPU count or the
native multiprocessing start method. With the supported CLI, spawn imports
repeat the configured shared seed, matching the original runner behavior;
fork may inherit shared RNG state. **Initial worker RNG states can duplicate.**
Independent streams and bitwise reproducibility are not promised. Dynamic
task assignment can change which worker supplies a particle's random numbers.

Raw samples and all original stage fields are retained. Outside the sampler,
every final likelihood is recalculated, its constant Uniform recording offset
is decoded, and sampled-mean training RMSE/curve are compared with saved ML.
No sample filtering, prior changes or automatic repeats occur. Output remains
`completed_requires_training_review`; reaching exponent 1, importance-weight
ESS or many distinct particles does not establish adequate exploration.
No forecasts are implemented here. **No new sampling was run for this package.**
See [sampler limitations](docs/SAMPLER_LIMITATIONS.md) before optional inference.

## Verification and files

```bash
python -B -m unittest discover -s tests -v
python -B verify_examples.py
```

Tests use deterministic records/vectors and intercept any full sampler call.
The reference check reproduces 11 case-one, 10 case-two and 16 Toronto primary
candidates and saved ML likelihood values without new fitting. The optional
turn procedure remains separate. See [file guide](docs/FILE_GUIDE.md) and
[scientific method](docs/SCIENTIFIC_METHOD.md) for exact detector/ML thresholds.

Primary outputs: `candidates.json`, full detection/consolidation audits,
`ML_result.json`, `ML_parameters.csv`, `ML_bounds.json`, `optimizer_attempts.json`,
`training_metrics.json`, `fitted_values.csv`, `training_fit.pdf`,
`TMCMC_priors.json`, `TMCMC_configuration.json`, `status.json` and the explicit
`optional_TMCMC_command.txt`. Generated outputs stay ignored.

Only `PUBLICATION_FILES.txt` belongs to this publication change. Parent history
already contains tracked outputs; ignore rules do not remove them. Historical
BIC/refinement, day-156 overrides, restricted priors and total-denominator runs
remain separate. OpenAI Codex assistance is documented without an unverifiable
backend model identifier. Existing GPL-3.0 attribution and citation are retained.
