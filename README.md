# Sum of Sigmoid Infection Rate Inference

This repository contains code for Bayesian parameter estimation and prediction using a sum-of-sigmoids model for the infection rate. The associated journal article has been submitted to PLOS ONE and will be linked here when available.

For **[Automatic location selection](automatic_tuning/README.md)**, start with the
dedicated README. It provides installation, self-contained synthetic-case-one,
case-two and Toronto examples, input definitions, scientific documentation and
[compact accepted results](automatic_tuning/docs/RESULTS.md).

The current automatic workflow uses the total-population denominator
`S + E + I + R + D`, free E0/I0 and the original TMCMC implementation.
Case one and Toronto use the original adaptive move controls; the accepted
case-two example uses fixed 50 attempted moves per chain at each stage.
The original sampler files remain unchanged; the case-two control copy is
isolated in the output directory. See the dedicated README for exact settings,
original worker-RNG limitations and optional sampling commands.

## Supported workflow: prepare → infer → plot

Use Python 3.11 and install the dependencies from the repository root:

```bash
python -m pip install -r automatic_tuning/requirements.txt
python -m pip install -r automatic_tuning/handoff/requirements-notebook.txt
```

**Route 1 — run a new workflow.** For example, synthetic case two:

```bash
python automatic_tuning/run.py prepare --config automatic_tuning/examples/case2.json --output automatic_tuning/outputs/my_case2
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case2 --dry-run
# Explicitly launch inference only when desired (this integration task did not):
python automatic_tuning/run.py tmcmc --prepared automatic_tuning/outputs/my_case2
python automatic_tuning/run.py posterior --inference automatic_tuning/outputs/my_case2/tmcmc --data automatic_tuning/saved_baselines/case2/observed_daily.csv --truth automatic_tuning/saved_baselines/case2/generating_curves.npz --output automatic_tuning/regenerated/my_case2
```

Replace `case2` with `case1` or `toronto` and use the corresponding example.
For Toronto omit `--truth`. The examples contain **training data only**;
the plot command's separate daily observation/reference inputs come from the
small saved-result package or must be supplied for your dataset. A valid plotting
table has columns `day,I,is_observed`, days 0–199, with missing or placeholder
observations explicitly marked `is_observed=0`. No held-out value enters inference.

Preparation writes `prepared_run.json`: dataset identity, time coordinates,
fixed dates, complete scientific settings, accepted joint ML, signs and priors.
Inference reads and validates it, then freezes its own `tmcmc/input_snapshot/`
and `tmcmc/run_configuration.json`. Named `posterior_samples.csv` columns map
amplitudes to dates. **Do not copy parameters between scripts.** The plot command
uses the run-owned configuration and stops on failed training checks. See the
[three-stage interface](automatic_tuning/docs/THREE_STAGE_WORKFLOW.md) for inputs,
outputs, sampler limitations and exact commands for all three cases.

**Route 2 — reproduce supplied results without inference.** Extract the small
handoff next to its README, or place its `saved_baselines/` directory inside
`automatic_tuning/`. Open and Run All in one of:

- [Synthetic case one posterior notebook](automatic_tuning/notebooks/synthetic_case_one.ipynb)
- [Synthetic case two posterior notebook](automatic_tuning/notebooks/synthetic_case_two.ipynb)
- [Toronto posterior notebook](automatic_tuning/notebooks/toronto.ipynb)

Select the Python environment above. `DRAW_ID=None` evaluates all supplied
samples; an integer selects one row. These notebooks never optimize or sample.
They regenerate continuous E/I/beta trajectories, separate latent and
observation-predictive outputs, figures and accepted scores, and compare compact
saved references. Generated ensembles are intentionally excluded from Git and the
small package. Source-only Git checkouts need the separately supplied samples.

The manual notebooks and quickstart below are **historical alternatives** with
their earlier settings, independent equations and hard-coded configurations.
They are preserved for provenance and are not the supported automatic pipeline.

## Table of Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Data Organization](#data-organization)
- [Reproduction Workflows](#reproduction-workflows)
  - [Synthetic Data Case 1](#synthetic-data-case-1)
  - [Synthetic Data Case 2](#synthetic-data-case-2)
  - [Real Toronto Data](#real-toronto-data)
- [MCMC Execution Notes](#mcmc-execution-notes)
- [Outputs](#outputs)
- [Related Repositories](#related-repositories)
- [Citation](#citation)
- [License](#license)

## Quickstart

After installing the dependencies, run the synthetic case 1 example:

```bash
python examples/quickstart.py
```

The quickstart runs:

1. `inference/mcmc/singlephu_mult_mcmc_synth_1.py`
2. `inference/plotting/plot_synth_case1.ipynb`

It writes MCMC samples to `data/out/`, MCMC diagnostic figures to `figs/mcmc/synth_case1/`, prediction figures to `figs/predictions/test/`, and an executed copy of the plotting notebook to `examples/notebooks/`.

The MLE notebook is intentionally not part of the quickstart. MLE is optional because the MCMC scripts use broad uniform bounds.

## Installation

The code was developed with Python 3.11.5. The main Python dependencies are listed in `requirements.txt`.

Using conda:

```bash
conda create -n sumsigmoid python=3.11.5
conda activate sumsigmoid
pip install -r requirements.txt
```

The quickstart executes a Jupyter notebook from the command line, so `nbconvert` and `ipykernel` are included in `requirements.txt`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `PHU_Data/` | Infection and population data for Ontario public health units. |
| `data/in/` | Retained input data and reference truth files. |
| `data/out/` | Retained or generated MCMC samples and sampler log files. |
| `figs/` | Generated figures from MCMC and prediction workflows. |
| `examples/` | Reviewer-facing quickstart runner and executed notebook outputs. |
| `inference/manual_tuning/` | Manual tuning notebooks and exploratory data inspection. |
| `inference/mle/` | Maximum likelihood estimation notebooks. |
| `inference/mcmc/` | MCMC scripts and the bundled TMCMC helper module. |
| `inference/plotting/` | Plotting notebooks for posterior fits and predictions. |
| `misc/` | Optional or exploratory files that are not part of the main reproduction workflow. |

## Data Organization

The `data/` directory is split into inputs and generated or retained outputs:

- `data/in/`: synthetic data, truth files, and real-data reference quantities used as inputs.
- `data/out/`: MCMC sample files, long chains, and TMCMC stat/log files.

This separation is intended to make it clear which files are loaded as inputs and which files are created or overwritten by inference runs.

## Reproduction Workflows

The repository contains three main workflows:

1. Synthetic data case 1
2. Synthetic data case 2
3. Real Toronto infection data

The general workflow is:

1. Generate or load the data set.
2. Optionally use manual tuning to inspect data, estimate inflection points, and identify parameter signs.
3. Optionally use MLE to refine parameter values.
4. Run MCMC inference to estimate posterior parameter samples.
5. Use saved MCMC outputs to plot posterior fits and predictions.

### Synthetic Data Case 1

Recommended quickstart:

```bash
python examples/quickstart.py
```

Manual workflow:

1. Generate or inspect the synthetic data:
   - `inference/manual_tuning/generate_synethtic_Mult.ipynb`
2. Optionally run MLE:
   - `inference/mle/1phu_synthetic_MLE.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_synth_1.py`
4. Plot MCMC results:
   - `inference/plotting/plot_synth_case1.ipynb`

### Synthetic Data Case 2

1. Load the synthetic case 2 data from `data/in/`.
2. Optionally run MLE:
   - `inference/mle/1phu_MLE_synthetic_case2.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_synth_2.py`
4. Plot MCMC results:
   - `inference/plotting/plot_synth_case2.ipynb`

Additional synthetic case 2 plots are retained in:

- `inference/plotting/predictions_1phu_synth_case2_multiple.ipynb`

### Real Toronto Data

1. Inspect and manually tune the Toronto infection data:
   - `inference/manual_tuning/Infection_data_apr1-dec31.ipynb`
   - `inference/manual_tuning/manual_tune.ipynb`
2. Optionally run MLE:
   - `inference/mle/1phu_real_MLE.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_real.py`
4. Plot MCMC results:
   - `inference/plotting/plot_real_toronto.ipynb`

Some MAP and prediction notebooks are retained as secondary or exploratory analysis files, but they are not part of the main reproduction workflow.

## MCMC Execution Notes

The MCMC scripts use local multiprocessing by default:

```python
parallel_processing = 'multiprocessing'
```

For cluster runs with MPI:

```python
parallel_processing = 'mpi'
```

`mpi4py` must be installed against the MPI implementation available on the cluster. The `mpi4py` package is included in `requirements.txt` without a pinned version because the tested cluster version should be recorded from the cluster environment.


## Related Repositories

https://github.com/BMGRobinson/IDM_bayesian_framework.git

## Citation

If you use this code, please cite the repository using the metadata in `CITATION.cff`.

## License

Released under GPL-3.0. See `LICENSE` for details.


[def]: https://github.com/BMGRobinson/IDM_bayesian_framework.git

## Publication scope

[automatic_tuning/PUBLICATION_FILES.txt](automatic_tuning/PUBLICATION_FILES.txt)
lists the automatic-workflow publication files. Generated outputs, the earlier
local `review_package/`, private files and research experiment folders are
excluded from this change and remain preserved locally. Existing tracked
historical artifacts remain tracked; ignore rules do not remove Git history.
See [saved-result provenance](automatic_tuning/docs/SAVED_RESULTS.md) before
comparing results generated by original and corrected-worker samplers.

Saved-particle reconstruction is documented in
[the collaborator notebook guide](automatic_tuning/handoff/README.md).
See [current validation status and per-case sampler controls](automatic_tuning/docs/VALIDATION_HANDOFF.md).
Case one and Toronto retain original adaptive controls. The accepted case-two
example explicitly uses the isolated fixed-50 control setting. Posterior arrays and research archives are kept outside this Git change.
