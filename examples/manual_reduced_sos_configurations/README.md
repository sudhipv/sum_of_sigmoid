# Manual and reduced automatic SoS configurations

This lightweight package contains existing posterior samples, observations, settings and runnable forward-model code. It contains **no precomputed posterior trajectories or archived reference arrays**. Every run calculates the trajectories; no fitting, prior draws or TMCMC occur. Keep the full verified packages separately.

## Quick start

From the repository root, enter `examples/manual_reduced_sos_configurations/`, which contains `forward_model.ipynb` and `run.py`:

```bash
cd examples/manual_reduced_sos_configurations
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab forward_model.ipynb
```

Open **forward_model.ipynb** and execute all cells in order. `CASE="all"` calculates every supplied configuration; change it to one ID below to calculate only that case. The notebook calls the packaged Euler model for every original posterior row, through day 199.9, then plots fitting/forecast I, E and beta. It may take several minutes and prints progress. Posterior medians/means are across trajectories, not trajectories at median/mean parameters.

For a terminal-only calculation, from the same directory:

```bash
python run.py --list
python run.py --case all
```

Available IDs:

- `case1_manual`
- `case1_reduced`
- `case2_manual`
- `case2_reduced`
- `toronto_manual`
- `toronto_reduced`

Results go to a new `outputs/forward_<timestamp>/<id>/` directory (or `--output <new-directory>`). Generated files: `daily.npz` (all original rows of S/E/I/R/D/beta, daily time 0–199, original and summary weights, row IDs, locations and validity), `summaries.npz`, `training_NLL.npz`, `physical_validity.csv`, `model_configuration.json`, `run.json`, `fit_forecast.pdf`. Full native-grid validity is audited before daily export. Outputs are created when you run the package and are not included in this ZIP.

Plots show pointwise medians and central 95% **model-output** intervals, with observed counts and synthetic generating truth where applicable. They exclude observation noise; no observation-predictive scoring or substitute WIS implementation is performed. Training is 0–159; forecasts are 160–199. For real examples day0 is April 1, 2020, day159 September 7, day199 October 17. Synthetic truth and noisy observations are separate quantities.

## Inputs and file map

- `cases/<id>/posterior_samples.csv`: original parameter rows and original `weight`; no duplicate sample matrix.
- `cases/<id>/configuration.json`: population, total-population denominator, fixed rates, dt, sigma, locations/width, units, relative data paths and declared prior supports.
- `cases/<id>/parameter_order.json`, `TMCMC_priors.json`, `parameter_priors_and_ML_bounds.csv`: complete names/mapping, priors and distinct ML bounds.
- `cases/<id>/sampling_provenance.json`: historical sampler/worker/seed information; no inference entry point is included.
- `datasets/<dataset>/observations.csv`: exact observations, calendar/model times and training/forecast masks; shared once across configurations of the same dataset.
- `datasets/case1/generating_truth.npz` and `datasets/case2/generating_truth.npz` where supplied: original generating curves, not posterior ensembles.
- `cases/ottawa/exclusion_mask.npy` where supplied: original mask; all 4,000 parameter rows remain in the CSV.
- `forward_model.py`: exact scientific model/likelihood; `posterior_forward.py`: input validation, calculation, weighting and plots; `run.py`: optional CLI.
- `MODEL.md`, `LICENSE`, `CITATION.cff`: assumptions, limitations, conditioning, attribution and data-source information.

## Weights and Ottawa

Explicit sample weights are normalized; arbitrary row counts and nonuniform weights are supported. Equal positive weights use NumPy linear sample quantiles; unequal weights use an inverse weighted empirical CDF. All row IDs and original weights are retained. E0/I0 are always the last two internal parameters; S0=N-E0-I0, R0=D0=0.

For the exact supplied Ottawa set, the default `CONDITIONING="auto"` uses its original mask: 3,996 of 4,000 rows contribute to **both** fit and forecast summaries. The four excluded rows are still propagated/audited and saved with zero summary weight. The package does not conceal this conditioning. It never clips, changes parameters or replaces samples. Other supplied cases retain all rows. Read MODEL.md for the actual training-support and forecast-validity policies.

## Compatible alternative samples (optional)

Use `SAMPLE_FILE` and `CONFIG_FILE` in the notebook, or:

```bash
python run.py --config cases/case1_manual/configuration.json --samples my_samples.csv --output outputs/my_samples
```

Supply required named columns `beta0,A_day<location>...,E0,I0` and explicit `weight`; the exact names are in configuration.json. Optional `original_index_zero_based` must be unique integers. Alternatively labelled NPZ requires `parameters`, `parameter_names`, `weights` (optional row IDs/locations). A different location set or term count requires matching configuration and supports; values are validated without changing bounds. Alternative samples **do not inherit Ottawa's mask**. Invalid positive-weight paths are reported, raw arrays retained and summaries/plots withheld. No independent reference files, exact archived NLL matches or local experiment folders are required to calculate compatible samples.

Numerical verification against the previously accepted results is maintained externally by the authors; no reference ensembles/checking archives are bundled. Scientific settings and the full original packages remain unchanged.

## Intentional standalone model snapshot

`forward_model.py` is a byte-identical copy of the repository's
[automatic forward model](../../automatic_tuning/sumsigmoid/model.py) at
commit `668c6cc2eb800ce886d34bf01e8e330360026849`. Its SHA256 is
`b9593cc2cd1c08b5b9fe0ce8293b2f779a56c2511b6ef139c59a65b939483fa8`.
The local copy and matching `posterior_forward.py` helper are intentionally
retained so the notebook and CLI remain standalone and preserve the verified
parameter mapping and numerical behavior. They do not import or execute TMCMC.
The two example directories deliberately retain their own copies rather than
changing the tested import layout. Original license and citation files are
retained with each standalone example.

Local `.gitattributes` preserves LF line endings for the model snapshot and
parameter CSVs so checkout settings do not invalidate their provenance hashes.
