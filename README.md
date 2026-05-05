### This repository contains the codes to carry out Bayesian parameter estimation and predictions for the sum of sigmoid model of infection rate.

The journal article associated with the code is submitted to PLOS-One and will be available online soon.

### SETUP

The code was developed with Python 3.11.5. The main Python dependencies are listed in `requirements.txt`.

Using conda:

```bash
conda create -n sumsigmoid python=3.11.5
conda activate sumsigmoid
pip install -r requirements.txt
```

Run the synthetic case 1 quickstart workflow:

```bash
python examples/quickstart.py
```

The quickstart runs the existing synthetic case 1 MCMC script and plotting notebook. MCMC outputs are written to `data/out/`, MCMC diagnostic figures to `figs/mcmc/synth_case1/`, prediction figures to `figs/predictions/test/`, and the executed plotting notebook copy to `examples/notebooks/`.

The MCMC scripts use local multiprocessing by default:

```python
parallel_processing = 'multiprocessing'
```

For cluster runs with:

```python
parallel_processing = 'mpi'
```

`mpi4py` must be installed against the MPI implementation available on the cluster. The `mpi4py` package is included in `requirements.txt` without a pinned version for now because the tested cluster version should be recorded from the cluster environment.

### REPO STRUCTURE

1. PHU_Data : Infection and population data for all the PHUs
2. data : Input and output data used by the workflows
   - `data/in`: retained input data and reference truth files
   - `data/out`: retained or generated MCMC samples and sampler log files
3. figs : All the figures generated from different codes located here
4. inference : main source code for manual tuning, mle and mcmc
5. misc : Optional or exploratory files that are not part of the main reproduction workflow

### WORKFLOW

The repository contains three main data workflows:

1. Synthetic data case 1
2. Synthetic data case 2
3. Real Toronto infection data

The general workflow is:

1. Generate or load the data set.
2. Optionally use manual tuning to inspect the data, estimate inflection points, and identify parameter signs.
3. Optionally use maximum likelihood estimation (MLE) to refine parameter values. The MCMC scripts use broad uniform bounds, so MLE is not required for the example workflow.
4. Run MCMC inference to estimate posterior parameter samples.
5. Use the saved MCMC output files to plot posterior fits and compare against the truth or observed data.

#### Manual tuning

Manual tuning is optional, but recommended before MLE and MCMC because it gives useful initial information about inflection points and parameter signs.

- `inference/manual_tuning/Infection_data_apr1-dec31.ipynb`: plots infection data.
- `inference/manual_tuning/manual_tune.ipynb`: manually tunes model parameters.

#### Synthetic data case 1

1. Generate the synthetic data:
   - `inference/manual_tuning/generate_synethtic_Mult.ipynb`
2. Optionally run MLE:
   - `inference/mle/1phu_synthetic_MLE.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_synth_1.py`
4. Plot the MCMC results:
   - `inference/plotting/plot_synth_case1.ipynb`

For a self-contained reviewer example, run `examples/quickstart.py`.

#### Synthetic data case 2

1. Load the synthetic case 2 data from `data/in/`.
2. Optionally run MLE:
   - `inference/mle/1phu_MLE_synthetic_case2.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_synth_2.py`
4. Plot the MCMC results:
   - `inference/plotting/plot_synth_case2.ipynb`

The notebook `inference/plotting/predictions_1phu_synth_case2_multiple.ipynb` contains additional plots for synthetic case 2.

#### Real Toronto data

1. Inspect and manually tune the Toronto infection data:
   - `inference/manual_tuning/Infection_data_apr1-dec31.ipynb`
   - `inference/manual_tuning/manual_tune.ipynb`
2. Optionally run MLE:
   - `inference/mle/1phu_real_MLE.ipynb`
3. Run MCMC inference:
   - `inference/mcmc/singlephu_mult_mcmc_real.py`
4. Plot the MCMC results:
   - `inference/plotting/plot_real_toronto.ipynb`

Some MAP and prediction notebooks are retained for now as secondary or exploratory analysis files, but they are not part of the main reproduction workflow above.

### LICENSE

Released under GPL-3.0. See `LICENSE` for details.
