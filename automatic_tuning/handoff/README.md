# SoS saved-particle handoff

This source notebook reconstructs final SoS outputs without any new optimization
or TMCMC. Download the separate result archive linked in the final run report,
extract it, and open its top-level `forward_model.ipynb` in Jupyter/VS Code.
The archive contains this code, `sumsigmoid/`, inputs, labelled particles and
reference outputs for available cases. Read top-level `STATUS.json` first;
failed populations remain in the separate diagnostic archive and are not
distributed as posterior samples. No historical folders or personal paths are
needed. `case2` is the predesignated seed-106 reference. Validation repeats, when
cleared, are `case2_seed107` and `case2_seed108`; they are never selected using
forecast scores. Set `CASE` to one of these names, `case1`, `toronto` or `all`.
`all` runs only available, cleared case folders.

Use Python 3.11 with the parent package's pinned numpy/scipy/matplotlib versions.
Notebook display also uses IPython; a normal Jupyter Python kernel supplies it.
The command-line path needs only the three scientific dependencies:

```bash
python -m pip install -r requirements.txt
# Optional notebook/validation dependencies, pinned to the verified environment:
python -m pip install -r requirements-notebook.txt
```

The standalone archive also records the exact tested versions in `ENVIRONMENT.json`.

```bash
python -B handoff_tools.py --root . --case all --output regenerated --verify
```

From the Git source directory, select the extracted archive with `--root` or
set the notebook's `DATA_ROOT`; large result arrays are intentionally not tracked.
The builder is for maintainers, with explicit fresh-run and input paths:

```bash
python build_package.py --run /path/to/verified/run --case case1 \
  --inputs /path/to/inputs --output /path/to/new/handoff
```

Each case retains all 4000 fresh-prior final particles, with explicit column names,
fixed activation dates, units and weights. Both E0 and I0 are inferred. The model
uses **S+E+I+R+D**, Euler 0.1 day, sigmoid width 1 day, and observation SD 0.1*I.
Inference uses observations 0–159, while the proposed daily scoring masks are
1–159 for fit and 160–199 for forecast. The dense plotting grid is 0–199.9.
Latent E/I/beta and observation-predictive I are separate arrays.
Synthetic case one has noisy observations only through day159: later zeros in
the original file are placeholders. They remain byte-preserved but are marked
`is_observed=0` in `observed_daily.csv`, never plotted/scored as observed counts. Predictive
Gaussian draws have recorded local seeds; analytic mixture quantiles use all
particles and are not identical to empirical finite-noise quantiles. No clipping,
state reset, held-out assimilation or sample removal occurs.

Figures reproduce the manuscript's blue SoS identity and beta/E/I terminology,
with three rows and training/forecast columns. They use pointwise equal-tail
intervals, not the historical Gaussian mean ±1.96SD approximation. Synthetic
truth is preserved exactly; Toronto E/beta have no observed truth. Old synthetic
generating trajectories used the historical living denominator and are not
regenerated to match the final total-denominator inference convention.

Final WIS remains **pending Brandon's exact scorer**. The notebook supplies all
23 quantiles for the 11 stated central levels (10–90%,95%,98%) and the median;
it does not substitute old three-interval WIS. Saved RMSE/coverage/width are
explicitly labelled diagnostics and have no claim of exact scorer equivalence.

The original sampler/distributions are included verbatim for provenance, but
this notebook never imports or calls the sampler. Original worker RNG states may
duplicate; tempering completion and importance-weight ESS are not convergence or
independent-chain ESS. Posteriors are conditional on fixed data-selected dates,
ML-selected signs and configured supports. Toronto boundary mass is reported;
no support is silently changed. No comparator results are reconstructed here.

Case one and Toronto retain the original adaptive 5–10-attempt policy. The
case-two validation uses an explicit experimental 50-attempt policy: only the
existing adaptation flag is disabled in a run-local copy and the initial count
is 50. Each case's sampling configuration records that distinction and its exact
executed hash. The experimental derivative is retained for provenance, not used
by this notebook. Read validation status before interpreting the arrays; even
agreement across repeats does not establish full posterior convergence.

OpenAI Codex assisted with code and verification; no unverifiable backend model
identifier is asserted. Human scientific review remains required. See
`DATA_LICENSES.txt`, `LICENSE` and `CITATION.cff` for attribution and data terms.
