# Automatic location selection: scientific method

Human scientific review is pending. This document describes the packaged code;
it does not replace the manuscript or approve a new scientific method.

## Candidate proposals

`frozen_detector.py`, `legacy_consolidation.py` and `turn_consolidation.py` are
byte-for-byte copies of the verified study modules. Their rules were specified
before these comparisons and are held constant; no thresholds changed.

1. Let `s[t]` be the mean of raw counts from `t-3` to `t+3`, clipped at the
   available edges. The five-day relative change is
   `(s[t]-s[t-5])/max(s[t-5],1)`. Rising/falling categories require at least
   +15% / at most -15%; otherwise the category is within +/-15%.
2. A category change requires three **daily, overlapping** checks, not three
   disjoint five-day blocks. Sustained entry into the middle category proposes
   a flattening onset. Sign-only changes of the five-day relative change,
   similarly confirmed for three daily checks, identify peaks/troughs; the
   exact extremum search interval is retained in each flag.
3. In declining episodes, inspect cumulative decline from the running peak.
   Once it is at least 20%, compare five-day slopes before/after each day.
   Both slopes must be negative. Relative absolute-rate change of at least
   +30% is steepening and at most -30% is slowing. At least three successive
   eligible daily rows form a run. The largest absolute contrast is its
   representative; the first wins a tie.
4. Provisional flags remain in the raw output. A flag is eligible only when its
   recorded `available_by_raw_day <= cutoff`. For example, a closing slope-run
   check needs centered evidence through `last_run_day+9`. Edge-truncated
   smoothing can propose a feature before that full evidence is available;
   eligibility does not waive this requirement.
5. Identical-date groups retain every flag. Legacy consolidation associates a
   flattening start inside/on the day immediately following a slowing run,
   subject to the existing confirmed-reversal, sustained-steepening and pending
   evidence vetoes. It retains the slowing representative. Full intervals,
   availability requirements and decisions remain in the audit.

These are heuristic transmission-change proposals from the observed curve,
not proof that each flag is a distinct physical change or an exhaustive list of
possible sigmoid dates. No minimum-spacing or grid-filling rule is added.

`turn_optional` applies the separately named turn-anchored procedure over legacy
groups. Its source preserves exact compatibility brackets, unique-anchor
requirements, vetoes and the `h+10` finalization guard. It is not the default.

Default preparation does **not** use BIC, coordinate refinement, manual dates,
future observations, provisional day156 overrides, or ML-centered restricted
priors. The historical experiment folders remain untouched outside the package.

## Model, units and likelihood

Beta is `A0 + sum(Aj/(1+exp((location_j-t)/width)))`, with width1 day.
The population argument is in persons. The approved final model uses
`N(t)=S+E+I+R+D` in the infection-flow denominator at every Euler step. The
compartment equations conserve this total at its initial value N0; including D
does not change any other population-based bound or initial condition. The
verified Toronto reference permits at most1e-6person conservation error over
the training grid. The shared `model.py` serves ML, TMCMC and training plots.
Initial states are
`S0=population-E0-I0`, `R0=D0=0`; E0/I0 are inferred parameters.

The explicit Euler grid has step0.1 day and extends through the last training
day plus0.9, matching the original implementation. Likelihood observations
enter at integer days including day0. At each day the observed infectious count
has Gaussian mean `I` and standard deviation `0.1*I`. The full density includes
`-log(sqrt(2*pi)*0.1*I)` and the squared standardized residual. Nonpositive
predicted infectious counts or nonfinite likelihoods are inadmissible.

Raw observed counts enter the likelihood. The seven-day smoother is confined
to detection. Beta nonnegativity is checked on the whole training Euler grid
with tolerance1e-12; no post-sampling removal or beta clipping is introduced.

## Joint ML and bounded retries

Every ML attempt jointly fits `K+3` parameters: beta0, K two-sided amplitudes,
E0 and I0. No amplitude sign is imposed during ML.

- SLSQP uses `ftol=1e-9`, finite-difference `eps=1e-8`, `maxiter=300`.
- The solver objective is full normalized NLL divided by the original fixed
  factor160, including for shorter prefixes. All acceptance comparisons use
  the **full unscaled NLL**. Saved-reference checks cover160 observations;
  shorter prefixes are supported inputs, not independently validated studies.
- Scaling is 0.1 for rate parameters and 1000 for count parameters.
- First start: constant beta0=0.1, all amplitudes0, and the **user's explicit**
  E0/I0 guesses. Those guesses are not fixed during fitting.
- Later starts: Gaussian perturbations around that same initial vector, with
  standard deviations0.03 for beta0,0.06 for amplitudes,150 persons for E0/I0,
  clipped to ML supports. The original initialization mechanism is preserved.
  One continuing local RandomState stream is used; duplicate vectors and
  infeasible seeds are rejected, with at most10,000 draws per required start.
  No unapproved feasibility repair is invented if this fails.
- Complete each batch of4 attempts before assessment. Failed optimizer calls
  consume attempts. At most12 attempts are permitted.
- Each valid fit must pass original success, finite NLL, bounds and beta checks,
  plus the original per-attempt reliability checks: no deterioration from its
  start by more than1e-5, and no missing a better observed objective by more
  than1e-4. Intermediate warnings/failures are recorded.
- Let `best` be the lowest NLL among valid results. Accept only if at least two
  distinct starting vectors reach NLL <= best+1e-4 and **every** valid result in
  that competitive group has the same nonzero amplitude signs. Poorer fits
  outside the competitive group do not veto acceptance.
- An unconfirmed new best cannot be replaced by an older, worse reproduced
  result. Exact-zero amplitude signs remain unresolved. At exhaustion save the
  best found fit/diagnostics, label unresolved and block signed priors/TMCMC.

This is an operational acceptance rule, not a proof of global optimality. The
generic adaptation replaces the former hardcoded synthetic430-person seed with
required configuration values. The seven original policy checks are retained.

## Optional inference and limitations

The two original files are unmodified Git objects from the pinned commit.
No worker-stream correction or mathematical fixes are on the execution path.
CPU count and start method are native, recorded and checked against requested
worker count; no hidden override occurs. CLI spawn imports share the sampling
seed, so worker RNG states can duplicate. See SAMPLER_LIMITATIONS.md.

The original Uniform recording offset is decoded outside the engine only:
true logL = raw engine value + sum(-log(upper-lower)). Original raw records are
kept. All final likelihoods and the ensemble training fit are checked externally.
Case one and Toronto retain the original adaptive 5–10 move controls. The
accepted case-two example uses fixed 50 attempted moves per chain per stage.
An isolated output-local source copy changes only the existing
`Adap_calc_Nsteps` flag to `no` and receives both existing caps as 50. The two
original sampler files remain byte-for-byte unchanged. Tempering completion is
the original stopping condition; external checks never declare convergence or
silently launch a repeat.

The approved denominator update changes only the infection-flow population
sum; all rates, widths, noise, likelihood indexing/normalization, Euler step,
bounds, beta support, detection, optimizer and acceptance settings are preserved.
Old living-population checks use only `fixtures/historical_living_model.py`,
an isolated historical snapshot outside the inference path. Its NLLs are not
reference values for the final model. Prepared-result metadata prevents mixing
these conventions in optional inference.

No new TMCMC, forecast or sensitivity experiment was performed for packaging.
Historical results retain their actual original/corrected sampler provenance.
