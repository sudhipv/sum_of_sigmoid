# Original sampler: preserved behavior and external checks

The original tmcmc_mod.py/distribution utilities (file name `pdfs.py`) are not edited.
Their hashes, commit and actual import paths are checked before optional use.
No bug fix described here has been applied inside them.

- The CLI retains a shared configured seed on spawn import; the original Pool
  has no initializer. Initial worker RNG states can duplicate. Fork can inherit
  a shared state too. This is preserved behavior, not independent-chain sampling.
- Native CPU count determines original Pool size. A requested count mismatch
  stops before sampling rather than monkeypatching multiprocessing or the engine.
- Mutations start at 5 and adapt with a cap of 10; at high acceptance the general
  rule can choose fewer than 5. Completion only checks tempering exponent 1.
- The original prior-ratio implementation is not general to nonuniform priors.
  This package exposes only the verified constant-density Uniform supports.
- The likelihood-only callback causes a constant Uniform offset in raw recorded
  likelihood/evidence. Reporting adds it back; relative Uniform MH/weights are
  unaffected. Raw records are retained unmodified.
- Existing nonfinite edge handling and covariance behavior are inherited. No
  covariance conditioning, altered acceptance or extra mutation steps are added.

External checks verify original source hashes, parameter order, signed supports,
hard beta admissibility and ML-callback equivalence before sampling. Afterwards,
all final likelihoods are evaluated directly and compared with corrected cached
values; a training-only mean trajectory, RMSE and ML comparison are saved. Near-ML
counts are descriptive, not a universal convergence threshold. These checks do
not alter particles/weights, filter samples, change priors or rerun anything.
Poor or unresolved output must be reviewed. Importance-weight ESS is not chain
ESS; saved population files are not continuous independent mutation chains.

The one controlled original-source Toronto total-denominator run reached the
known good training region, but this does not prove that the corrected-worker
initializer caused the earlier failure or validate duplicated worker streams.
It also does not justify changing the living-denominator publication default.
