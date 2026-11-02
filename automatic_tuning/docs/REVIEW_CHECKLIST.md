# Human review before merge or scientific use

- Review detector availability/thresholds, optional turn association and cutoff<=159 limit.
- Confirm input semantics, population, time origin, cutoff, rates and explicit E0/I0 guesses.
- Confirm the approved total S+E+I+R+D model and explicitly historical living checks.
- Review supports as assumptions, not confidence intervals; sign/location selection uncertainty is excluded.
- Review all ML attempts and the bounded-retry acceptance rule, including failed starts and zero-sign handling.
- Verify both original source hashes and imports; no initializer or mathematical patches should be present.
- Understand duplicated original worker RNG states and tempering-only completion; inspect external likelihood/training diagnostics.
- Keep historical results labeled with their actual sampler and scientific settings.
- Review license/citation/data attribution, test results and scoped staged diff before merging.
- The current denominator update is for local review only: no commit/push, teammate handoff or manuscript changes.
