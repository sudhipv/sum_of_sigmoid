# Packaging verification — 2026-11-01

- Both original files match the pinned Git objects byte-for-byte; actual module/function paths and hashes are verified in the entrypoint.
- 14 deterministic unittest methods passed. These retain the seven original acceptance scenarios and failed-record/input guards, plus original-source, Uniform MH, CLI seed, intercepted-interface and external-likelihood checks.
- Frozen primary lists reproduced: case one11, case two10, Toronto16; saved-parameter NLL, beta and training-compartment reference errors were zero in local artifact checks. Optional turn stays separate.
- Exactly one new preparation+ML smoke (case two) completed without manual seed edits:4starts,2reliable agreeing fits,2recorded status4failures. Best NLL861.2059506985091and full parameters exactly match the saved reference. No new sampling was run.
- Case-one14parameter,case-two13parameter andToronto19parameter optional entrypoints were inspected with saved/current ML:correct names,dates,signed0.5supports,E0/I0,actual callback likelihood and original imports. No workers or sampler call occurred.
- External checks were replayed on saved4000-particle corrected-worker Toronto training output only;RMSE matched within4.41e-13. This is not a new original-sampler result or forecast.
- Original sampler calls were intercepted at an external bridge in tests,not executed. No new forecasts,sensitivity tests,other scientific runs or manuscript edits occurred.

The original generic execution path has not been sampled as part of packaging.
Human review remains required; reaching tempering1 is not adequate exploration.
Matplotlib/pyparsing emitted dependency deprecation warnings during plot tests;
they did not change numerical checks. Use the pinned Python3.11 environment.
The separate total-denominator control does not change the living default.
