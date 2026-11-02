# Saved-result provenance

These are existing saved results, not new calculations or reruns. Switching
the publication entrypoint does not change their sampler labels. The table
summarizes the main primary baselines and directly relevant controls; historical
timing/BIC/refinement/restricted-prior variants remain separate experiments.
Full source/configuration paths and actual engine SHA256s are recorded in
`../SAVED_RESULTS_PROVENANCE.csv` (repository-relative paths refer to preserved
local research history, not files bundled in this publication folder).

| Saved result | Actual sampler | Transmission denominator | K / particles | Worker policy |
|---|---|---|---:|---|
| Toronto historical accepted reproduction | original repository sampler | S+E+I+R | 5 / 2000 | old shared worker behavior;no distinct-stream initializer |
| Case one frozen legacy11 | corrected-worker variant | S+E+I+R | 11 / 4000 | SeedSequence-derived distinct worker streams |
| Case two frozen legacy10 | corrected-worker variant | S+E+I+R | 10 / 4000 | SeedSequence-derived distinct worker streams |
| Toronto matched manual5 | corrected-worker variant | S+E+I+R | 5 / 4000 | SeedSequence-derived distinct worker streams |
| Toronto optional-turn11 | corrected-worker variant | S+E+I+R | 11 / 4000 | SeedSequence-derived distinct worker streams |
| Toronto primary automatic16 | corrected-worker variant | S+E+I+R | 16 / 4000 | SeedSequence-derived distinct worker streams |
| Case two exploratory +156 | corrected-worker variant | S+E+I+R | 11 / 4000 | SeedSequence-derived distinct worker streams |
| Toronto total-denominator active run | corrected-worker variant | S+E+I+R+D | 16 / 4000 | SeedSequence-derived distinct worker streams |
| Toronto total-denominator original control | exact original sampler | S+E+I+R+D | 16 / 4000 | original duplicated spawn-import states |

All rows above use signed amplitude supports of magnitude0.5 and inferred E0/I0.
The historical2000run has no explicit hard-beta target constraint; later matched
rows do. Older case-one magnitude0.2 results are distinct and are not relabeled
as these magnitude0.5 matched-procedure results. Archived original4000Toronto
particles have incomplete generating provenance; they are not substituted for
the documented historical reproduction or matched baselines.

The new publication path uses the exact original sampler with the subsequently
approved total-population denominator S+E+I+R+D. At the initial publication commit
`7612237`, no sampling had yet been run through that package. Thus none
of the existing corrected-worker results is evidence of an original-sampler
publication run. The original total-denominator control is a separate scientific
configuration; its success does not establish that worker initialization caused
another run's failure. The denominator update follows explicit author instruction.
Historical living results above keep their original labels and parameters.

## Subsequent final-baseline validation

Final original-sampler case-one and case-two runs were subsequently completed.
Case one passed the descriptive training checks; the original default case-two
run failed badly and remains preserved as failed sampled output. Later case-two
fixed-50 experiments are explicitly experimental control derivatives, not
relabeled unmodified-original runs. The final reference uses seed106, with
exactly two new validation repeats at107/108. A compatible completed Toronto
total-denominator original-sampler run was independently verified for reuse.

See [current case-specific settings and limitations](VALIDATION_HANDOFF.md) and
`../fixtures/final_validation_controls.json`. Passing training/domain checks or
repeat agreement is not full posterior convergence. The separate collaborator
archive includes validation status; failed populations stay in diagnostic history.
