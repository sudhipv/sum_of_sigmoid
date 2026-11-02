# Provenance, attribution and scope

Human review remains pending. The verbatim sampler and distributions are from
repository commit `e412562ac57cff503f0f6cc4c2bd9d63da04785b` at
`inference/mcmc/tmcmc_mod/tmcmc_mod.py` and `pdfs.py`. They live unchanged in
`sumsigmoid/original/`. SAMPLER_REFERENCE.json records Git blobs and SHA256s.
Original headers credit Mukesh K. Ramancha and David Clarabut; existing repository
attribution, GPL-3.0 license and CITATION.cff are retained. No mathematical or
worker-initializer patch is applied. Explanations are outside those two files.

The versioned detector, legacy consolidation and optional turn consolidation derive
from `reproduction/2026-10-31_case2_three_procedure_timing/src/` and are copied
byte-for-byte. Joint likelihood/model and ML retain the verified arithmetic (with the
subsequently approved addition of D to the infection-flow denominator),
bounds, SLSQP settings and fixed NLL/160 divisor. ML amplitude signs are free;
E0/I0 remain jointly free. The existing four-start/12-attempt acceptance rule is
retained from `2026-10-31_case2_day156_accepted_completion` and the prior review
package; JSON null serialization of nonfinite failed diagnostics is I/O only.
The old `review_package/` and all research artifacts stay local and unchanged.

The subsequent approved final inference convention is total population
S+E+I+R+D, validated against the saved Toronto total-denominator preparation.
The original living model is preserved only as a historical verification fixture.
BIC, coordinate searches, provisional-day overrides and ML-centered restricted
priors are not defaults. SAVED_RESULTS.md labels main historical runs by their
actual engine, worker policy, denominator, support and configuration. Switching
the publication entrypoint does not relabel or resample any historical result.

Example data copy the first160 observations already retained in the repository:
`data/in/synthetic_case1_data.csv`, `data/in/synthetic_case2_data.dat`, and
`PHU_Data/30-Toronto.csv`. Only explicit time labels were added. Toronto's
2020-04-01 origin and 2020-09-07 inclusive cutoff follow the accepted date map.
No synthetic truth or held-out observations are example inputs. Review retained
data-source/citation metadata and redistribution obligations before release.

OpenAI Codex assisted with code preparation, documentation and verification.
An exact backend model/version is not verifiably recorded and is not asserted.
Assistance does not replace author review of science, code, licensing or data.
This is a scoped review branch, not a claim of manuscript acceptance, merge,
scientific validation or successful original-prior convergence.
