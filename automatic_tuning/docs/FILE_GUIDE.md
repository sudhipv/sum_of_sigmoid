# File guide

| Path | Purpose |
|---|---|
| `run.py` | CLI; preparation/ML default, sampling explicit; preserves original shared spawn-import seed behavior. |
| `examples/` | Complete case1/case2/Toronto configurations and small training-count inputs. |
| `sumsigmoid/config.py`, `io.py` | Required inputs, defaults, units, dates and training-only reading. |
| `sumsigmoid/frozen_detector.py` | Unchanged feature detection and availability. |
| `sumsigmoid/legacy_consolidation.py` | Unchanged primary consolidation. |
| `sumsigmoid/turn_consolidation.py` | Separately named optional alternative. |
| `sumsigmoid/candidates.py` | Thin procedure selection and complete audit outputs. |
| `sumsigmoid/model.py` | Shared total-population equations and normalized likelihood; used by ML/TMCMC/plots. |
| `sumsigmoid/optimization.py`, `acceptance.py` | Joint free-IC ML and agreed bounded retries. |
| `sumsigmoid/workflow.py` | Preparation stages, durable output files and terminal paths. |
| `sumsigmoid/original/tmcmc_mod.py`, `pdfs.py` | Exact original Git source; do not edit for documentation. |
| `sumsigmoid/sampler_provenance.py` | Check source hashes and actual imported paths. |
| `sumsigmoid/target.py` | Bind one immutable prepared likelihood to the optional sampler. |
| `sumsigmoid/optional_tmcmc.py` | Verified original interface; no added initializer, resuming or forecasts. |
| `sumsigmoid/training_checks.py` | External direct likelihood and training-fit diagnostics; no kernel changes. |
| `verify_examples.py`, `tests/`, `fixtures/` | Total-N saved reference, isolated historical living snapshot and deterministic checks; no sampling campaign. |
| `docs/SCIENTIFIC_METHOD.md` | Formulas, exact thresholds, supports and failure handling. |
| `docs/SAMPLER_LIMITATIONS.md` | Original worker/algorithm limitations and external checks. |
| `docs/SAVED_RESULTS.md`, `SAVED_RESULTS_PROVENANCE.csv` | Historical engine/configuration labels; not newly generated research results. |
| `SOURCE_PROVENANCE.json`, `SAMPLER_REFERENCE.json` | File lineage, hashes and pinned original commit. |
| `PUBLICATION_FILES.txt`, `PUBLICATION_MANIFEST.json` | Explicit clean publication list and hashes. |
| `requirements.txt`, `pyproject.toml`, `LICENSE`, `CITATION.cff` | Tested dependencies and retained attribution. |

Generated `outputs/`, previous `review_package/`, private audit directories,
large samples, credentials, caches and experiment runners are not publication
files. Original tracked historical artifacts remain tracked; none is deleted.
