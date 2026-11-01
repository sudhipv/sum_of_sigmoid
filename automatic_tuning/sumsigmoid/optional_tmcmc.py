"""Explicit original-sampler entrypoint; preparation never samples by default.

Inspection and training checks are outside the two verbatim original files.
No Pool initializer, covariance fix, altered MH rule or retry is applied.
"""
from pathlib import Path
import multiprocessing as mp
import os
import time
import numpy as np
from .io import read, save, sha
from .sampler_provenance import load_original


def inspect_prepared(prepared):
    """Load the real callback and original modules, but draw no particles."""
    folder = Path(prepared).resolve()
    if read(folder / "status.json")["status"] != "accepted":
        raise RuntimeError("ML preparation unresolved; TMCMC is blocked")
    config = read(folder / "configuration.json")
    priors = read(folder / "TMCMC_priors.json")
    ml = read(folder / "ML_result.json")
    if not ml["accepted"]:
        raise RuntimeError("An accepted joint ML fit is required")
    os.environ["SUMSIGMOID_PREPARED"] = str(folder)
    from . import target
    if target.FOLDER.resolve() != folder:
        raise RuntimeError("Use a fresh CLI process; a different target is already bound")
    engine, distributions, source = load_original()
    point = np.asarray(ml["parameters"], dtype=float)
    names, locations = target.MODEL.names, target.MODEL.locations
    if ml["parameter_names"] != names or priors["parameter_names"] != names:
        raise RuntimeError("Parameter ordering mismatch")
    if ml["locations"] != locations or priors["locations"] != locations:
        raise RuntimeError("Fixed-location mismatch")
    if len(point) != len(locations) + 3 or not np.isfinite(point).all():
        raise RuntimeError("Invalid joint parameter vector")
    signs = np.sign(point[1:-2]).astype(int)
    if np.any(signs == 0):
        raise RuntimeError("Zero ML amplitudes have no approved signed-prior rule")
    expected_lower = np.r_[0., np.where(signs < 0, -.5, 0.), 0., 0.]
    expected_upper = np.r_[.5, np.where(signs < 0, 0., .5), [.001 * config["population"]] * 2]
    lower, upper = np.asarray(priors["lower"]), np.asarray(priors["upper"])
    if not np.array_equal(lower, expected_lower) or not np.array_equal(upper, expected_upper):
        raise RuntimeError("Prior supports differ from the approved signed Uniform policy")
    if np.any(point < lower) or np.any(point > upper):
        raise RuntimeError("Accepted ML lies outside the proposed signed supports")
    actual_nll = -float(target.logtarget(point))
    if not np.isfinite(actual_nll) or abs(actual_nll - ml["NLL"]) > 1e-7:
        raise RuntimeError("Actual TMCMC callback does not reproduce the accepted ML NLL")
    pars = [distributions.Uniform(lower=a, upper=b) for a, b in zip(lower, upper)]
    settings = config["tmcmc"]
    plan = dict(
        particles=settings["particles"], seed=settings["seed"],
        requested_workers=settings["workers"], actual_native_worker_count=mp.cpu_count(),
        worker_count_matches=settings["workers"] == mp.cpu_count(),
        start_method=mp.get_start_method(), pool_initializer=None,
        locations=locations, parameter_names=names, lower=lower.tolist(), upper=upper.tolist(),
        prior_classes=[type(p).__module__ + "." + type(p).__name__ for p in pars],
        original_source=source, callback="sumsigmoid.target.logtarget",
        likelihood_convention="normalized likelihood only plus hard beta admissibility",
        denominator="living S+E+I+R", verified_ML_NLL=actual_nll,
        initialization="fresh configured base Uniforms; no ML/posterior/checkpoint injection",
        worker_rng_policy="original behavior; shared configured seed on CLI spawn import; no distinct-stream initializer",
        no_forecasts=True, training_review_required=True,
        importance_ESS_is_not_independent_chain_ESS=True,
    )
    return folder, config, priors, ml, target, engine, pars, plan


def run(prepared, dry_run=False):
    """Inspect, or explicitly sample once with the unmodified original engine."""
    folder, config, priors, ml, target, engine, pars, plan = inspect_prepared(prepared)
    if dry_run:
        return plan
    if not plan["worker_count_matches"]:
        raise RuntimeError(
            "Original sampler uses multiprocessing.cpu_count(): requested workers="
            f"{plan['requested_workers']}, native count={plan['actual_native_worker_count']}. "
            "Acknowledge the native count in configuration; no override is applied."
        )
    output = folder / "tmcmc"
    if output.exists():
        raise FileExistsError("TMCMC output exists; no resume, overwrite or repeat")
    output.mkdir()
    save(output / "configuration.json", plan)
    # Preserve the reference runner's shared seed policy; this is NOT a Pool
    # initializer. CLI spawn imports repeat the seed; fork may inherit a shared
    # state. Neither promises independent worker random streams.
    os.environ["SUMSIGMOID_ORIGINAL_TMCMC_SEED"] = str(plan["seed"])
    np.random.seed(plan["seed"])
    save(output / "sampling_started.json", dict(
        plan=plan, source_hashes=plan["original_source"],
        model_sha256=sha(Path(__file__).with_name("model.py")),
        training_input_sha256=sha(folder / "training_observations.csv"),
    ))
    save(folder / "status.json", {**read(folder / "status.json"), "TMCMC_started": True})
    started = time.perf_counter()
    try:
        samples, chain, trace, _ = _sample_once(
            engine, plan["particles"], pars, target.logtarget, str(output / "sampler_status.txt"))
    except Exception as exc:
        save(output / "sampling_status.json", dict(status="sampler_failed",error=repr(exc),
            elapsed_seconds=time.perf_counter()-started,no_automatic_retry=True))
        raise
    seconds = time.perf_counter() - started
    arrays = {}
    for j, stage in enumerate(trace):
        arrays.update({f"particles_{j}": stage[0], f"engine_value_{j}": stage[1],
                       f"weights_{j}": stage[2], f"ESS_{j}": np.asarray(stage[3]),
                       f"next_temperature_{j}": np.asarray(stage[4]),
                       f"resampled_{j}": np.asarray(stage[5])})
    np.savez_compressed(output / "original_stage_records.npz", **arrays)
    np.savez_compressed(output / "stage_populations.npz", chain=chain)
    np.savez_compressed(output / "particles.npz", particles=samples, weights=np.full(len(samples), 1 / len(samples)))
    np.savez_compressed(output / "initial_particles.npz", particles=trace[0][0])
    # Authentication only: replay locally, never inject these expected values.
    rng = np.random.RandomState(plan["seed"])
    expected = np.column_stack([(u-l)*rng.rand(plan["particles"])+l
                                for l, u in zip(priors["lower"], priors["upper"])])
    if not np.array_equal(expected, trace[0][0]):
        raise RuntimeError("Fresh prior provenance mismatch; saved output retained")
    from .training_checks import check_saved_samples
    try:
        checks = check_saved_samples(output, samples, trace[-1][1], priors, ml, target)
    except Exception as exc:
        save(output / "sampling_status.json", dict(status="failed_external_checks",error=repr(exc),
            sampling_seconds=seconds,raw_output_retained=True,no_automatic_retry=True))
        raise
    _, _, final_sources = load_original()
    save(output / "sampling_complete.json", dict(
        sampling_seconds=seconds, stages=len(trace)-1, original_source=final_sources,
        raw_engine_values_preserved=True, fresh_prior_exact_match=True,
        final_tempering_exponent=float(trace[-2][4]), external_checks=checks,
        status="completed_requires_training_review",
    ))
    return dict(status="completed_requires_training_review", output=str(output),
                sampling_seconds=seconds, checks=checks,
                outputs=dict(particles=str(output/"particles.npz"),training_figure=str(output/"training_fit_check.pdf"),
                    direct_likelihood_checks=str(output/"direct_likelihood_checks.csv"),sampler_configuration=str(output/"configuration.json")),
                message="Tempering completed; review the training diagnostics before accepting inference.")


def _sample_once(engine, count, priors, callback, status_file):
    """Unmodified five-argument interface; tests intercept this external seam."""
    return engine.run_tmcmc(count, priors, callback, "multiprocessing", status_file)
