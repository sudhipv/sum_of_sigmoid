"""Post-run training diagnostics outside all original sampler calculations."""
import csv
import time
import numpy as np
from .io import save


def check_saved_samples(output, particles, raw_engine_values, priors, ml, target):
    """Recalculate final likelihoods and compare the ensemble training fit.

    Particles/weights/priors are never changed. Near-ML counts are descriptive,
    not convergence criteria. No forecasts or automatic reruns follow.
    """
    begun = time.perf_counter()
    low, high = np.asarray(priors["lower"]), np.asarray(priors["upper"])
    if not np.isfinite(particles).all() or np.any(particles < low) or np.any(particles > high):
        raise RuntimeError("Nonfinite/out-of-support output; samples retained, no retry")
    constant = float(np.sum(-np.log(high-low)))
    direct = np.array([-float(target.logtarget(point)) for point in particles])
    decoded = -(np.asarray(raw_engine_values)+constant)
    if not np.isfinite(direct).all() or np.max(abs(direct-decoded)) > 1e-7:
        raise RuntimeError("Direct likelihood check failed; raw output retained")
    mean = np.zeros(len(target.MODEL.grid))
    for point in particles:
        _, states = target.MODEL.evaluate(point, keep_states=True)
        if not np.isfinite(states).all() or np.min(states) < -1e-8:
            raise RuntimeError("Invalid training compartments; no sample removal or retry")
        mean += states[:, 2] / len(particles)
    _, ml_states = target.MODEL.evaluate(ml["parameters"], keep_states=True)
    every = round(1/target.MODEL.settings["dt"])
    rmse = float(np.sqrt(np.mean((mean[::every]-target.MODEL.y)**2)))
    ml_rmse = float(np.sqrt(np.mean((ml_states[::every, 2]-target.MODEL.y)**2)))
    with (output / "direct_likelihood_checks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["particle", "direct_NLL", "offset_corrected_NLL", "absolute_difference"])
        writer.writerows((j, a, b, abs(a-b)) for j, (a, b) in enumerate(zip(direct, decoded)))
    np.savez_compressed(output / "training_summary.npz", time=target.MODEL.grid,
                        sampled_mean_I=mean, ML_I=ml_states[:, 2], observations=target.MODEL.y)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.scatter(np.arange(len(target.MODEL.y)), target.MODEL.y, s=10, c="#333333", label="Training observations")
    ax.plot(target.MODEL.grid, ml_states[:, 2], "--", label="Joint ML fit")
    ax.plot(target.MODEL.grid, mean, label="TMCMC sampled mean; requires review")
    ax.set(xlabel="Model day", ylabel="Infectious counts (persons)",
           title="Training diagnostic; tempering completion is not convergence")
    ax.legend(); fig.savefig(output / "training_fit_check.pdf"); plt.close(fig)
    result = dict(best_NLL=float(direct.min()), median_NLL=float(np.median(direct)),
                  reference_ML_NLL=ml["NLL"], sampled_mean_training_RMSE=rmse, ML_training_RMSE=ml_rmse,
                  best_NLL_gap_to_ML=float(direct.min()-ml["NLL"]),
                  particles_within1NLLofML=int(np.sum(direct <= ml["NLL"]+1)),
                  particles_within10NLLofML=int(np.sum(direct <= ml["NLL"]+10)),
                  max_direct_likelihood_error=float(np.max(abs(direct-decoded))),
                  Uniform_logprior_constant=constant, offset_is_reporting_only=True,
                  diagnostic_seconds=time.perf_counter()-begun, training_review_required=True,
                  no_convergence_claim=True, no_forecasts=True)
    save(output / "training_checks.json", result)
    return result
