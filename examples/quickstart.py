"""Run the synthetic case 1 MCMC and plotting workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHU_PATH = ROOT / "PHU_Data"
DATA_IN = ROOT / "data" / "in"
MPLCONFIGDIR = ROOT / "examples" / ".matplotlib"

MCMC_SCRIPT = ROOT / "inference" / "mcmc" / "singlephu_mult_mcmc_synth_1.py"
PLOTTING_NOTEBOOK = ROOT / "inference" / "plotting" / "plot_synth_case1.ipynb"
EXECUTED_NOTEBOOK_DIR = ROOT / "examples" / "notebooks"


def run_command(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_notebook(path: Path, env: dict[str, str]) -> None:
    if shutil.which("jupyter") is None:
        raise RuntimeError(
            "Running the plotting notebook requires Jupyter/nbconvert. "
            "Install it in the active environment, then rerun this script."
        )

    EXECUTED_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            "--output",
            "plot_synth_case1_executed.ipynb",
            "--output-dir",
            str(EXECUTED_NOTEBOOK_DIR),
            "--ExecutePreprocessor.timeout=-1",
        ],
        env,
        "Run synthetic case 1 plotting notebook",
    )


def smoke_check() -> None:
    toronto_cases = np.genfromtxt(PHU_PATH / "30-Toronto.csv", delimiter=",")
    population_by_phu = np.genfromtxt(PHU_PATH / "population_by_phu.csv", delimiter=",")
    synthetic_case1 = np.genfromtxt(DATA_IN / "synthetic_case1_data.csv", delimiter=",")

    print("Input data check passed.")
    print(f"Toronto data points: {toronto_cases.shape[0]}")
    print(f"Toronto population: {population_by_phu[29, 1]:.0f}")
    print(f"Synthetic case 1 data points: {synthetic_case1.shape[0]}")


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    return env


def main() -> None:
    env = build_env()
    smoke_check()

    run_command([sys.executable, str(MCMC_SCRIPT)], env, "Run synthetic case 1 MCMC script")
    run_notebook(PLOTTING_NOTEBOOK, env)

    print("\nQuickstart complete.")
    print(f"MCMC outputs: {ROOT / 'data' / 'out'}")
    print(f"MCMC diagnostic figures: {ROOT / 'figs' / 'mcmc' / 'synth_case1'}")
    print(f"Prediction figures: {ROOT / 'figs' / 'predictions' / 'test'}")
    print(f"Executed plotting notebook: {EXECUTED_NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
