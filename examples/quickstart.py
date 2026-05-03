"""Minimal smoke test for the sum_of_sigmoid repository.

This script checks that the core dependencies import, the bundled data files can be
loaded, and the TMCMC helper module is importable. It does not run MCMC.
"""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHU_PATH = ROOT / "PHU_Data"
DATA_PATH = ROOT / "data"

sys.path.insert(0, str(ROOT))

from inference.mcmc.tmcmc_mod import pdfs


def main():
    toronto_cases = np.genfromtxt(PHU_PATH / "30-Toronto.csv", delimiter=",")
    population_by_phu = np.genfromtxt(PHU_PATH / "population_by_phu.csv", delimiter=",")
    synthetic_case1 = np.genfromtxt(DATA_PATH / "toronto_synthetic_data_noise10.csv", delimiter=",")
    synthetic_case2 = np.loadtxt(DATA_PATH / "synthetic_case2_data.dat")

    prior = pdfs.Uniform(lower=0.0, upper=0.2)
    prior_sample = prior.generate_rns(1)[0]

    print("Quickstart smoke test passed.")
    print(f"Toronto data points: {toronto_cases.shape[0]}")
    print(f"Toronto population: {population_by_phu[29, 1]:.0f}")
    print(f"Synthetic case 1 data points: {synthetic_case1.shape[0]}")
    print(f"Synthetic case 2 data points: {synthetic_case2.shape[0]}")
    print(f"Example Uniform(0, 0.2) prior sample: {prior_sample:.6f}")


if __name__ == "__main__":
    main()
