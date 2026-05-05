"""Run the synthetic case 1 example workflow.

This wrapper runs the existing case 1 files in order:

1. MLE notebook: inference/mle/1phu_synthetic_MLE.ipynb
2. MCMC script: inference/mcmc/singlephu_mult_mcmc_synth_1.py
3. Plotting notebook: inference/plotting/plot_synth_case1.ipynb

Generated figures are written under examples/figs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHU_PATH = ROOT / "PHU_Data"
DATA_IN = ROOT / "data" / "in"
EXAMPLE_DIR = ROOT / "examples"
EXAMPLE_FIGS = EXAMPLE_DIR / "figs"
EXAMPLE_DATA = EXAMPLE_DIR / "data" / "synth_case1"
EXAMPLE_NOTEBOOKS = EXAMPLE_DIR / "notebooks"
EXAMPLE_WORKFLOW = EXAMPLE_DIR / "workflow"
MPLCONFIGDIR = EXAMPLE_DIR / ".matplotlib"


def run_command(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_notebook(path: Path, output_name: str, env: dict[str, str], label: str) -> None:
    if shutil.which("jupyter") is None:
        raise RuntimeError(
            "Running the quickstart notebooks requires Jupyter/nbconvert. "
            "Install it in the active environment, then rerun this script."
        )

    EXAMPLE_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(path),
            "--output",
            output_name,
            "--output-dir",
            str(EXAMPLE_NOTEBOOKS),
            "--ExecutePreprocessor.timeout=-1",
        ],
        env,
        label,
    )


def patch_notebook(source: Path, target: Path) -> None:
    nb = json.loads(source.read_text())
    replacements = {
        'data_out = ROOT / "data" / "out"\n': f'data_out = Path(r"{EXAMPLE_DATA}")\n',
        'figpath = ROOT / "figs"\n': f'figpath = Path(r"{EXAMPLE_FIGS}")\n',
        'figpath = ROOT / "figs" / "predictions" / "test"\n': f'figpath = Path(r"{EXAMPLE_FIGS}") / "predictions" / "test"\n',
    }
    for cell in nb.get("cells", []):
        source_lines = cell.get("source")
        if not isinstance(source_lines, list):
            continue
        cell["source"] = [replacements.get(line, line) for line in source_lines]
    target.write_text(json.dumps(nb, indent=1) + "\n")


def patch_mcmc_script(source: Path, target: Path) -> None:
    text = source.read_text()
    text = text.replace(
        "figpath = ROOT / 'figs' / 'mcmc' / 'synth_case1'",
        f"figpath = Path(r'{EXAMPLE_FIGS}') / 'mcmc' / 'synth_case1'",
    )
    text = text.replace(
        "data_out = ROOT / 'data' / 'out'",
        f"data_out = Path(r'{EXAMPLE_DATA}')",
    )
    target.write_text(text)


def prepare_example_workflow() -> dict[str, Path]:
    EXAMPLE_WORKFLOW.mkdir(parents=True, exist_ok=True)
    paths = {
        "mle": EXAMPLE_WORKFLOW / "1phu_synthetic_MLE.ipynb",
        "mcmc": EXAMPLE_WORKFLOW / "singlephu_mult_mcmc_synth_1.py",
        "plot": EXAMPLE_WORKFLOW / "plot_synth_case1.ipynb",
    }
    patch_notebook(ROOT / "inference" / "mle" / "1phu_synthetic_MLE.ipynb", paths["mle"])
    patch_mcmc_script(ROOT / "inference" / "mcmc" / "singlephu_mult_mcmc_synth_1.py", paths["mcmc"])
    patch_notebook(ROOT / "inference" / "plotting" / "plot_synth_case1.ipynb", paths["plot"])
    return paths


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
    env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mle", action="store_true", help="Skip the MLE notebook.")
    parser.add_argument("--skip-mcmc", action="store_true", help="Skip the MCMC script.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip the plotting notebook.")
    args = parser.parse_args()

    EXAMPLE_FIGS.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DATA.mkdir(parents=True, exist_ok=True)
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

    env = build_env()
    smoke_check()
    workflow_paths = prepare_example_workflow()

    if not args.skip_mle:
        run_notebook(
            workflow_paths["mle"],
            "1phu_synthetic_MLE_executed.ipynb",
            env,
            "Run synthetic case 1 MLE notebook",
        )

    if not args.skip_mcmc:
        run_command(
            [sys.executable, str(workflow_paths["mcmc"])],
            env,
            "Run synthetic case 1 MCMC script",
        )

    if not args.skip_plot:
        run_notebook(
            workflow_paths["plot"],
            "plot_synth_case1_executed.ipynb",
            env,
            "Run synthetic case 1 plotting notebook",
        )

    print("\nQuickstart complete.")
    print(f"Figures: {EXAMPLE_FIGS}")
    print(f"MCMC data: {EXAMPLE_DATA}")
    print(f"Executed notebooks: {EXAMPLE_NOTEBOOKS}")
    print(f"Workflow copies: {EXAMPLE_WORKFLOW}")


if __name__ == "__main__":
    main()
