"""Run autonomous SoC-SMC simulations over the reliable-adult theta x p_gen grid.

Unlike ``method3_human_data_leading_fitting.py``, this script does not load or
replay participant actions.  At every trial ``smc_soc.Engine`` selects its own
action, and the environment supplies the outcome.

Outputs (under ``training_results/sweep_25_08_2026``):
  * one ``theta_*__gen_*__trueprior_*.csv`` file per configuration, containing
    empirical P(n boxes open) by trial. Each file is written immediately after
    that configuration completes.
  * a matching ``__runs.csv`` file with one row per simulation run.
  * ``sweep_config.json``: exact reproducibility configuration.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "This sweep displays progress with tqdm. Install it first with: "
        "python -m pip install tqdm"
    ) from exc

# ``smc_soc.py`` imports ``environment`` as a top-level module, so make the
# smc directory importable regardless of whether this file is run from the
# project root or from inside smc/fitting_methods.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMC_DIR = PROJECT_ROOT / "smc"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SMC_DIR) not in sys.path:
    sys.path.insert(0, str(SMC_DIR))

from environment import Environment  # noqa: E402
from generator import Generator  # noqa: E402
from smc_soc import Engine  # noqa: E402


# Heatmap axes: y = reliability theta, x = generator p_gen.
INIT_THETA_LIST = [
    (1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (4, 1),
    (5, 1), (6, 1), (9, 1), (15, 1), (19, 1),
]
PROP_RANDOM_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TRUE_PRIOR = 0.02                 # Fixed: deliberately not swept.

# Simulation controls.  Raise NUM_RUNS for smoother empirical probabilities.
NUM_RUNS = 100
MAX_TRIALS = 70
NUM_PARTICLES = 30
OPENING_PROB = 1.0
RANDOM_SEED: int | None = None    # Set an integer to make a run reproducible.

OUTPUT_DIR = PROJECT_ROOT / "training_results" / "sweep_25_08_2026"


class Logger:
    def log(self, _message: str) -> None:
        """Suppress per-trial engine logging during a large sweep."""


def make_generator_config(prop_random: float) -> dict[str, Any]:
    """Configuration consumed by ``Generator`` for one p_gen value."""
    return {
        "train": False,
        "prop_random": prop_random,
        "true_prior": TRUE_PRIOR,
    }


def make_smc_config(init_theta: tuple[int, int]) -> dict[str, Any]:
    """Configuration consumed by the autonomous ``smc_soc.Engine``."""
    return {
        "num_particles": NUM_PARTICLES,
        "init_theta": init_theta,
        "ess_threshold": 0.5,
        "skill": True,
        "mode": "soc",
        "prior": "uniform",
    }


def float_tag(value: float) -> str:
    """Format a decimal safely for filenames, e.g. 0.02 -> 0p02."""
    return str(value).replace(".", "p")


def setting_filename(init_theta: tuple[int, int], prop_random: float) -> str:
    """Return a self-describing filename stem for one parameter setting."""
    return (
        f"theta_{init_theta[0]}_{init_theta[1]}"
        f"__gen_{float_tag(prop_random)}"
        f"__trueprior_{float_tag(TRUE_PRIOR)}"
    )


def run_setting(init_theta: tuple[int, int], prop_random: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one grid cell and return its trial counts and per-run summaries."""
    trial_counts: dict[int, Counter[int]] = defaultdict(Counter)
    run_rows: list[dict[str, Any]] = []
    logger = Logger()

    for run_number in range(1, NUM_RUNS + 1):
        env = Environment(opening_prob=OPENING_PROB, include_inspect=False)
        generator = Generator(make_generator_config(prop_random), env)
        engine = Engine(make_smc_config(init_theta), env, generator, logger)
        history = engine.run(max_trials=MAX_TRIALS)

        # Engine records t=0 and every completed trial.  Pad short solved runs
        # with their final number-opened value, producing a complete trial grid.
        opened_by_trial = {int(row["t"]): int(row["opened"]) for row in history}
        final_opened = opened_by_trial[max(opened_by_trial)]
        for trial_no in range(1, MAX_TRIALS + 1):
            trial_counts[trial_no][opened_by_trial.get(trial_no, final_opened)] += 1

        run_rows.append({
            "theta": str(init_theta).replace(" ", ""),
            "gen": prop_random,
            "true_prior": TRUE_PRIOR,
            "run_number": run_number,
            "trials_completed": len(history) - 1,
            "boxes_opened": final_opened,
            "solved": final_opened >= 5,
            "final_theta": history[-1]["theta"],
        })

    trial_rows: list[dict[str, Any]] = []
    for trial_no in range(1, MAX_TRIALS + 1):
        counts = trial_counts[trial_no]
        row: dict[str, Any] = {
            "theta": str(init_theta).replace(" ", ""),
            "gen": prop_random,
            "true_prior": TRUE_PRIOR,
            "trialNo": trial_no,
        }
        for opened in range(6):
            row[f"P(n={opened} boxes open)"] = counts[opened] / NUM_RUNS
        trial_rows.append(row)

    return trial_rows, run_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_settings = len(INIT_THETA_LIST) * len(PROP_RANDOM_LIST)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "theta_values": INIT_THETA_LIST,
        "p_gen_values": PROP_RANDOM_LIST,
        "true_prior": TRUE_PRIOR,
        "num_runs_per_configuration": NUM_RUNS,
        "max_trials": MAX_TRIALS,
        "num_particles": NUM_PARTICLES,
        "opening_prob": OPENING_PROB,
        "random_seed": RANDOM_SEED,
        "num_configurations": total_settings,
    }
    with (OUTPUT_DIR / "sweep_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    settings = product(INIT_THETA_LIST, PROP_RANDOM_LIST)
    for init_theta, prop_random in tqdm(
        settings,
        total=total_settings,
        desc="Autonomous SoC sweep",
        unit="setting",
    ):
        trial_rows, run_rows = run_setting(init_theta, prop_random)
        filename = setting_filename(init_theta, prop_random)
        write_csv(OUTPUT_DIR / f"{filename}.csv", trial_rows)
        write_csv(OUTPUT_DIR / f"{filename}__runs.csv", run_rows)

    print(f"Completed {total_settings} settings. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
