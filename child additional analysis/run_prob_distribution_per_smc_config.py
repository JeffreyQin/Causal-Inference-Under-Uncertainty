"""Run 100 autonomous SMC simulations for every configuration in the NLL grid.

The 11 theta values and 9 generator probabilities reproduce all 99 cells in
the GPT-5 heatmap. Results are written after each configuration finishes, so
completed work is retained if the script is interrupted.
"""

from __future__ import annotations

import argparse
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
        "This script requires tqdm. Install it with: python -m pip install tqdm"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMC_DIR = PROJECT_ROOT / "smc"
OUTPUT_DIR = PROJECT_ROOT / "training_results" / "prob distribution per smc config"

# smc_soc.py imports these modules as top-level modules.
for import_dir in (PROJECT_ROOT, SMC_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from environment import Environment  # noqa: E402
from generator import Generator  # noqa: E402
from smc_soc import Engine  # noqa: E402


# Exact y- and x-axis values in the referenced NLL heatmap (11 x 9 = 99).
INIT_THETA_LIST = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (9, 1),
    (15, 1),
    (19, 1),
]
PROP_RANDOM_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

TRUE_PRIOR = 0.02
DEFAULT_NUM_RUNS = 100
DEFAULT_MAX_TRIALS = 70
NUM_PARTICLES = 30
OPENING_PROB = 1.0


class QuietLogger:
    def log(self, _message: str) -> None:
        """Suppress the engine's per-trial messages."""


def make_generator_config(prop_random: float) -> dict[str, Any]:
    return {"train": False, "prop_random": prop_random, "true_prior": TRUE_PRIOR}


def make_smc_config(init_theta: tuple[int, int]) -> dict[str, Any]:
    return {
        "num_particles": NUM_PARTICLES,
        "init_theta": init_theta,
        "ess_threshold": 0.5,
        "skill": True,
        "mode": "soc",
        "prior": "uniform",
    }


def float_tag(value: float) -> str:
    return str(value).replace(".", "p")


def setting_slug(init_theta: tuple[int, int], prop_random: float) -> str:
    return (
        f"theta_{init_theta[0]}_{init_theta[1]}"
        f"__gen_{float_tag(prop_random)}"
        f"__trueprior_{float_tag(TRUE_PRIOR)}"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_setting(
    init_theta: tuple[int, int],
    prop_random: float,
    num_runs: int,
    max_trials: int,
    model_position: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_counts: dict[int, Counter[int]] = defaultdict(Counter)
    run_rows: list[dict[str, Any]] = []
    logger = QuietLogger()

    runs = tqdm(
        range(1, num_runs + 1),
        desc=f"theta={init_theta}, gen={prop_random} runs",
        unit="run",
        position=model_position,
        leave=False,
    )
    for run_number in runs:
        env = Environment(opening_prob=OPENING_PROB, include_inspect=False)
        generator = Generator(make_generator_config(prop_random), env)
        engine = Engine(make_smc_config(init_theta), env, generator, logger)
        history = engine.run(max_trials=max_trials)

        opened_by_trial = {int(row["t"]): int(row["opened"]) for row in history}
        final_opened = opened_by_trial[max(opened_by_trial)]
        for trial_no in range(1, max_trials + 1):
            opened = opened_by_trial.get(trial_no, final_opened)
            trial_counts[trial_no][opened] += 1

        run_rows.append(
            {
                "theta": str(init_theta).replace(" ", ""),
                "gen": prop_random,
                "true_prior": TRUE_PRIOR,
                "run_number": run_number,
                "trials_completed": len(history) - 1,
                "boxes_opened": final_opened,
                "solved": final_opened >= 5,
                "final_theta": history[-1]["theta"],
            }
        )
        runs.set_postfix(solved=sum(row["solved"] for row in run_rows))

    probability_rows: list[dict[str, Any]] = []
    for trial_no in range(1, max_trials + 1):
        row: dict[str, Any] = {
            "theta": str(init_theta).replace(" ", ""),
            "gen": prop_random,
            "true_prior": TRUE_PRIOR,
            "trialNo": trial_no,
        }
        for opened in range(6):
            row[f"P(n={opened} boxes open)"] = trial_counts[trial_no][opened] / num_runs
        probability_rows.append(row)
    return probability_rows, run_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed; omitted by default for independent simulations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.max_trials < 1:
        raise SystemExit("--runs and --max-trials must both be positive integers")
    if args.seed is not None:
        random.seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_analysis": (
            "adult behaviour analysis zhuangfei/GPT5/"
            "unreliable_kes_partially_observed/17aug_smc_prob_dist_aic/"
            "gpt5_sweep_17_8_2026_aic_heatmap.png"
        ),
        "grid_source": "all theta x p_gen cells shown in the NLL heatmap",
        "theta_values": INIT_THETA_LIST,
        "p_gen_values": PROP_RANDOM_LIST,
        "true_prior": TRUE_PRIOR,
        "num_runs_per_configuration": args.runs,
        "max_trials": args.max_trials,
        "num_particles": NUM_PARTICLES,
        "opening_prob": OPENING_PROB,
        "random_seed": args.seed,
        "num_configurations": len(INIT_THETA_LIST) * len(PROP_RANDOM_LIST),
    }
    with (OUTPUT_DIR / "smc_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    all_probability_rows: list[dict[str, Any]] = []
    all_run_rows: list[dict[str, Any]] = []
    settings = list(product(INIT_THETA_LIST, PROP_RANDOM_LIST))
    configurations = tqdm(
        settings, desc="SMC configurations", unit="config", position=0
    )
    for init_theta, prop_random in configurations:
        configurations.set_postfix_str(f"theta={init_theta}, gen={prop_random}")
        probability_rows, run_rows = run_setting(
            init_theta=init_theta,
            prop_random=prop_random,
            num_runs=args.runs,
            max_trials=args.max_trials,
            model_position=1,
        )
        slug = setting_slug(init_theta, prop_random)
        write_csv(OUTPUT_DIR / f"{slug}.csv", probability_rows)
        write_csv(OUTPUT_DIR / f"{slug}__runs.csv", run_rows)
        all_probability_rows.extend(probability_rows)
        all_run_rows.extend(run_rows)

    write_csv(OUTPUT_DIR / "all_configs_probabilities.csv", all_probability_rows)
    write_csv(OUTPUT_DIR / "all_configs_runs.csv", all_run_rows)
    tqdm.write(f"Completed {len(settings)} configurations x {args.runs} runs.")
    tqdm.write(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
