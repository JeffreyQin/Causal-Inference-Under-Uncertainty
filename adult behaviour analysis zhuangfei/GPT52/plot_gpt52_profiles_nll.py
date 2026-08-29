"""Plot marginal GPT-5.2 NLL profiles across theta and p_gen.

The theta profile averages each run's NLL across all p_gen values, while the
p_gen profile averages each run's NLL across all theta values. The final mean
and normal-approximation 95% CI are then calculated across GPT runs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPT52_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei" / "GPT52"
CONDITION_FOLDERS = {
    "reliable": "reliable_keys",
    "unreliable": "unreliable_keys_observed",
}
SITUATIONS = {
    "reliable_17aug": ("reliable", "sweep_17_8_2026", "17aug_smc_prob_dist"),
    "reliable_17aug_redo": (
        "reliable",
        "sweep_17_8_2026",
        "17aug_smc_prob_dist_redo",
    ),
    "reliable_25aug": ("reliable", "sweep_25_08_2026", "25aug_smc_prob_dist"),
    "unreliable_17aug": (
        "unreliable",
        "sweep_17_8_2026",
        "17aug_smc_prob_dist",
    ),
    "unreliable_25aug": (
        "unreliable",
        "sweep_25_08_2026",
        "25Aug_smc_prob_dist",
    ),
}
TRUE_PRIOR = 0.02
MAX_OPEN = 5
EPS = 1e-10


def theta_sort_key(value: str) -> tuple[int, int]:
    return tuple(int(part.strip()) for part in value.strip("()").split(","))


def load_sequences(path: Path) -> dict[str, list[tuple[int, int]]]:
    sequences = {}
    for row_number, row in pd.read_csv(path).iterrows():
        record = json.loads(row["data"])
        run_id = str(row["id"]) if pd.notna(row.get("id")) else str(row_number)
        opened: set[str] = set()
        observations = []
        attempts = [
            attempt
            for attempt in record.get("attempts", [])
            if attempt.get("phase") == "learning"
        ]
        for trial_number, attempt in enumerate(attempts, start=1):
            if bool(attempt.get("correct")):
                opened.add(str(attempt.get("doorName")))
            observations.append((trial_number, min(len(opened), MAX_OPEN)))
        sequences[run_id] = observations
    return sequences


def load_sweep(path: Path) -> pd.DataFrame:
    files = sorted(
        file
        for file in path.glob("theta_*__gen_*__trueprior_*.csv")
        if not file.name.endswith("__runs.csv")
    )
    if not files:
        raise FileNotFoundError(f"No sweep summary CSVs found in {path}")
    frame = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    frame = frame[np.isclose(frame["true_prior"].astype(float), TRUE_PRIOR)].copy()
    frame["theta"] = frame["theta"].astype(str)
    frame["gen"] = frame["gen"].astype(float)
    frame["trialNo"] = frame["trialNo"].astype(int)
    return frame


def calculate_run_scores(sequences: dict, sweep: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [f"P(n={n} boxes open)" for n in range(MAX_OPEN + 1)]
    rows = []
    for (theta, p_gen), group in sweep.groupby(["theta", "gen"], sort=False):
        table = group.drop_duplicates("trialNo").set_index("trialNo")
        for run_id, observations in sequences.items():
            total_nll = 0.0
            used = 0
            for trial_number, number_opened in observations:
                if trial_number not in table.index:
                    continue
                probability = float(
                    table.at[trial_number, probability_columns[number_opened]]
                )
                total_nll -= math.log(max(probability, EPS))
                used += 1
            rows.append(
                {
                    "run_id": run_id,
                    "theta": theta,
                    "p_gen": float(p_gen),
                    "total_nll": total_nll,
                    "num_trials": used,
                }
            )
    return pd.DataFrame(rows)


def summarise_marginal(run_scores: pd.DataFrame, dimension: str) -> pd.DataFrame:
    per_run = (
        run_scores.groupby([dimension, "run_id"], as_index=False)["total_nll"]
        .mean()
        .rename(columns={"total_nll": "run_mean_nll"})
    )
    rows = []
    for value, group in per_run.groupby(dimension, sort=False):
        values = group["run_mean_nll"].to_numpy(float)
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
        margin = 1.96 * standard_error
        rows.append(
            {
                "profile": dimension,
                "value": value,
                "mean_total_nll": mean,
                "standard_error": standard_error,
                "ci95_lower": mean - margin,
                "ci95_upper": mean + margin,
                "num_runs": len(values),
                "averaged_over": "9 p_gen values" if dimension == "theta" else "11 theta values",
            }
        )
    result = pd.DataFrame(rows)
    if dimension == "theta":
        order = sorted(result["value"].astype(str), key=theta_sort_key)
        result["value"] = pd.Categorical(result["value"], order, ordered=True)
        result = result.sort_values("value")
        result["value"] = result["value"].astype(str)
    else:
        result["value"] = result["value"].astype(float)
        result = result.sort_values("value")
    return result.reset_index(drop=True)


def plot_marginal_profiles(
    theta_profile: pd.DataFrame,
    pgen_profile: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    panels = [
        (axes[0], pgen_profile, "p_gen", "Mean NLL by p_gen\n(averaged across theta)", "#55A868"),
        (axes[1], theta_profile, "theta", "Mean NLL by theta\n(averaged across p_gen)", "#4C72B0"),
    ]
    for axis, profile, xlabel, panel_title, color in panels:
        x = np.arange(len(profile))
        means = profile["mean_total_nll"].to_numpy(float)
        errors = 1.96 * profile["standard_error"].to_numpy(float)
        labels = [str(value) for value in profile["value"]]
        bars = axis.bar(x, means, color=color, alpha=0.78, width=0.76)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#333333", capsize=4)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35 if xlabel == "theta" else 0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean total negative log-likelihood")
        axis.set_title(panel_title, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=3, fontsize=8)
        axis.set_ylim(0, max(means + errors) * 1.16)
    figure.suptitle(title + " (95% CI; lower is better)", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_condition(condition: str, sweep_name: str, output_folder: str) -> None:
    condition_folder = CONDITION_FOLDERS[condition]
    data_path = GPT52_ROOT / condition_folder / "experiments_gpt5.2.csv"
    sweep_path = PROJECT_ROOT / "training_results" / sweep_name
    output_dir = GPT52_ROOT / condition_folder / output_folder / "likelihood_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_scores = calculate_run_scores(load_sequences(data_path), load_sweep(sweep_path))
    theta_profile = summarise_marginal(run_scores, "theta")
    pgen_profile = summarise_marginal(run_scores, "p_gen")
    profiles = pd.concat([pgen_profile, theta_profile], ignore_index=True)
    profiles.to_csv(output_dir / "mean_nll_marginal_profiles.csv", index=False)
    label = "Reliable keys" if condition == "reliable" else "Unreliable keys observed"
    plot_marginal_profiles(
        theta_profile,
        pgen_profile,
        output_dir / "mean_nll_marginal_profiles.png",
        f"{label}: {sweep_name}",
    )
    print(f"{condition}/{sweep_name}: saved marginal profiles to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--situation", choices=("all", *SITUATIONS), default="all"
    )
    args = parser.parse_args()
    names = SITUATIONS if args.situation == "all" else [args.situation]
    for name in names:
        run_condition(*SITUATIONS[name])


if __name__ == "__main__":
    main()
