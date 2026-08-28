"""Plot marginal GPT-5.2 AIC profiles across theta and p_gen.

This is the AIC counterpart of ``plot_gpt52_profiles_nll.py``. For
each run and sweep cell it calculates AIC = 2*NLL + 2*k, averages each run's
AIC over the other sweep dimension, and then calculates the across-run mean
and normal-approximation 95% confidence interval.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPT52_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei" / "GPT52"
SITUATIONS = {
    "reliable_17aug": ("reliable", "sweep_17_8_2026", "17aug_smc_prob_dist_aic"),
    "reliable_25aug": ("reliable", "sweep_25_08_2026", "25aug_smc_prob_dist_aic"),
    "unreliable_17aug": (
        "unreliable",
        "sweep_17_8_2026",
        "17aug_smc_prob_dist_aic",
    ),
    "unreliable_25aug": (
        "unreliable",
        "sweep_25_08_2026",
        "25Aug_smc_prob_dist_aic",
    ),
}
CONDITION_FOLDERS = {
    "reliable": "reliable_keys",
    "unreliable": "unreliable_keys_observed",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_run_aic(run_scores: pd.DataFrame) -> pd.DataFrame:
    aic_helpers = load_module(
        "gpt52_aic_helpers", GPT52_ROOT / "plot_gpt52_heatmap_aic.py"
    )
    result = run_scores.copy()
    result["k"] = [
        aic_helpers.parameter_count(theta, p_gen)
        for theta, p_gen in zip(result["theta"], result["p_gen"])
    ]
    result["aic"] = 2.0 * result["total_nll"] + 2.0 * result["k"]
    return result


def summarise_marginal(run_scores: pd.DataFrame, dimension: str) -> pd.DataFrame:
    per_run = (
        run_scores.groupby([dimension, "run_id"], as_index=False)["aic"]
        .mean()
        .rename(columns={"aic": "run_mean_aic"})
    )
    rows = []
    for value, group in per_run.groupby(dimension, sort=False):
        values = group["run_mean_aic"].to_numpy(float)
        mean = float(values.mean())
        standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
        margin = 1.96 * standard_error
        rows.append(
            {
                "profile": dimension,
                "value": value,
                "mean_aic": mean,
                "standard_error": standard_error,
                "ci95_lower": mean - margin,
                "ci95_upper": mean + margin,
                "num_runs": len(values),
                "averaged_over": (
                    f"{run_scores['p_gen'].nunique()} p_gen values"
                    if dimension == "theta"
                    else f"{run_scores['theta'].nunique()} theta values"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if dimension == "theta":
        base = load_module(
            "gpt52_nll_profiles", GPT52_ROOT / "plot_gpt52_profiles_nll.py"
        )
        order = sorted(result["value"].astype(str), key=base.theta_sort_key)
        result["value"] = pd.Categorical(result["value"], order, ordered=True)
        result = result.sort_values("value")
        result["value"] = result["value"].astype(str)
    else:
        result["value"] = result["value"].astype(float)
        result = result.sort_values("value")
    return result.reset_index(drop=True)


def plot_profiles(
    theta_profile: pd.DataFrame,
    pgen_profile: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    panels = [
        (axes[0], pgen_profile, "p_gen", "Mean AIC by p_gen\n(averaged across theta)", "#55A868"),
        (axes[1], theta_profile, "theta", "Mean AIC by theta\n(averaged across p_gen)", "#4C72B0"),
    ]
    for axis, profile, xlabel, panel_title, color in panels:
        x = np.arange(len(profile))
        means = profile["mean_aic"].to_numpy(float)
        errors = 1.96 * profile["standard_error"].to_numpy(float)
        bars = axis.bar(x, means, color=color, alpha=0.78, width=0.76)
        axis.errorbar(x, means, yerr=errors, fmt="none", ecolor="#333333", capsize=4)
        axis.set_xticks(x, [str(value) for value in profile["value"]])
        if xlabel == "theta":
            axis.tick_params(axis="x", labelrotation=35)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean AIC per run")
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
    base = load_module(
        "gpt52_nll_profiles", GPT52_ROOT / "plot_gpt52_profiles_nll.py"
    )
    condition_folder = CONDITION_FOLDERS[condition]
    data_path = GPT52_ROOT / condition_folder / "experiments_gpt5.2.csv"
    sweep_path = PROJECT_ROOT / "training_results" / sweep_name
    output_dir = GPT52_ROOT / condition_folder / output_folder / "likelihood_surface"
    output_dir.mkdir(parents=True, exist_ok=True)

    nll_scores = base.calculate_run_scores(
        base.load_sequences(data_path), base.load_sweep(sweep_path)
    )
    run_scores = calculate_run_aic(nll_scores)
    theta_profile = summarise_marginal(run_scores, "theta")
    pgen_profile = summarise_marginal(run_scores, "p_gen")
    profiles = pd.concat([pgen_profile, theta_profile], ignore_index=True)
    profiles.to_csv(output_dir / "mean_aic_marginal_profiles.csv", index=False)
    run_scores.to_csv(output_dir / "run_aic_surface.csv", index=False)

    label = "Reliable keys" if condition == "reliable" else "Unreliable keys observed"
    plot_profiles(
        theta_profile,
        pgen_profile,
        output_dir / "mean_aic_marginal_profiles.png",
        f"{label}: {sweep_name} AIC profiles",
    )
    print(f"{condition}/{sweep_name}: saved AIC profiles to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--situation", choices=("all", *SITUATIONS), default="unreliable_17aug"
    )
    args = parser.parse_args()
    names = SITUATIONS if args.situation == "all" else [args.situation]
    for name in names:
        run_condition(*SITUATIONS[name])


if __name__ == "__main__":
    main()
