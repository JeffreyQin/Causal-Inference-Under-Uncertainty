"""Fit reliable-adult learning data to the 17 August 2026 sweep outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADULT_CSV = (
    PROJECT_ROOT
    / "adult behaviour analysis zhuangfei"
    / "reliablekey_vs_unreliablekey"
    / "experiments_reliable_key.csv"
)
MODEL_RESULTS_DIR = (
    PROJECT_ROOT
    / "training_results"
    / "sweep_17_8_2026"
)
OUTPUT_CSV = (
    PROJECT_ROOT
    / "adult behaviour analysis zhuangfei"
    / "reliable_adults_sweep_17_8_2026_mean_nll.csv"
)
OUTPUT_HEATMAP = (
    PROJECT_ROOT
    / "adult behaviour analysis zhuangfei"
    / "reliable_adults_sweep_17_8_2026_heatmap.png"
)

TARGET_TRUE_PRIOR = 0.02
EPS = 1e-10
MAX_OPEN = 5


def load_adult_sequences(path: Path) -> dict[str, list[tuple[int, int]]]:
    """Return cumulative distinct doors opened after every learning attempt."""
    frame = pd.read_csv(path)
    sequences: dict[str, list[tuple[int, int]]] = {}

    for row_number, row in frame.iterrows():
        record = json.loads(row["data"])
        participant = str(row.get("id", row_number))
        opened: set[str] = set()
        sequence: list[tuple[int, int]] = []

        learning_attempts = [
            attempt
            for attempt in record.get("attempts", [])
            if attempt.get("phase") == "learning"
        ]
        for trial_number, attempt in enumerate(learning_attempts, start=1):
            if bool(attempt.get("correct")):
                opened.add(str(attempt.get("doorName")))
            sequence.append((trial_number, min(len(opened), MAX_OPEN)))

        sequences[participant] = sequence

    return sequences


def load_model_tables(
    path: Path,
) -> tuple[dict[tuple[str, float], pd.DataFrame], list[str], list[float], int]:
    # Each configuration in this sweep is stored in its own summary CSV.  The
    # ``__runs`` files contain individual simulation runs and are deliberately
    # excluded: the summary files contain the required probability distribution
    # for each trial.
    model_files = sorted(
        file
        for file in path.glob("theta_*__gen_*__trueprior_*.csv")
        if not file.name.endswith("__runs.csv")
    )
    if not model_files:
        raise FileNotFoundError(f"No sweep summary CSVs found in {path}")
    frame = pd.concat((pd.read_csv(file) for file in model_files), ignore_index=True)
    frame["true_prior"] = frame["true_prior"].astype(float)
    frame = frame[np.isclose(frame["true_prior"], TARGET_TRUE_PRIOR)].copy()
    frame["gen"] = frame["gen"].astype(float)
    frame["trialNo"] = frame["trialNo"].astype(int)

    probability_columns = [f"P(n={n} boxes open)" for n in range(MAX_OPEN + 1)]
    theta_values = sorted(
        frame["theta"].unique(),
        key=lambda value: tuple(
            int(part) for part in str(value).strip("()").split(",")
        ),
    )
    generation_values = sorted(frame["gen"].unique())
    maximum_trial = int(frame["trialNo"].max())
    tables: dict[tuple[str, float], pd.DataFrame] = {}

    for theta in theta_values:
        for generation in generation_values:
            subset = frame[
                (frame["theta"] == theta)
                & np.isclose(frame["gen"], generation)
            ].drop_duplicates(subset=["trialNo"], keep="first")
            tables[(theta, generation)] = subset.set_index("trialNo")[
                probability_columns
            ]

    return tables, theta_values, generation_values, maximum_trial


def calculate_mean_total_nll() -> pd.DataFrame:
    sequences = load_adult_sequences(ADULT_CSV)
    tables, theta_values, generation_values, maximum_trial = load_model_tables(
        MODEL_RESULTS_DIR
    )
    rows = []

    for theta in theta_values:
        for generation in generation_values:
            table = tables[(theta, generation)]
            participant_nlls = []
            num_observations = 0

            for sequence in sequences.values():
                log_likelihood = 0.0
                for trial_number, number_opened in sequence:
                    if trial_number > maximum_trial:
                        continue
                    probability = float(
                        table.loc[
                            trial_number,
                            f"P(n={number_opened} boxes open)",
                        ]
                    )
                    log_likelihood += np.log(max(probability, EPS))
                    num_observations += 1
                participant_nlls.append(-log_likelihood)

            rows.append(
                {
                    "theta": theta,
                    "gen": generation,
                    "true_prior": TARGET_TRUE_PRIOR,
                    "mean_total_nll_per_adult": float(np.mean(participant_nlls)),
                    "num_adults": len(participant_nlls),
                    "num_adult_trials": num_observations,
                    "mean_nll_per_trial": (
                        float(np.sum(participant_nlls) / num_observations)
                        if num_observations
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame) -> None:
    theta_values = list(dict.fromkeys(results["theta"]))
    generation_values = sorted(results["gen"].unique())
    grid = np.array(
        [
            [
                results[
                    (results["theta"] == theta)
                    & np.isclose(results["gen"], generation)
                ]["mean_total_nll_per_adult"].iloc[0]
                for generation in generation_values
            ]
            for theta in theta_values
        ]
    )

    figure, axis = plt.subplots(figsize=(10, 6.5))
    image = axis.imshow(grid, aspect="auto", cmap="YlOrRd", origin="upper")
    axis.set_xticks(range(len(generation_values)))
    axis.set_xticklabels([str(value) for value in generation_values])
    axis.set_yticks(range(len(theta_values)))
    axis.set_yticklabels(theta_values)
    axis.set_xlabel(r"generator $p_{\mathrm{gen}}$")
    axis.set_ylabel(r"reliability $\theta$")
    axis.set_title(
        "Reliable adults: Mean NLL by (theta, gen)\n"
        f"sweep_17_8_2026; true prior = {TARGET_TRUE_PRIOR:.2f}",
        fontweight="bold",
    )

    midpoint = float(np.nanmean(grid))
    for row in range(len(theta_values)):
        for column in range(len(generation_values)):
            value = grid[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if value > midpoint else "black",
            )

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean total NLL per adult (lower is better)")
    figure.tight_layout()
    figure.savefig(OUTPUT_HEATMAP, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    results = calculate_mean_total_nll()
    results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    plot_heatmap(results)
    best = results.loc[results["mean_total_nll_per_adult"].idxmin()]
    print(f"Saved results: {OUTPUT_CSV}")
    print(f"Saved heatmap: {OUTPUT_HEATMAP}")
    print(
        "Best setting: "
        f"theta={best['theta']}, gen={best['gen']}, "
        f"mean total NLL/adult={best['mean_total_nll_per_adult']:.6f}"
    )
    print(
        f"Adults={int(best['num_adults'])}, "
        f"adult trials={int(best['num_adult_trials'])}"
    )


if __name__ == "__main__":
    main()
