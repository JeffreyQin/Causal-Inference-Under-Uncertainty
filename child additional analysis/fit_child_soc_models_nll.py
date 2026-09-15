"""Fit every child to all 99 autonomous SoC configurations using NLL."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        """Run normally when tqdm is unavailable."""
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHILD_TRIALS = PROJECT_ROOT / "smc" / "child_data" / "trials.csv"
MODEL_DIR = PROJECT_ROOT / "training_results" / "prob distribution per smc config"
OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_FILE = MODEL_DIR / "all_configs_probabilities.csv"

EPSILON = 1e-10
MAX_OPEN = 5
PROBABILITY_COLUMNS = [f"P(n={n} boxes open)" for n in range(MAX_OPEN + 1)]


def normalise_theta(value: object) -> str:
    return str(value).replace(" ", "")


def theta_sort_key(value: str) -> tuple[int, int]:
    first, second = value.strip("()").split(",")
    return int(first), int(second)


def soc_variant(theta: str, p_gen: float) -> str:
    """Name the four regions used in the NLL heatmap analysis."""
    reliability_lesioned = normalise_theta(theta) == "(19,1)"
    generator_lesioned = math.isclose(p_gen, 0.1)
    if reliability_lesioned and generator_lesioned:
        return "SoC-Lesioned"
    if generator_lesioned:
        return "SoC-Rel"
    if reliability_lesioned:
        return "SoC-Gen"
    return "SoC-Full"


def load_models(path: Path) -> dict[tuple[str, float], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run run_prob_distribution_per_smc_config.py first."
        )
    frame = pd.read_csv(path)
    required = {"theta", "gen", "trialNo", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame["theta"] = frame["theta"].map(normalise_theta)
    models: dict[tuple[str, float], pd.DataFrame] = {}
    for (theta, p_gen), group in frame.groupby(["theta", "gen"], sort=False):
        table = group.drop_duplicates("trialNo").set_index("trialNo")
        models[(str(theta), float(p_gen))] = table[PROBABILITY_COLUMNS]
    if len(models) != 99:
        raise ValueError(f"Expected 99 configurations but found {len(models)} in {path}")
    return models


def load_child_observations(path: Path) -> dict[str, list[tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing child trial data: {path}")
    trials = pd.read_csv(path)
    required = {"ID", "Order", "Box", "Worked"}
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    children: dict[str, list[tuple[int, int]]] = {}
    trials = trials.sort_values(["ID", "Order"], kind="stable")
    for child_id, group in trials.groupby("ID", sort=False):
        opened_boxes: set[str] = set()
        observations: list[tuple[int, int]] = []
        for trial_number, (_, trial) in enumerate(group.iterrows(), start=1):
            if float(trial["Worked"]) == 1.0 and pd.notna(trial["Box"]):
                opened_boxes.add(str(trial["Box"]))
            observations.append((trial_number, min(len(opened_boxes), MAX_OPEN)))
        children[str(child_id)] = observations
    return children


def calculate_nll(
    observations: list[tuple[int, int]], model: pd.DataFrame
) -> tuple[float, int]:
    total_nll = 0.0
    trials_used = 0
    for trial_number, opened in observations:
        if trial_number not in model.index:
            continue
        probability = float(model.at[trial_number, f"P(n={opened} boxes open)"])
        total_nll -= math.log(max(probability, EPSILON))
        trials_used += 1
    return total_nll, trials_used


def fit_children(
    children: dict[str, list[tuple[int, int]]],
    models: dict[tuple[str, float], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for child_id, observations in tqdm(children.items(), desc="Fitting children", unit="child"):
        child_scores: list[dict[str, object]] = []
        for (theta, p_gen), model in models.items():
            total_nll, trials_used = calculate_nll(observations, model)
            score = {
                "Child_ID": child_id,
                "theta": theta,
                "p_gen": p_gen,
                "true_prior": 0.02,
                "SoC_variant": soc_variant(theta, p_gen),
                "NLL": total_nll,
                "trials_used": trials_used,
                "total_child_trials": len(observations),
                "trials_beyond_model_horizon": len(observations) - trials_used,
            }
            child_scores.append(score)
            score_rows.append(score)

        best = min(child_scores, key=lambda row: float(row["NLL"]))
        ties = sum(
            math.isclose(float(row["NLL"]), float(best["NLL"]), abs_tol=1e-12)
            for row in child_scores
        )
        best_rows.append({**best, "number_of_tied_best_configs": ties})

    return pd.DataFrame(best_rows), pd.DataFrame(score_rows)


def make_summaries(best: pd.DataFrame) -> None:
    config_summary = (
        best.groupby(["SoC_variant", "theta", "p_gen"], as_index=False)
        .agg(
            child_count=("Child_ID", "size"),
            mean_best_nll=("NLL", "mean"),
            median_best_nll=("NLL", "median"),
        )
    )
    config_summary["percent_of_children"] = 100 * config_summary["child_count"] / len(best)
    config_summary = config_summary.sort_values(
        ["child_count", "mean_best_nll"], ascending=[False, True]
    )
    config_summary.to_csv(OUTPUT_DIR / "child_nll_config_summary.csv", index=False)

    variant_summary = best["SoC_variant"].value_counts().rename_axis("SoC_variant").reset_index(name="child_count")
    variant_summary["percent_of_children"] = 100 * variant_summary["child_count"] / len(best)
    variant_summary.to_csv(OUTPUT_DIR / "child_nll_variant_summary.csv", index=False)

    theta_summary = best["theta"].value_counts().rename_axis("theta").reset_index(name="child_count")
    theta_summary["percent_of_children"] = 100 * theta_summary["child_count"] / len(best)
    theta_summary["_sort"] = theta_summary["theta"].map(theta_sort_key)
    theta_summary.sort_values("_sort").drop(columns="_sort").to_csv(
        OUTPUT_DIR / "child_nll_theta_summary.csv", index=False
    )

    pgen_summary = best["p_gen"].value_counts().sort_index().rename_axis("p_gen").reset_index(name="child_count")
    pgen_summary["percent_of_children"] = 100 * pgen_summary["child_count"] / len(best)
    pgen_summary.to_csv(OUTPUT_DIR / "child_nll_p_gen_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-trials", type=Path, default=CHILD_TRIALS)
    parser.add_argument("--models", type=Path, default=MODEL_FILE)
    parser.add_argument(
        "--save-all-scores",
        action="store_true",
        help="Also save the 100 x 99 table of every child/configuration NLL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_models(args.models)
    children = load_child_observations(args.child_trials)
    best, scores = fit_children(children, models)
    best.to_csv(OUTPUT_DIR / "child_best_fitted_soc_nll.csv", index=False)
    if args.save_all_scores:
        scores.to_csv(OUTPUT_DIR / "child_all_soc_config_nll.csv", index=False)
    make_summaries(best)

    print(f"Fitted {len(children)} children against {len(models)} configurations.")
    print("\nBest-fitting SoC variant counts:")
    print(best["SoC_variant"].value_counts().to_string())
    print("\nMost frequent exact configurations:")
    print(best.value_counts(["theta", "p_gen"]).head(10).to_string())
    print(f"\nResults saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
