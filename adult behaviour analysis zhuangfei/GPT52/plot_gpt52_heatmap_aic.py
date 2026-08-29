"""Convert a GPT-5.2 sweep's mean NLL surface into an AIC heatmap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei"
GPT_ROOT = ANALYSIS_ROOT / "GPT52"

ADULT_CSV = GPT_ROOT / "unreliable_keys_observed" / "experiments_gpt5.2.csv"
MODEL_RESULTS_DIR = PROJECT_ROOT / "training_results" / "sweep_17_8_2026"
OUTPUT_DIR = GPT_ROOT / "unreliable_keys_observed" / "17aug_smc_prob_dist_aic"
CONDITION_LABEL = "Unreliable-key GPT-5.2 runs"

# Boundaries used by calculate_selected_model_person_nll.py to define the
# lesioned models in the unreliable-key condition.
LESIONED_THETA = "(19,1)"
LESIONED_GEN = 0.1


def load_nll_heatmap_module():
    path = ANALYSIS_ROOT / "negative_log_likelihood_heatmaps" / "fit_unreliable_adults.py"
    spec = importlib.util.spec_from_file_location("unreliable_nll_heatmap", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_count(theta: str, generation: float) -> int:
    """Return k for the nested SoC model represented by one sweep cell."""
    reliability_active = str(theta).replace(" ", "") != LESIONED_THETA
    generator_active = not np.isclose(float(generation), LESIONED_GEN)
    return int(reliability_active) + int(generator_active)


def calculate_mean_aic() -> pd.DataFrame:
    module = load_nll_heatmap_module()
    module.ADULT_CSV = ADULT_CSV
    module.MODEL_RESULTS_DIR = MODEL_RESULTS_DIR
    results = module.calculate_mean_total_nll().copy()
    results["k"] = [
        parameter_count(theta, generation)
        for theta, generation in zip(results["theta"], results["gen"])
    ]
    results["mean_aic_per_run"] = (
        2.0 * results["mean_total_nll_per_adult"] + 2.0 * results["k"]
    )
    return results


def plot_aic_heatmap(results: pd.DataFrame, path: Path) -> None:
    theta_values = list(dict.fromkeys(results["theta"]))
    generation_values = sorted(results["gen"].unique())
    grid = np.array(
        [
            [
                results[
                    (results["theta"] == theta)
                    & np.isclose(results["gen"], generation)
                ]["mean_aic_per_run"].iloc[0]
                for generation in generation_values
            ]
            for theta in theta_values
        ]
    )

    figure, axis = plt.subplots(figsize=(10, 6.5))
    image = axis.imshow(grid, aspect="auto", cmap="YlOrRd", origin="upper")
    axis.set_xticks(range(len(generation_values)), [str(v) for v in generation_values])
    axis.set_yticks(range(len(theta_values)), theta_values)
    axis.set_xlabel(r"generator $p_{\mathrm{gen}}$")
    axis.set_ylabel(r"reliability $\theta$")
    axis.set_title(
        f"{CONDITION_LABEL}: Mean AIC by (theta, gen)\n"
        f"{MODEL_RESULTS_DIR.name}; AIC = 2 NLL + 2k",
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
    colorbar.set_label("Mean AIC per run (lower is better)")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f"gpt5.2_{MODEL_RESULTS_DIR.name}_mean_aic.csv"
    output_heatmap = OUTPUT_DIR / f"gpt5.2_{MODEL_RESULTS_DIR.name}_aic_heatmap.png"
    results = calculate_mean_aic()
    results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    plot_aic_heatmap(results, output_heatmap)
    best = results.loc[results["mean_aic_per_run"].idxmin()]
    print(f"Saved AIC results: {output_csv}")
    print(f"Saved AIC heatmap: {output_heatmap}")
    print(
        f"Best setting: theta={best['theta']}, gen={best['gen']}, "
        f"k={int(best['k'])}, mean AIC/run={best['mean_aic_per_run']:.6f}"
    )


if __name__ == "__main__":
    main()
