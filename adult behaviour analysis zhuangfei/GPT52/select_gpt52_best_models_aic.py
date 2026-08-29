"""Select the best-fitting SoC model for each GPT-5.2 run using AIC.

The score for model j is::

    AIC_j = 2 * NLL_j* + 2 * k_j

where NLL_j* is the model's negative log-likelihood and k_j is the number of
free parameters. Lower AIC is better.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei"
GPT52_ROOT = ANALYSIS_ROOT / "GPT52"

DEFAULT_DATA = GPT52_ROOT / "unreliable_keys_observed" / "experiments_gpt5.2.csv"
DEFAULT_SWEEP = PROJECT_ROOT / "training_results" / "sweep_17_8_2026"
DEFAULT_OUTPUT_DIR = GPT52_ROOT / "unreliable_keys_observed" / "17aug_smc_prob_dist_aic"

PARAMETER_COUNTS = {
    "SoC-Lesioned": 0,
    "SoC-Rel": 1,
    "SoC-Gen": 1,
    "SoC-Full": 2,
}

# The Lesioned, Rel, and Gen configurations are their AIC-surface region
# minima. SoC-Full deliberately uses the configuration fitted by the NLL
# analysis; AIC is still used to select the winning model for each run.
UNRELIABLE_AIC_MODEL_SPECS = {
    "SoC-Lesioned": ((19, 1), 0.1),
    "SoC-Rel": ((6, 1), 0.1),
    "SoC-Gen": ((19, 1), 0.4),
    "SoC-Full": ((1, 3), 0.8),  # NLL-fitted Full configuration
}

# Reliable-key AIC region minima. Its NLL-fitted Full configuration is also
# theta=(1,3), p_gen=0.7, so no numerical substitution is needed.
RELIABLE_AIC_MODEL_SPECS = {
    "SoC-Lesioned": ((19, 1), 0.1),
    "SoC-Rel": ((6, 1), 0.1),
    "SoC-Gen": ((19, 1), 0.4),
    "SoC-Full": ((1, 3), 0.7),
}


def load_nll_module():
    """Load the established data parsing and NLL functions without duplicating them."""
    path = ANALYSIS_ROOT / "calculate_selected_model_person_nll.py"
    spec = importlib.util.spec_from_file_location("selected_model_nll", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_condition_aic(
    data_path: Path,
    sweep_dir: Path,
    model_specs: dict | None = None,
) -> pd.DataFrame:
    module = load_nll_module()
    module.SWEEP_DIR = sweep_dir
    if model_specs is None:
        model_specs = UNRELIABLE_AIC_MODEL_SPECS

    people = module.load_people(data_path)
    models = {
        name: (theta, p_gen, module.load_model(theta, p_gen))
        for name, (theta, p_gen) in model_specs.items()
    }

    rows = []
    for person_id, observations in people:
        nll_scores = {
            name: module.nll(observations, table)
            for name, (_, _, table) in models.items()
        }
        aic_scores = {
            name: 2.0 * score + 2.0 * PARAMETER_COUNTS[name]
            for name, score in nll_scores.items()
        }
        best_model = min(aic_scores, key=aic_scores.get)
        theta, p_gen, _ = models[best_model]
        rows.append(
            {
                "Person_Id": person_id,
                "Model": best_model,
                "negative Loglikelihood": nll_scores[best_model],
                "k": PARAMETER_COUNTS[best_model],
                "AIC": aic_scores[best_model],
                "parameters": (
                    f"theta=({theta[0]},{theta[1]}), "
                    f"p_gen={p_gen:g}, true_prior={module.TRUE_PRIOR:g}"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best SoC model per GPT-5.2 run using AIC."
    )
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument(
        "--condition", choices=("all", "reliable", "unreliable"), default="all"
    )
    args = parser.parse_args()

    module = load_nll_module()
    conditions = {
        "reliable": {
            "data": GPT52_ROOT / "reliable_keys" / "experiments_gpt5.2.csv",
            "output": GPT52_ROOT / "reliable_keys" / "17aug_smc_prob_dist_aic",
            "specs": RELIABLE_AIC_MODEL_SPECS,
            "title": "Best-fitting model per GPT-5.2 reliable-key run (AIC; NLL Full)",
        },
        "unreliable": {
            "data": DEFAULT_DATA,
            "output": DEFAULT_OUTPUT_DIR,
            "specs": UNRELIABLE_AIC_MODEL_SPECS,
            "title": "Best-fitting model per GPT-5.2 unreliable-key run (AIC; NLL Full)",
        },
    }
    names = conditions if args.condition == "all" else [args.condition]
    for name in names:
        config = conditions[name]
        result = fit_condition_aic(config["data"], args.sweep, config["specs"])
        config["output"].mkdir(parents=True, exist_ok=True)
        output_csv = config["output"] / "gpt5.2_best_model_aic.csv"
        output_plot = config["output"] / "gpt5.2_best_model_aic_counts.png"
        result.to_csv(output_csv, index=False, encoding="utf-8-sig")
        module.plot_model_counts(result, output_plot, config["title"])
        print(f"{name}: saved {len(result)} runs to {output_csv}")
        print(f"{name}: saved model-count graph to {output_plot}")
        print(result["Model"].value_counts().to_string())


if __name__ == "__main__":
    main()
