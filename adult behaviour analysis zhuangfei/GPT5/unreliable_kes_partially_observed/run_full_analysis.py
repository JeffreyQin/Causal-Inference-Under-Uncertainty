"""Run the complete NLL/AIC pipeline for partially observed GPT-5 runs.

Order is intentional: the full NLL surface is calculated first.  The lowest
mean-NLL configuration inside each of the four SoC regions is then used for
both per-run NLL and per-run AIC model selection.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei"
GPT52_ROOT = ANALYSIS_ROOT / "GPT52"
HERE = Path(__file__).resolve().parent
DATA = HERE / "experiments_gpt5.csv"
SWEEP = PROJECT_ROOT / "training_results" / "sweep_17_8_2026"
NLL_DIR = HERE / "17aug_smc_prob_dist"
AIC_DIR = HERE / "17aug_smc_prob_dist_aic"
GRID_DIR = HERE / "test_grid" / "partial_worst_rel_gen_5x5"
LESIONED_THETA = "(19,1)"
LESIONED_GEN = 0.1


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalise_theta(value: object) -> str:
    return str(value).replace(" ", "")


def theta_tuple(value: object) -> tuple[int, int]:
    return tuple(int(part) for part in normalise_theta(value).strip("()").split(","))


def region_masks(surface: pd.DataFrame) -> dict[str, pd.Series]:
    theta_lesioned = surface["theta"].map(normalise_theta).eq(LESIONED_THETA)
    gen_lesioned = np.isclose(surface["gen"], LESIONED_GEN)
    return {
        "SoC-Lesioned": theta_lesioned & gen_lesioned,
        "SoC-Rel": ~theta_lesioned & gen_lesioned,
        "SoC-Gen": theta_lesioned & ~gen_lesioned,
        "SoC-Full": ~theta_lesioned & ~gen_lesioned,
    }


def select_region_nll_minima(surface: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    specs = {}
    rows = []
    for model, mask in region_masks(surface).items():
        region = surface[mask]
        if region.empty:
            raise ValueError(f"NLL surface has no cells for {model}")
        best = region.loc[region["mean_total_nll_per_adult"].idxmin()]
        specs[model] = (theta_tuple(best["theta"]), float(best["gen"]))
        rows.append(
            {
                "Model": model,
                "theta": normalise_theta(best["theta"]),
                "p_gen": float(best["gen"]),
                "true_prior": float(best["true_prior"]),
                "mean_total_nll_per_run": float(best["mean_total_nll_per_adult"]),
            }
        )
    return specs, pd.DataFrame(rows)


def run_nll_surface() -> pd.DataFrame:
    module = load_module(
        "gpt5_partial_nll_surface",
        ANALYSIS_ROOT / "negative_log_likelihood_heatmaps" / "fit_unreliable_adults.py",
    )
    NLL_DIR.mkdir(parents=True, exist_ok=True)
    module.ADULT_CSV = DATA
    module.MODEL_RESULTS_DIR = SWEEP
    module.OUTPUT_CSV = NLL_DIR / "gpt5_sweep_17_8_2026_mean_nll.csv"
    module.OUTPUT_HEATMAP = NLL_DIR / "gpt5_sweep_17_8_2026_nll_heatmap.png"
    surface = module.calculate_mean_total_nll()
    surface.to_csv(module.OUTPUT_CSV, index=False, encoding="utf-8-sig")
    module.plot_heatmap(surface)
    return surface


def run_nll_profiles() -> pd.DataFrame:
    module = load_module("gpt5_nll_profiles", GPT52_ROOT / "plot_gpt52_profiles_nll.py")
    output = NLL_DIR / "likelihood_surface"
    output.mkdir(parents=True, exist_ok=True)
    scores = module.calculate_run_scores(module.load_sequences(DATA), module.load_sweep(SWEEP))
    theta = module.summarise_marginal(scores, "theta")
    pgen = module.summarise_marginal(scores, "p_gen")
    pd.concat([pgen, theta], ignore_index=True).to_csv(
        output / "mean_nll_marginal_profiles.csv", index=False
    )
    module.plot_marginal_profiles(
        theta,
        pgen,
        output / "mean_nll_marginal_profiles.png",
        "GPT-5 unreliable keys partially observed: sweep_17_8_2026",
    )
    return scores


def run_nll_selection(specs: dict) -> pd.DataFrame:
    module = load_module("gpt5_nll_selection", ANALYSIS_ROOT / "calculate_selected_model_person_nll.py")
    module.SWEEP_DIR = SWEEP
    config = {
        "data": DATA,
        "models": specs,
        "output": NLL_DIR / "gpt5_best_model_nll.csv",
        "plot": NLL_DIR / "gpt5_best_model_counts.png",
        "title": "Best-fitting model per partially observed GPT-5 run (NLL)",
    }
    result = module.fit_condition(config)
    result.to_csv(config["output"], index=False, encoding="utf-8-sig")
    module.plot_model_counts(result, config["plot"], config["title"])
    return result


def run_aic_surface() -> pd.DataFrame:
    module = load_module("gpt5_aic_surface", GPT52_ROOT / "plot_gpt52_heatmap_aic.py")
    AIC_DIR.mkdir(parents=True, exist_ok=True)
    module.ADULT_CSV = DATA
    module.MODEL_RESULTS_DIR = SWEEP
    module.CONDITION_LABEL = "Partially observed unreliable-key GPT-5 runs"
    surface = module.calculate_mean_aic()
    surface.to_csv(AIC_DIR / "gpt5_sweep_17_8_2026_mean_aic.csv", index=False)
    module.plot_aic_heatmap(surface, AIC_DIR / "gpt5_sweep_17_8_2026_aic_heatmap.png")
    return surface


def run_aic_profiles(nll_scores: pd.DataFrame) -> None:
    module = load_module("gpt5_aic_profiles", GPT52_ROOT / "plot_gpt52_profiles_aic.py")
    output = AIC_DIR / "likelihood_surface"
    output.mkdir(parents=True, exist_ok=True)
    scores = module.calculate_run_aic(nll_scores)
    theta = module.summarise_marginal(scores, "theta")
    pgen = module.summarise_marginal(scores, "p_gen")
    pd.concat([pgen, theta], ignore_index=True).to_csv(
        output / "mean_aic_marginal_profiles.csv", index=False
    )
    scores.to_csv(output / "run_aic_surface.csv", index=False)
    module.plot_profiles(
        theta,
        pgen,
        output / "mean_aic_marginal_profiles.png",
        "GPT-5 unreliable keys partially observed: sweep_17_8_2026 AIC profiles",
    )


def run_aic_selection(specs: dict) -> pd.DataFrame:
    selector = load_module("gpt5_aic_selection", GPT52_ROOT / "select_gpt52_best_models_aic.py")
    result = selector.fit_condition_aic(DATA, SWEEP, specs)
    output_csv = AIC_DIR / "gpt5_best_model_aic.csv"
    output_plot = AIC_DIR / "gpt5_best_model_aic_counts.png"
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    module = selector.load_nll_module()
    module.plot_model_counts(
        result, output_plot, "Best-fitting model per partially observed GPT-5 run (AIC; NLL configs)"
    )
    return result


def run_worst_grid(aic_surface: pd.DataFrame) -> None:
    grid_module = load_module(
        "gpt5_worst_grid", GPT52_ROOT / "test_grid" / "build_worst_rel_gen_5x5_heatmap.py"
    )
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    surface = aic_surface.copy()
    surface["theta"] = surface["theta"].map(normalise_theta)
    rel, gen = grid_module.select_worst_boundaries(surface, 5)
    grid, theta_rows, pgen_columns = grid_module.cross_grid(surface, rel, gen)
    surface.to_csv(GRID_DIR / "all_99_config_aic_surface.csv", index=False)
    rel.to_csv(GRID_DIR / "worst_soc_rel_configs.csv", index=False)
    gen.to_csv(GRID_DIR / "worst_soc_gen_configs.csv", index=False)
    grid.to_csv(GRID_DIR / "worst_rel_gen_crossed_5x5_surface.csv", index=False)
    grid_module.plot_grid(
        grid, theta_rows, pgen_columns, GRID_DIR / "worst_rel_gen_crossed_5x5_aic_heatmap.png"
    )


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(f"Run summarise_gpt5_runs.py first: missing {DATA}")
    nll_surface = run_nll_surface()
    specs, selected = select_region_nll_minima(nll_surface)
    selected.to_csv(NLL_DIR / "nll_selected_model_configs.csv", index=False)
    with (NLL_DIR / "nll_selected_model_configs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {name: {"theta": theta, "p_gen": pgen} for name, (theta, pgen) in specs.items()},
            handle,
            indent=2,
        )
    nll_scores = run_nll_profiles()
    nll_result = run_nll_selection(specs)
    aic_surface = run_aic_surface()
    run_aic_profiles(nll_scores)
    aic_result = run_aic_selection(specs)
    run_worst_grid(aic_surface)

    print("NLL-selected configurations:")
    print(selected.to_string(index=False))
    print("NLL winner counts:")
    print(nll_result["Model"].value_counts().to_string())
    print("AIC winner counts (same configurations):")
    print(aic_result["Model"].value_counts().to_string())
    print(f"Saved complete analysis under {HERE}")


if __name__ == "__main__":
    main()
