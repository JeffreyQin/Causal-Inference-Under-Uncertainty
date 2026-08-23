"""Run the adult-data fitting analyses on the GPT-5.2 run summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei"
GPT_ROOT = ANALYSIS_ROOT / "GPT52"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_selected_model_fits() -> None:
    module = load_module(
        "selected_model_fit",
        ANALYSIS_ROOT / "calculate_selected_model_person_nll.py",
    )
    conditions = {
        "reliable_keys": {
            "data": GPT_ROOT / "reliable_keys" / "experiments_gpt5.2.csv",
            "output": GPT_ROOT / "reliable_keys" / "gpt5.2_best_model_nll.csv",
            "plot": GPT_ROOT / "reliable_keys" / "gpt5.2_best_model_counts.png",
            "title": "Best-fitting model per GPT-5.2 reliable-key run",
            "models": module.CONDITIONS["reliable"]["models"],
        },
        "unreliable_keys_observed": {
            "data": GPT_ROOT / "unreliable_keys_observed" / "experiments_gpt5.2.csv",
            "output": GPT_ROOT / "unreliable_keys_observed" / "gpt5.2_best_model_nll.csv",
            "plot": GPT_ROOT / "unreliable_keys_observed" / "gpt5.2_best_model_counts.png",
            "title": "Best-fitting model per GPT-5.2 unreliable-key run",
            "models": module.CONDITIONS["unreliable"]["models"],
        },
    }
    for name, config in conditions.items():
        result = module.fit_condition(config)
        result.to_csv(config["output"], index=False, encoding="utf-8-sig")
        module.plot_model_counts(result, config["plot"], config["title"])
        print(f"{name}: selected-model fit saved ({len(result)} runs)")


def run_sweep_fit(module_name: str, script_name: str, folder_name: str) -> None:
    module = load_module(
        module_name,
        ANALYSIS_ROOT / "negative_log_likelihood_heatmaps" / script_name,
    )
    output_dir = GPT_ROOT / folder_name
    module.ADULT_CSV = output_dir / "experiments_gpt5.2.csv"
    module.MODEL_RESULTS_DIR = PROJECT_ROOT / "training_results" / "sweep_17_8_2026"
    module.OUTPUT_CSV = output_dir / "gpt5.2_sweep_17_8_2026_mean_nll.csv"
    module.OUTPUT_HEATMAP = output_dir / "gpt5.2_sweep_17_8_2026_heatmap.png"
    module.main()


def main() -> None:
    run_selected_model_fits()
    run_sweep_fit("reliable_sweep_fit", "fit_reliable_adultsl.py", "reliable_keys")
    run_sweep_fit(
        "unreliable_sweep_fit",
        "fit_unreliable_adults.py",
        "unreliable_keys_observed",
    )


if __name__ == "__main__":
    main()
