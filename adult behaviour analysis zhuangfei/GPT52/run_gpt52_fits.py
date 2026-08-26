"""Run the adult-data fitting analyses on the GPT-5.2 run summaries."""

from __future__ import annotations

import importlib.util
import argparse
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


def run_selected_model_fits(
    sweep_name: str,
    condition: str,
    output_subdir: str | None,
) -> None:
    module = load_module(
        "selected_model_fit",
        ANALYSIS_ROOT / "calculate_selected_model_person_nll.py",
    )
    module.SWEEP_DIR = PROJECT_ROOT / "training_results" / sweep_name
    conditions = {
        "reliable_keys": {
            "data": GPT_ROOT / "reliable_keys" / "experiments_gpt5.2.csv",
            "output_dir": GPT_ROOT / "reliable_keys",
            "title": "Best-fitting model per GPT-5.2 reliable-key run",
            "models": module.CONDITIONS["reliable"]["models"],
        },
        "unreliable_keys_observed": {
            "data": GPT_ROOT / "unreliable_keys_observed" / "experiments_gpt5.2.csv",
            "output_dir": GPT_ROOT / "unreliable_keys_observed",
            "title": "Best-fitting model per GPT-5.2 unreliable-key run",
            "models": module.CONDITIONS["unreliable"]["models"],
        },
    }
    if condition != "all":
        selected_name = (
            "reliable_keys" if condition == "reliable" else "unreliable_keys_observed"
        )
        conditions = {selected_name: conditions[selected_name]}
    for name, config in conditions.items():
        output_dir = config.pop("output_dir")
        if output_subdir:
            output_dir /= output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        config["output"] = output_dir / "gpt5.2_best_model_nll.csv"
        config["plot"] = output_dir / "gpt5.2_best_model_counts.png"
        result = module.fit_condition(config)
        result.to_csv(config["output"], index=False, encoding="utf-8-sig")
        module.plot_model_counts(result, config["plot"], config["title"])
        print(f"{name}: selected-model fit saved ({len(result)} runs)")


def run_sweep_fit(
    module_name: str,
    script_name: str,
    folder_name: str,
    sweep_name: str,
    output_subdir: str | None,
) -> None:
    module = load_module(
        module_name,
        ANALYSIS_ROOT / "negative_log_likelihood_heatmaps" / script_name,
    )
    output_dir = GPT_ROOT / folder_name
    if output_subdir:
        output_dir /= output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    module.ADULT_CSV = GPT_ROOT / folder_name / "experiments_gpt5.2.csv"
    module.MODEL_RESULTS_DIR = PROJECT_ROOT / "training_results" / sweep_name
    module.OUTPUT_CSV = output_dir / f"gpt5.2_{sweep_name}_mean_nll.csv"
    module.OUTPUT_HEATMAP = output_dir / f"gpt5.2_{sweep_name}_heatmap.png"
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="sweep_17_8_2026")
    parser.add_argument(
        "--condition",
        choices=("all", "reliable", "unreliable"),
        default="all",
    )
    parser.add_argument("--skip-selected-models", action="store_true")
    parser.add_argument("--selected-output-subdir")
    args = parser.parse_args()

    if not args.skip_selected_models:
        run_selected_model_fits(args.sweep, args.condition, args.selected_output_subdir)
    if args.condition in ("all", "reliable"):
        run_sweep_fit(
            "reliable_sweep_fit",
            "fit_reliable_adultsl.py",
            "reliable_keys",
            args.sweep,
            args.selected_output_subdir,
        )
    if args.condition in ("all", "unreliable"):
        run_sweep_fit(
            "unreliable_sweep_fit",
            "fit_unreliable_adults.py",
            "unreliable_keys_observed",
            args.sweep,
            args.selected_output_subdir,
        )


if __name__ == "__main__":
    main()
