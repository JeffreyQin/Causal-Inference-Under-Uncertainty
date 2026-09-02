"""Select the best of four fixed SoC configurations using NLL."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="sweep_17_8_2026")
    parser.add_argument(
        "--condition", choices=("all", "reliable", "unreliable", "partial"), default="all"
    )
    parser.add_argument("--output-subdir")
    args = parser.parse_args()

    module = load_module(
        "selected_model_nll", ANALYSIS_ROOT / "calculate_selected_model_person_nll.py"
    )
    module.SWEEP_DIR = PROJECT_ROOT / "training_results" / args.sweep
    conditions = {
        "reliable": {
            "data": GPT_ROOT / "reliable_keys" / "experiments_gpt5.2.csv",
            "folder": "reliable_keys",
            "models": module.CONDITIONS["reliable"]["models"],
            "title": "Best-fitting model per GPT-5.2 reliable-key run (NLL)",
        },
        "unreliable": {
            "data": GPT_ROOT / "unreliable_keys_observed" / "experiments_gpt5.2.csv",
            "folder": "unreliable_keys_observed",
            "models": module.CONDITIONS["unreliable"]["models"],
            "title": "Best-fitting model per GPT-5.2 unreliable-key run (NLL)",
        },
        "partial": {
            "data": GPT_ROOT / "unreliable_kes_partially_observed" / "experiments_gpt5.2.csv",
            "folder": "unreliable_kes_partially_observed",
            # The partially observed sweep minimum is theta=(5,1), p_gen=0.8.
            # Retain the established Rel/Gen/Lesioned controls, but use this
            # dataset's fitted sweep result for SoC-Full.
            "models": {
                **module.CONDITIONS["unreliable"]["models"],
                "SoC-Full": ((5, 1), 0.8),
                "SoC-Rel": ((2, 1), 0.1),
                "SoC-Gen": ((19, 1), 0.9),
            },
            "title": "Best-fitting model per GPT-5.2 partially observed run (NLL)",
        },
    }
    names = conditions if args.condition == "all" else [args.condition]
    for name in names:
        config = conditions[name]
        output_dir = GPT_ROOT / config.pop("folder")
        if args.output_subdir:
            output_dir /= args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        config["output"] = output_dir / "gpt5.2_best_model_nll.csv"
        config["plot"] = output_dir / "gpt5.2_best_model_counts.png"
        result = module.fit_condition(config)
        result.to_csv(config["output"], index=False, encoding="utf-8-sig")
        module.plot_model_counts(result, config["plot"], config["title"])
        print(f"{name}: saved NLL selection for {len(result)} runs to {output_dir}")


if __name__ == "__main__":
    main()
