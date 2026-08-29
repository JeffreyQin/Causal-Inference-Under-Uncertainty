"""Calculate and plot GPT-5.2 mean-NLL sweep heatmaps."""

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


def run_condition(condition: str, sweep_name: str, output_subdir: str | None) -> None:
    if condition == "reliable":
        folder = "reliable_keys"
        script = "fit_reliable_adultsl.py"
    else:
        folder = "unreliable_keys_observed"
        script = "fit_unreliable_adults.py"
    module = load_module(
        f"{condition}_nll_heatmap",
        ANALYSIS_ROOT / "negative_log_likelihood_heatmaps" / script,
    )
    output_dir = GPT_ROOT / folder
    if output_subdir:
        output_dir /= output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    module.ADULT_CSV = GPT_ROOT / folder / "experiments_gpt5.2.csv"
    module.MODEL_RESULTS_DIR = PROJECT_ROOT / "training_results" / sweep_name
    module.OUTPUT_CSV = output_dir / f"gpt5.2_{sweep_name}_mean_nll.csv"
    module.OUTPUT_HEATMAP = output_dir / f"gpt5.2_{sweep_name}_nll_heatmap.png"
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="sweep_17_8_2026")
    parser.add_argument(
        "--condition", choices=("all", "reliable", "unreliable"), default="all"
    )
    parser.add_argument("--output-subdir")
    args = parser.parse_args()
    names = ("reliable", "unreliable") if args.condition == "all" else (args.condition,)
    for condition in names:
        run_condition(condition, args.sweep, args.output_subdir)


if __name__ == "__main__":
    main()
