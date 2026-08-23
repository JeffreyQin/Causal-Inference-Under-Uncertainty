"""Select the best of four pre-selected SoC models for every adult."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "adult behaviour analysis zhuangfei" / "reliablekey_vs_unreliablekey"
SWEEP_DIR = PROJECT_ROOT / "training_results" / "sweep_17_8_2026"
OUTPUT_DIR = PROJECT_ROOT / "adult behaviour analysis zhuangfei"

TRUE_PRIOR = 0.02
EPS = 1e-10
MAX_OPEN = 5

CONDITIONS = {
    "reliable": {
        "data": DATA_DIR / "experiments_reliable_key.csv",
        "output": OUTPUT_DIR / "reliable_adults_best_model_nll.csv",
        "plot": OUTPUT_DIR / "reliable_adults_best_model_counts.png",
        "title": "Best-fitting model per reliable adult",
        "models": {
            "SoC-Full": ((1, 3), 0.7),
            "SoC-Rel": ((1, 3), 0.1),
            "SoC-Gen": ((19, 1), 0.7),
            "SoC-Lesioned": ((19, 1), 0.1),
        },
    },
    "unreliable": {
        "data": DATA_DIR / "experiments_unreliable_key.csv",
        "output": OUTPUT_DIR / "unreliable_key_adults_best_model_nll.csv",
        "plot": OUTPUT_DIR / "unreliable_key_adults_best_model_counts.png",
        "title": "Best-fitting model per unreliable-key adult",
        "models": {
            "SoC-Full": ((1, 3), 0.8),
            "SoC-Rel": ((1, 3), 0.1),
            "SoC-Gen": ((19, 1), 0.8),
            "SoC-Lesioned": ((19, 1), 0.1),
        },
    },
}


def sweep_filename(theta: tuple[int, int], p_gen: float) -> Path:
    gen_token = str(p_gen).replace(".", "p")
    prior_token = str(TRUE_PRIOR).replace(".", "p")
    return SWEEP_DIR / (
        f"theta_{theta[0]}_{theta[1]}__gen_{gen_token}"
        f"__trueprior_{prior_token}.csv"
    )


def load_model(theta: tuple[int, int], p_gen: float) -> pd.DataFrame:
    path = sweep_filename(theta, p_gen)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected sweep model: {path}")
    frame = pd.read_csv(path)
    expected = [f"P(n={n} boxes open)" for n in range(MAX_OPEN + 1)]
    missing = [column for column in ["trialNo", *expected] if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame.drop_duplicates("trialNo").set_index("trialNo")[expected]


def load_people(path: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    people = []
    for row_number, row in pd.read_csv(path).iterrows():
        record = json.loads(row["data"])
        person_id = str(row["id"]) if pd.notna(row.get("id")) else str(row_number)
        opened: set[str] = set()
        observations = []
        attempts = [a for a in record.get("attempts", []) if a.get("phase") == "learning"]
        for trial_number, attempt in enumerate(attempts, start=1):
            if bool(attempt.get("correct")):
                opened.add(str(attempt.get("doorName")))
            observations.append((trial_number, min(len(opened), MAX_OPEN)))
        people.append((person_id, observations))
    return people


def nll(observations: list[tuple[int, int]], model: pd.DataFrame) -> float:
    total = 0.0
    for trial_number, number_opened in observations:
        if trial_number not in model.index:
            continue
        probability = float(model.at[trial_number, f"P(n={number_opened} boxes open)"])
        total -= float(np.log(max(probability, EPS)))
    return total


def fit_condition(config: dict) -> pd.DataFrame:
    people = load_people(config["data"])
    models = {
        name: (theta, p_gen, load_model(theta, p_gen))
        for name, (theta, p_gen) in config["models"].items()
    }
    rows = []
    for person_id, observations in people:
        scores = {
            name: nll(observations, table)
            for name, (_, _, table) in models.items()
        }
        best_model = min(scores, key=scores.get)
        theta, p_gen, _ = models[best_model]
        rows.append(
            {
                "Person_Id": person_id,
                "Model": best_model,
                "negative Loglikelihood": scores[best_model],
                "parameters": (
                    f"theta=({theta[0]},{theta[1]}), "
                    f"p_gen={p_gen:g}, true_prior={TRUE_PRIOR:g}"
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_model_counts(result: pd.DataFrame, path: Path, title: str) -> None:
    model_order = ["SoC-Lesioned", "SoC-Rel", "SoC-Gen", "SoC-Full"]
    colors = ["#8B8B8B", "#4C72B0", "#55A868", "#C97E4B"]
    counts = result["Model"].value_counts().reindex(model_order, fill_value=0)

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    bars = axis.bar(model_order, counts.values, color=colors, alpha=0.88, width=0.72)
    axis.set_title(title, fontsize=15, fontweight="bold", pad=12)
    axis.set_ylabel("Number of adults", fontsize=12)
    axis.set_axisbelow(True)
    axis.grid(axis="both", color="#D5D5D5", linewidth=1, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CFCFCF")
    axis.spines["bottom"].set_color("#CFCFCF")
    axis.tick_params(axis="x", labelrotation=25, labelsize=11)
    axis.tick_params(axis="y", labelsize=11)
    axis.set_ylim(0, max(counts.max() * 1.16, 1))
    axis.bar_label(bars, labels=[str(value) for value in counts], padding=3, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    for condition, config in CONDITIONS.items():
        result = fit_condition(config)
        result.to_csv(config["output"], index=False, encoding="utf-8-sig")
        plot_model_counts(result, config["plot"], config["title"])
        print(f"{condition}: saved {len(result)} people to {config['output']}")
        print(f"{condition}: saved model-count graph to {config['plot']}")
        print(result["Model"].value_counts().to_string())


if __name__ == "__main__":
    main()
