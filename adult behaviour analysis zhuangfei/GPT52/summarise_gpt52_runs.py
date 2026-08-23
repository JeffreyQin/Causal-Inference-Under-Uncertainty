"""Convert GPT-5.2 run logs to the row-per-agent format of the adult data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "smc" / "gpt5.2_data" / "full_run_logs"
OUTPUT_ROOT = PROJECT_ROOT / "adult behaviour analysis zhuangfei" / "GPT52"

CONDITIONS = {
    "gpt5.2_deterministic": ("reliable_keys", "llm_ps_deterministic_run_*.json"),
    "gpt5.2_stochastic_0.6_oracle": (
        "unreliable_keys_observed",
        "llm_ps_stochastic_run_*.json",
    ),
    "gpt5.2_partial_observed_0.6_oracle": (
        "unreliable_keys_unfullyobserved",
        "llm_ps_partially_obs_run_*.json",
    ),
}


def run_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def human_style_record(log: dict, condition: str) -> dict:
    attempts = []
    for item in log.get("history", []):
        action = item.get("action")
        if not isinstance(action, list) or len(action) != 2:
            continue
        key, door = action
        if key is None or door is None:  # Partial-observation examine action.
            continue
        attempts.append(
            {
                "time": item.get("t"),
                "phase": "learning",
                "keyName": key,
                "doorName": door,
                "correct": bool(item.get("success", item.get("outcome", False))),
                "supposed_to_open": item.get("supposed_to_open"),
                "error": item.get("error"),
                "hypothesis": item.get("hypothesis"),
            }
        )

    meta = log.get("meta", {})
    number = int(meta.get("run_number"))
    return {
        "age": None,
        "gender": None,
        "attempts": attempts,
        "comments": "",
        "hypothesis": (
            log.get("history", [{}])[-1].get("hypothesis", "")
            if log.get("history")
            else ""
        ),
        "rule_guess": "",
        "session_id": f"gpt5.2-{condition}-run-{number}",
        "genAttempts": [],
        "model_name": meta.get("model_name", "gpt-5.2"),
        "condition": condition,
        "run_number": number,
        "solved": bool(log.get("solved", False)),
        "trials": log.get("trials"),
        "opened": log.get("opened"),
        "success_pairs": log.get("success_pairs", []),
    }


def convert_condition(source_name: str, output_name: str, pattern: str) -> Path:
    source_dir = SOURCE_ROOT / source_name
    output_dir = OUTPUT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiments_gpt5.2.csv"
    paths = sorted(source_dir.glob(pattern), key=run_number)
    if not paths:
        raise FileNotFoundError(f"No run logs matched {source_dir / pattern}")

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "created_at", "data", "session_id"])
        writer.writeheader()
        for path in paths:
            with path.open(encoding="utf-8") as source:
                log = json.load(source)
            record = human_style_record(log, source_name)
            writer.writerow(
                {
                    "id": record["run_number"],
                    "created_at": "",
                    "data": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                    "session_id": record["session_id"],
                }
            )
    return output_path


def main() -> None:
    for source_name, (output_name, pattern) in CONDITIONS.items():
        path = convert_condition(source_name, output_name, pattern)
        print(f"Created {path}")


if __name__ == "__main__":
    main()
