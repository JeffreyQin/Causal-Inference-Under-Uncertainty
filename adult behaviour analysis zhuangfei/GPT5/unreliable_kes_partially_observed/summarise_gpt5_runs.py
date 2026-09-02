"""Convert GPT-5 partially observed histories to row-per-run adult format."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "smc" / "gpt5_partial_obs_result"
OUTPUT = Path(__file__).resolve().parent / "experiments_gpt5.csv"
PATTERN = "sp_baseline_partially_observed_history_*.json"


def run_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def convert(log: dict) -> dict:
    attempts = []
    for item in log.get("history", []):
        action = item.get("action")
        if not isinstance(action, list) or len(action) != 2:
            continue
        key, door = action
        if key is None or door is None or key == "examine":
            continue
        attempts.append(
            {
                "time": len(attempts) + 1,
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
    number = int(meta["run_number"])
    history = log.get("history", [])
    return {
        "age": None,
        "gender": None,
        "attempts": attempts,
        "comments": "",
        "hypothesis": history[-1].get("hypothesis", "") if history else "",
        "rule_guess": "",
        "session_id": f"gpt5-partial-observed-run-{number}",
        "genAttempts": [],
        "model_name": meta.get("model_name", "gpt-5"),
        "condition": "gpt5_partial_observed",
        "run_number": number,
        "solved": bool(log.get("solved", False)),
        "trials": log.get("trials"),
        "opened": log.get("opened"),
        "success_pairs": log.get("success_pairs", []),
    }


def main() -> None:
    paths = sorted(SOURCE_DIR.glob(PATTERN), key=run_number)
    if not paths:
        raise FileNotFoundError(f"No histories matched {SOURCE_DIR / PATTERN}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "created_at", "data", "session_id"])
        writer.writeheader()
        for path in paths:
            with path.open(encoding="utf-8") as source:
                record = convert(json.load(source))
            writer.writerow(
                {
                    "id": record["run_number"],
                    "created_at": "",
                    "data": json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                    "session_id": record["session_id"],
                }
            )
    print(f"Created {OUTPUT} from {len(paths)} runs")


if __name__ == "__main__":
    main()
