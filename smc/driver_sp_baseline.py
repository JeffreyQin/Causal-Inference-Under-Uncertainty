import csv
import json
import pathlib
import re
from typing import Optional

from environment import Environment, key_box_mapping
from smc.llm_ps import SPBaseline
from utils.plot_hist_empirical_error import plot_empirical_success_histogram

HISTORY_STEM = "sp_baseline_sto_history"
ACTIONS_STEM = "sp_baseline_sto_actions"
LAST_HYP_HISTORY_DIR = "sp_baseline_sto_last_hypothesis_history"
LAST_HYP_BY_RUN_FILE = "last_hypothesis_sto.json"
_RUN_NUMBER_RE = re.compile(rf"^{re.escape(HISTORY_STEM)}_(\d+)\.json$")


def next_sp_baseline_run_number(out_dir: pathlib.Path) -> int:
    """Next run index from existing numbered history JSON files in ``out_dir``."""
    max_n = 0
    for p in out_dir.iterdir():
        m = _RUN_NUMBER_RE.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def last_hypothesis_from_history(history: list) -> Optional[str]:
    for entry in reversed(history):
        h = entry.get("hypothesis")
        if h is not None:
            return h
    return None


def append_last_hypothesis_log(
    out_dir: pathlib.Path,
    run_number: int,
    last_hypothesis: Optional[str],
    result: dict,
) -> pathlib.Path:
    path = out_dir / LAST_HYP_HISTORY_DIR / LAST_HYP_BY_RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"runs": []}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    runs = [r for r in data.get("runs", []) if r.get("run_number") != run_number]
    runs.append(
        {
            "run_number": run_number,
            "last_hypothesis": last_hypothesis,
            "solved": result["solved"],
            "trials": result["trials"],
            "opened": result["opened"],
        }
    )
    data["runs"] = sorted(runs, key=lambda r: int(r["run_number"]))
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def enrich_sp_baseline_history(history: list) -> list:
    """Derived success / supposed_to_open / error for each open attempt (same as partial driver)."""
    enriched = []
    for step in history:
        action = step.get("action") or []
        if len(action) >= 2 and "outcome" in step:
            key_id, box_id = action[0], action[1]
            success = bool(step.get("outcome", False))
            supposed_to_open = (key_id, box_id) in key_box_mapping
            error = bool(supposed_to_open and (not success))
            enriched.append(
                {
                    **step,
                    "success": int(success),
                    "supposed_to_open": int(supposed_to_open),
                    "error": int(error),
                }
            )
        else:
            enriched.append(dict(step))
    return enriched


def _csv_rows_from_history(history: list) -> list:
    rows = []
    for step in history:
        aid = int(step.get("t", 0))
        action = step.get("action") or []
        if len(action) >= 2 and "outcome" in step:
            rows.append(
                {
                    "action_id": aid,
                    "action_type": "attempt",
                    "key": action[0],
                    "box": action[1],
                    "success": step["success"],
                    "supposed_to_open": step["supposed_to_open"],
                    "error": step["error"],
                }
            )
    return rows


def write_sp_baseline_run_artifacts(
    out_dir: pathlib.Path,
    run_number: int,
    result: dict,
    model_name: Optional[str],
    max_trials: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Write enriched history JSON, CSV, and aggregate last-hypothesis log for one run."""
    history = enrich_sp_baseline_history(result["history"])
    payload = {
        "meta": {
            "run_number": run_number,
            "model_name": model_name,
            "max_trials": max_trials,
        },
        **{**result, "history": history},
    }
    json_path = out_dir / f"{HISTORY_STEM}_{run_number}.json"
    csv_path = out_dir / f"{ACTIONS_STEM}_{run_number}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = _csv_rows_from_history(history)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "action_id",
                    "action_type",
                    "key",
                    "box",
                    "success",
                    "supposed_to_open",
                    "error",
                ],
            )
            w.writeheader()
            w.writerows(rows)

    last_h = last_hypothesis_from_history(history)
    last_hyp_path = append_last_hypothesis_log(out_dir, run_number, last_h, result)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Updated {last_hyp_path} (last hypothesis for run {run_number})")

    return json_path, csv_path, last_hyp_path


class _NoopLogger:
    def log(self, _msg: str) -> None:
        pass


def empirical_success_rate_from_history(history: list) -> Optional[float]:
    """(# successes on supposed-to-open steps) / (count of supposed-to-open steps)."""
    supposed = sum(int(s.get("supposed_to_open", 0) or 0) for s in history)
    if supposed == 0:
        return None
    successes = sum(
        int(s.get("success", 0) or 0)
        for s in history
        if int(s.get("supposed_to_open", 0) or 0)
    )
    return successes / supposed


if __name__ == "__main__":
    max_trials = 70
    model_name = "gpt-5.2"
    opening_prob = 0.6
    runs = 20
    run_number_base = None  # set an int to force suffixes: base, base+1, ...

    out_dir = pathlib.Path(__file__).resolve().parent
    histogram_out = out_dir / "hist_sp_baseline_empirical_success.png"

    payloads_for_histogram = []
    total_supposed_to_open = 0
    total_error = 0
    total_success_on_supposed = 0
    per_run_empirical_rates: list[float] = []

    for k in range(runs):
        if run_number_base is not None:
            run_number = run_number_base + k
        else:
            run_number = next_sp_baseline_run_number(out_dir)

        env = Environment(opening_prob=opening_prob, include_inspect=False)
        logger = _NoopLogger()
        agent = SPBaseline(env, logger, model_name=model_name)
        result = agent.run(max_trials=max_trials)
        write_sp_baseline_run_artifacts(
            out_dir,
            run_number,
            result,
            getattr(agent.llm, "model", None),
            max_trials,
        )

        json_path = out_dir / f"{HISTORY_STEM}_{run_number}.json"
        with json_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        payloads_for_histogram.append(payload)

        hist = enrich_sp_baseline_history(result["history"])
        for step in hist:
            if int(step.get("supposed_to_open", 0) or 0):
                total_supposed_to_open += 1
                if int(step.get("error", 0) or 0):
                    total_error += 1
                if int(step.get("success", 0) or 0):
                    total_success_on_supposed += 1

        r = empirical_success_rate_from_history(hist)
        if r is not None:
            per_run_empirical_rates.append(r)

    error_rate = (
        (total_error / total_supposed_to_open) if total_supposed_to_open > 0 else 0.0
    )
    aggregate_empirical_success = (
        (total_success_on_supposed / total_supposed_to_open)
        if total_supposed_to_open > 0
        else 0.0
    )
    mean_per_run_empirical_success = (
        sum(per_run_empirical_rates) / len(per_run_empirical_rates)
        if per_run_empirical_rates
        else None
    )

    print(
        f"Aggregate error rate (supposed-to-open but failed): "
        f"{total_error}/{total_supposed_to_open} = {error_rate:.4f}"
    )
    print(
        f"Aggregate empirical success (on supposed-to-open attempts): "
        f"{total_success_on_supposed}/{total_supposed_to_open} = {aggregate_empirical_success:.4f}"
    )
    if mean_per_run_empirical_success is not None:
        print(
            f"Mean per-run empirical success rate: {mean_per_run_empirical_success:.4f} "
            f"({len(per_run_empirical_rates)} runs)"
        )

    runs_for_plot = [
        {
            "run_idx": (p.get("meta") or {}).get("run_number", "?"),
            "history": p.get("history") or [],
        }
        for p in payloads_for_histogram
    ]
    label = f"{HISTORY_STEM}_<n>.json ({len(runs_for_plot)} runs)"
    if not plot_empirical_success_histogram(
        runs_for_plot,
        histogram_out,
        source_label=label,
    ):
        print("no valid runs for empirical success histogram (no supposed_to_open steps)")