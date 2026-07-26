import csv
import json
import pathlib
import re
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from typing import Optional

from environment import Environment, key_box_mapping
from smc.llm_ps_robust import LlmPS
from llm.llm import LLM
#from plot_utils.plot_hist_empirical_error import plot_empirical_success_histogram
import matplotlib.pyplot as plt

HISTORY_STEM = "llm_ps_stochastic_history"
ACTIONS_STEM = "llm_ps_stochastic_actions"
LAST_HYP_HISTORY_DIR = "llm_ps_stochastic_last_hypothesis_history"
LAST_HYP_BY_RUN_FILE = "last_hypothesis_stochastic.json"
_RUN_NUMBER_RE = re.compile(rf"^{re.escape(HISTORY_STEM)}_(\d+)\.json$")


# =========================
# CONFIGS
# =========================
max_trials = 70
model_name = "qwen3.6-plus" #"deepseek-chat",#qwen3.6-plus
opening_prob = 1.0
runs = 20
run_number_base = None

RUN_TIMEOUT_SECONDS = 5 * 60

temperature = 0.7
max_tokens = 2000
ROOT_SAVE_DIR = pathlib.Path(
    r"C:\Users\MSN\Documents\Python\Causal-Inference-Under-Uncertainty"
    r"\training_results\LLM_baseline_results_072000\qwen36plusreliable"
)

MAKE_TIMESTAMP_SUBFOLDER = True

def safe_name(x) -> str:
    return str(x).replace(".", "p").replace("/", "_").replace("\\", "_").replace(" ", "")


def build_output_dir() -> pathlib.Path:
    if MAKE_TIMESTAMP_SUBFOLDER:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"{safe_name(model_name)}"
            f"__op_{safe_name(opening_prob)}"
            f"__runs_{runs}"
            f"__trials_{max_trials}"
            f"__timeout_{RUN_TIMEOUT_SECONDS}s"
            f"__{timestamp}"
        )
        out_dir = ROOT_SAVE_DIR / folder_name
    else:
        out_dir = ROOT_SAVE_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def next_llm_ps_stochastic_run_number(out_dir: pathlib.Path) -> int:
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

    runs_data = [r for r in data.get("runs", []) if r.get("run_number") != run_number]
    runs_data.append(
        {
            "run_number": run_number,
            "last_hypothesis": last_hypothesis,
            "solved": result.get("solved", False),
            "trials": result.get("trials", 0),
            "opened": result.get("opened", 0),
            "aborted": result.get("aborted", False),
            "abort_reason": result.get("abort_reason"),
            "run_duration_seconds": result.get("run_duration_seconds"),
        }
    )

    data["runs"] = sorted(runs_data, key=lambda r: int(r["run_number"]))

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return path


def enrich_llm_ps_stochastic_history(history: list) -> list:
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


def write_llm_ps_stochastic_run_artifacts(
    out_dir: pathlib.Path,
    run_number: int,
    result: dict,
    model_name: Optional[str],
    max_trials: int,
) -> tuple[pathlib.Path, Optional[pathlib.Path], pathlib.Path]:
    history = enrich_llm_ps_stochastic_history(result.get("history", []))

    payload = {
        "meta": {
            "run_number": run_number,
            "model_name": model_name,
            "max_trials": max_trials,
            "opening_prob": opening_prob,
            "configured_temperature": temperature,
            "configured_max_tokens": max_tokens,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
        },
        **{**result, "history": history},
    }

    json_path = out_dir / f"{HISTORY_STEM}_{run_number}.json"
    csv_path = out_dir / f"{ACTIONS_STEM}_{run_number}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = _csv_rows_from_history(history)
    written_csv_path = None
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
        written_csv_path = csv_path

    last_h = last_hypothesis_from_history(history)
    last_hyp_path = append_last_hypothesis_log(out_dir, run_number, last_h, result)

    print(f"Wrote {json_path}")
    if written_csv_path:
        print(f"Wrote {written_csv_path}")
    else:
        print(f"No action CSV written for run {run_number}; history was empty or aborted.")
    print(f"Updated {last_hyp_path}")

    return json_path, written_csv_path, last_hyp_path


class _NoopLogger:
    def log(self, _msg: str) -> None:
        pass


def empirical_success_rate_from_history(history: list) -> Optional[float]:
    supposed = sum(int(s.get("supposed_to_open", 0) or 0) for s in history)
    if supposed == 0:
        return None

    successes = sum(
        int(s.get("success", 0) or 0)
        for s in history
        if int(s.get("supposed_to_open", 0) or 0)
    )
    return successes / supposed


def _build_agent(env, logger):
    llm = LLM(
        model=model_name,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens if max_tokens is not None else 512,
    )

    return LlmPS(
        env,
        llm,
        logger,
        temperature=temperature if temperature is not None else 0.1,
        max_tokens=max_tokens if max_tokens is not None else 512,
    )


def _run_one_worker(queue: mp.Queue) -> None:
    """
    Runs inside a child process so the parent can kill it after timeout.
    """
    start = time.perf_counter()

    try:
        env = Environment(opening_prob=opening_prob, include_inspect=False)
        logger = _NoopLogger()
        agent = _build_agent(env, logger)

        result = agent.run(max_trials=max_trials)

        if not isinstance(result, dict):
            result = {
                "history": result if isinstance(result, list) else [],
                "solved": bool(env.is_solved()),
                "trials": len(result) if isinstance(result, list) else 0,
                "opened": len(env.success_pairs),
            }

        result.setdefault("history", [])
        result.setdefault("solved", bool(env.is_solved()))
        result.setdefault("trials", len(result.get("history", [])))
        result.setdefault("opened", len(env.success_pairs))

        result["aborted"] = False
        result["abort_reason"] = None
        result["run_duration_seconds"] = round(time.perf_counter() - start, 4)

        queue.put(
            {
                "ok": True,
                "result": result,
                "actual_model_name": getattr(getattr(agent, "llm", None), "model", model_name),
            }
        )

    except Exception:
        queue.put(
            {
                "ok": False,
                "result": {
                    "history": [],
                    "solved": False,
                    "trials": 0,
                    "opened": 0,
                    "aborted": True,
                    "abort_reason": "exception",
                    "exception_traceback": traceback.format_exc(),
                    "run_duration_seconds": round(time.perf_counter() - start, 4),
                },
                "actual_model_name": model_name,
            }
        )


def run_one_with_timeout() -> tuple[dict, str]:
    queue = mp.Queue()
    process = mp.Process(target=_run_one_worker, args=(queue,))

    start = time.perf_counter()
    process.start()
    process.join(timeout=RUN_TIMEOUT_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join()

        bad_result = {
            "history": [],
            "solved": False,
            "trials": 0,
            "opened": 0,
            "aborted": True,
            "abort_reason": f"timeout_over_{RUN_TIMEOUT_SECONDS}_seconds",
            "run_duration_seconds": round(time.perf_counter() - start, 4),
        }
        return bad_result, model_name

    if queue.empty():
        bad_result = {
            "history": [],
            "solved": False,
            "trials": 0,
            "opened": 0,
            "aborted": True,
            "abort_reason": "worker_exited_without_result",
            "run_duration_seconds": round(time.perf_counter() - start, 4),
        }
        return bad_result, model_name

    payload = queue.get()
    return payload["result"], payload.get("actual_model_name", model_name)

def plot_empirical_success_histogram_local(
    runs_for_plot: list,
    out_path: pathlib.Path,
    source_label: str = "",
) -> bool:
    """
    Local replacement for plot_empirical_success_histogram.

    Plots distribution of empirical success rate per run:
    successes on supposed-to-open attempts / supposed-to-open attempts.

    Returns False if no valid run exists.
    """
    rates = []

    for run in runs_for_plot:
        history = run.get("history") or []

        supposed = 0
        successes = 0

        for step in history:
            if int(step.get("supposed_to_open", 0) or 0):
                supposed += 1
                if int(step.get("success", 0) or 0):
                    successes += 1

        if supposed > 0:
            rates.append(successes / supposed)

    if not rates:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(rates, bins=10, edgecolor="black")
    plt.xlabel("Empirical success rate on supposed-to-open attempts")
    plt.ylabel("Number of runs")
    plt.title(source_label or "Empirical success histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return True

if __name__ == "__main__":
    mp.freeze_support()

    out_dir = build_output_dir()
    histogram_out = out_dir / "hist_llm_ps_stochastic_empirical_success.png"

    print(f"Saving all results under: {out_dir}")

    payloads_for_histogram = []
    total_supposed_to_open = 0
    total_error = 0
    total_success_on_supposed = 0
    per_run_empirical_rates: list[float] = []

    for k in range(runs):
        if run_number_base is not None:
            run_number = run_number_base + k
        else:
            run_number = next_llm_ps_stochastic_run_number(out_dir)

        print(f"\n===== RUN {run_number} / {runs} =====")
        result, actual_model_name = run_one_with_timeout()
        if result.get("aborted"):
            print("\nRUN ABORTED DETAILS:")
            print("reason:", result.get("abort_reason"))
            if result.get("exception_traceback"):
                print(result["exception_traceback"])
        write_llm_ps_stochastic_run_artifacts(
            out_dir,
            run_number,
            result,
            actual_model_name,
            max_trials,
        )

        json_path = out_dir / f"{HISTORY_STEM}_{run_number}.json"
        with json_path.open(encoding="utf-8") as f:
            payload = json.load(f)

        if not result.get("aborted", False):
            payloads_for_histogram.append(payload)

        hist = enrich_llm_ps_stochastic_history(result.get("history", []))
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

        print(
            f"Run {run_number} finished. "
            f"aborted={result.get('aborted', False)}, "
            f"reason={result.get('abort_reason')}, "
            f"duration={result.get('run_duration_seconds')}s"
        )

    error_rate = (
        total_error / total_supposed_to_open if total_supposed_to_open > 0 else 0.0
    )
    aggregate_empirical_success = (
        total_success_on_supposed / total_supposed_to_open
        if total_supposed_to_open > 0
        else 0.0
    )
    mean_per_run_empirical_success = (
        sum(per_run_empirical_rates) / len(per_run_empirical_rates)
        if per_run_empirical_rates
        else None
    )

    summary_path = out_dir / "summary.json"
    summary = {
        "model_name": model_name,
        "opening_prob": opening_prob,
        "runs": runs,
        "max_trials": max_trials,
        "timeout_seconds": RUN_TIMEOUT_SECONDS,
        "total_supposed_to_open": total_supposed_to_open,
        "total_error": total_error,
        "total_success_on_supposed": total_success_on_supposed,
        "error_rate": error_rate,
        "aggregate_empirical_success": aggregate_empirical_success,
        "mean_per_run_empirical_success": mean_per_run_empirical_success,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote summary: {summary_path}")

    print(
        f"Aggregate error rate: "
        f"{total_error}/{total_supposed_to_open} = {error_rate:.4f}"
    )
    print(
        f"Aggregate empirical success: "
        f"{total_success_on_supposed}/{total_supposed_to_open} = "
        f"{aggregate_empirical_success:.4f}"
    )

    if mean_per_run_empirical_success is not None:
        print(
            f"Mean per-run empirical success rate: "
            f"{mean_per_run_empirical_success:.4f} "
            f"({len(per_run_empirical_rates)} valid runs)"
        )

    runs_for_plot = [
        {
            "run_idx": (p.get("meta") or {}).get("run_number", "?"),
            "history": p.get("history") or [],
        }
        for p in payloads_for_histogram
    ]

    label = f"{HISTORY_STEM}_<n>.json ({len(runs_for_plot)} valid runs)"

    if not runs_for_plot:
        print("No valid non-aborted runs for histogram; skipping plot.")
    else:
        if not plot_empirical_success_histogram_local(
            runs_for_plot,
            histogram_out,
            source_label=label,
        ):
            print("No valid runs for empirical success histogram.")
        else:
            print(f"Wrote histogram: {histogram_out}")