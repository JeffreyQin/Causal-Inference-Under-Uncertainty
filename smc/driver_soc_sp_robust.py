from environment import Environment
from smc.llm.llm_robust import LLM
from smc.smc_sp_robust import Engine

import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


smc_config = {
    "num_particles": 5,
    "init_theta": (19, 1),
    "ess_threshold": 0.5,
    "act_mode": "sample",
    "max_refine_attempts_per_trial": 3,
    "rejuvenate_on_low_ess_only": True,
    "csv_dir": r"training_results\nips_llm_vs2\deepseekchatV32_200_07__op_0.6__theta_19_1__p_5__mode_sample__runs_100__trials_70",
}

llm_config = {
    "model": "deepseek-chat",#qwen3.6-plus deepseek-chat
    "temperature": 0.6,
    "max_tokens": 200,
}

max_trials = 70
num_runs = 20
opening_prob = 1.0


def format_particle_refine_summary(row, num_particles: int) -> str:
    parts = []
    for i in range(num_particles):
        attempts = row.get(f"particle_{i}_refine_attempts", 0)
        status = row.get(f"particle_{i}_refine_status", "not_needed")
        invalid_before = row.get(f"particle_{i}_invalid_before_refine", False)
        name = row.get(f"particle_{i}_name", f"particle_{i}")
        if invalid_before or attempts > 0 or status != "not_needed":
            parts.append(
                f"p{i}:{name}|invalid_before={invalid_before}|attempts={attempts}|status={status}"
            )
    return "; ".join(parts) if parts else "no particle needed refine"


def save_trials_to_solve_histogram(solve_trial_counts, save_dir, model_name, timestamp, opening_prob):
    solved_trial_counts = [t for t in solve_trial_counts if t is not None]

    if not solved_trial_counts:
        print("No solved runs to plot in histogram.")
        return None

    os.makedirs(save_dir, exist_ok=True)

    plot_path = os.path.join(
        save_dir,
        f"{model_name}_summary_{timestamp}_openingprob_{opening_prob}_trials_to_open_5_boxes_histogram.png"
    )

    counts = Counter(solved_trial_counts)
    x = sorted(counts.keys())
    y = [counts[v] for v in x]

    plt.figure(figsize=(8, 5))
    plt.bar(x, y)
    plt.xlabel("Trial number when 5th box was opened")
    plt.ylabel("Number of runs")
    plt.title("Histogram of trials needed to open all 5 boxes across runs")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved histogram to: {plot_path}")
    return plot_path


if __name__ == "__main__":
    os.makedirs(smc_config["csv_dir"], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 100)
    print("SMC-SP RETRY-CONTROL EXPERIMENT START")
    print("=" * 100)
    print(f"Timestamp: {timestamp}")
    print(f"Save dir: {smc_config['csv_dir']}")
    print(f"Model: {llm_config['model']}")
    print(f"Particles: {smc_config['num_particles']}")
    print(f"Theta init: {smc_config['init_theta']}")
    print(f"ESS threshold: {smc_config['ess_threshold']}")
    print(f"Action mode: {smc_config['act_mode']}")
    print(f"Max refine attempts per trial: {smc_config['max_refine_attempts_per_trial']}")
    print(f"ESS-gated rejuvenation only: {smc_config['rejuvenate_on_low_ess_only']}")
    print(f"Opening probability: {opening_prob}")

    trial_counts = []
    solve_trial_counts = []
    aborted_runs = []
    solved_runs = 0

    for run_idx in range(num_runs):
        print("\n" + "#" * 100)
        print(f"START OF RUN {run_idx + 1}/{num_runs}")
        print("#" * 100)

        environment = Environment(opening_prob=opening_prob, include_inspect=False)
        logger = Logger(logging=False)
        llm = LLM(
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )

        smc_engine = Engine(smc_config, environment, llm, logger)
        result = smc_engine.run(max_trials=max_trials, run_number=run_idx + 1)

        history = result["history"]
        csv_path = result["csv_path"]
        trial_rows = smc_engine.trial_rows

        print(f"Run {run_idx + 1}: total trial rows recorded = {len(trial_rows)}")
        for row in trial_rows:
            print("-" * 100)
            print(
                f"Run {row['run_number']} | Trial {row['trial_no']} | "
                f"Action {row['action_pair']} | Outcome={row['outcome']} | "
                f"Opened={row['boxes_opened']} | Theta={row['theta']:.4f} | ESS={row['ess']:.4f}"
            )
            print(
                f"Selected particle: idx={row['selected_particle_index']} "
                f"name={row['selected_particle_name']} | "
                f"Rejuvenated this trial: {row['rejuvenated_this_trial']}"
            )
            print(f"Success pairs so far: {row['success_pairs']}")
            print("Refine summary:")
            print(format_particle_refine_summary(row, smc_config["num_particles"]))
            if row["run_aborted"]:
                print(f"RUN ABORT FLAG SET: {row['run_abort_reason']}")

        trials_completed = result["trials_completed"]
        final_opened = history[-1]["opened"] if history else 0
        trial_counts.append(trials_completed)

        if result["run_aborted"]:
            aborted_runs.append((run_idx + 1, result["run_abort_reason"]))

        if result["solved"]:
            solved_runs += 1
            solve_trial_no = history[-1]["t"] if history else None
            solve_trial_counts.append(solve_trial_no)
        else:
            solve_trial_counts.append(None)

        print("\n" + "=" * 100)
        print(f"END OF RUN {run_idx + 1}/{num_runs}")
        print(f"Trials completed: {trials_completed}")
        print(f"Boxes opened: {final_opened}")
        print(f"Solved: {result['solved']}")
        print(f"Run aborted: {result['run_aborted']}")
        if result["run_aborted"]:
            print(f"Abort reason: {result['run_abort_reason']}")
        print(f"Saved CSV: {csv_path}")
        print("=" * 100)

    model_name = llm_config["model"].replace("-", "_")
    histogram_path = save_trials_to_solve_histogram(
        solve_trial_counts=solve_trial_counts,
        save_dir=smc_config["csv_dir"],
        model_name=model_name,
        timestamp=timestamp,
        opening_prob=opening_prob,
    )

    print("\n" + "#" * 100)
    print("EXPERIMENT FINISHED")
    print("#" * 100)
    print(f"Trials per run: {trial_counts}")
    print(f"Solve trial counts: {solve_trial_counts}")
    print(f"Solved runs: {solved_runs}/{num_runs}")
    if aborted_runs:
        print("Aborted runs:")
        for run_no, reason in aborted_runs:
            print(f"  Run {run_no}: {reason}")
    else:
        print("Aborted runs: none")

    if histogram_path is not None:
        print(f"Saved histogram: {histogram_path}")