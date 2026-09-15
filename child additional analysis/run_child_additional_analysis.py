"""Simulate each child's best fitted SoC setting and log additional behavior metrics.

Defaults: 100 runs per child, at most 70 attempts per run. Outputs are saved
under training_results/additional feature directional analysis, in a new
timestamped directory for each experiment. Fit each child within each of the
four variants before simulation, using the original grid NLL method.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for directory in (HERE.parent, HERE.parent / "smc"):
    sys.path.insert(0, str(directory))
from environment import Environment
from generator import Generator
from smc_soc import Engine

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def make_smc_config(theta):
    return {"num_particles": 30, "init_theta": theta, "ess_threshold": 0.5,
            "skill": True, "mode": "soc", "prior": "uniform"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


VARIANTS = ("SoC-Full", "SoC-Rel", "SoC-Gen", "SoC-Lesioned")
METRICS = ("unique_key_box_pairs", "revisit_rate", "attempts_before_abandoning_color_matching")


def fit_each_variant(overall_fits, model_path, trial_path):
    """Match fit_child_soc_models_nll.py, minimizing NLL within each region.

    Exact ties select the first configuration in string theta/numeric p_gen order,
    matching the original pandas groupby ordering for this grid.
    """
    models = {}
    with model_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            theta = tuple(ast.literal_eval(row["theta"]))
            key = (theta, float(row["gen"]), float(row["true_prior"]))
            models.setdefault(key, {})[int(row["trialNo"])] = [
                float(row[f"P(n={n} boxes open)"]) for n in range(6)]
    if len(models) != 99:
        raise ValueError(f"Expected 99 model configurations; got {len(models)}")
    trials = {}
    with trial_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            trials.setdefault(row["ID"], []).append(row)
    fits = []
    for overall in overall_fits:
        child = overall["Child_ID"]
        opened, observations = set(), []
        for row in sorted(trials[child], key=lambda r: float(r["Order"])):
            if float(row["Worked"]) == 1 and row["Box"]:
                opened.add(row["Box"])
            observations.append(min(len(opened), 5))
        best = {}
        for theta, p_gen, prior in sorted(models, key=lambda k: (str(k[0]).replace(" ", ""), k[1])):
            variant = ("SoC-Lesioned" if math.isclose(p_gen, 0.1) else "SoC-Gen") if theta == (19, 1) else ("SoC-Rel" if math.isclose(p_gen, 0.1) else "SoC-Full")
            table = models[(theta, p_gen, prior)]
            nll = sum(-math.log(max(table[t][n], 1e-10))
                      for t, n in enumerate(observations, 1) if t in table)
            if variant not in best or nll < best[variant]["NLL"]:
                best[variant] = {"Child_ID": child, "theta": theta,
                                 "p_gen": p_gen, "true_prior": prior,
                                 "SoC_variant": variant, "NLL": nll}
        if set(best) != set(VARIANTS):
            raise ValueError(f"Missing variant fits for {child}")
        # Check the reproduced global optimum against the supplied fitted CSV.
        if "NLL" in overall and not math.isclose(min(f["NLL"] for f in best.values()), float(overall["NLL"]), abs_tol=1e-8):
            raise ValueError(f"Recomputed NLL disagrees with supplied best fit for {child}")
        fits.extend(best[v] for v in VARIANTS)
    return fits


def fit_slug(fit):
    theta = "_".join(str(x) for x in fit["theta"])
    value = f"{fit['Child_ID']}__{fit['SoC_variant']}__theta_{theta}__pgen_{fit['p_gen']}__prior_{fit['true_prior']}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


class FileLogger:
    def __init__(self, handle):
        self.handle = handle

    def log(self, message):
        self.handle.write(str(message) + "\n")
        self.handle.flush()


def measure_attempts(evidence):
    """Count actual Attempt actions; revisit rate is undefined for fewer than two."""
    seen = set()
    rows = []
    initial_color_count = 0
    abandoned = False
    for key, box, outcome in evidence:
        if key == "inspect":
            continue
        pair = (key.id, box.id)
        revisit = pair in seen
        matched = key.color == box.color
        if not abandoned and matched:
            initial_color_count += 1
        if not matched:
            abandoned = True
        seen.add(pair)
        rows.append({
            "attempt_number": len(rows) + 1,
            "key": key.id, "box": box.id,
            "key_color": key.color, "box_color": box.color,
            "outcome": outcome, "color_matched": matched,
            "is_revisit": revisit,
        })
    count = len(rows)
    revisits = sum(row["is_revisit"] for row in rows)
    return {
        "attempts": count,
        "unique_key_box_pairs": len(seen),
        "revisit_attempts": revisits,
        "revisit_rate": revisits / (count - 1) if count > 1 else None,
        "attempts_before_abandoning_color_matching": initial_color_count,
        "color_matching_abandoned": abandoned,
    }, rows


def load_fits(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Child_ID", "theta", "p_gen", "true_prior", "SoC_variant"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input must contain columns: {sorted(required)}")
        fits = list(reader)
    seen = set()
    for fit in fits:
        child = fit["Child_ID"]
        if not child or child in seen:
            raise ValueError(f"Missing or duplicate Child_ID: {child!r}")
        seen.add(child)
        theta = ast.literal_eval(fit["theta"])
        if not isinstance(theta, (tuple, list)) or len(theta) != 2:
            raise ValueError(f"Invalid theta for {child}")
        if not all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in theta):
            raise ValueError(f"Theta must contain two positive finite numbers: {child}")
        fit["theta"] = tuple(theta)
        for name in ("p_gen", "true_prior"):
            fit[name] = float(fit[name])
            if not math.isfinite(fit[name]) or not 0 <= fit[name] <= 1:
                raise ValueError(f"Invalid {name} for {child}")
    if not fits:
        raise ValueError("Input contains no children")
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=HERE / "child_nll_summary" / "child_best_fitted_soc_nll.csv")
    parser.add_argument("--models", type=Path, default=HERE.parent / "training_results" / "prob distribution per smc config" / "all_configs_probabilities.csv")
    parser.add_argument("--child-trials", type=Path, default=HERE.parent / "smc" / "child_data" / "trials.csv")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory; existing directories are rejected to protect logs.")
    args = parser.parse_args()
    if args.runs < 1 or args.max_trials < 2:
        parser.error("--runs must be positive and --max-trials must be at least 2 for revisit rates")
    overall_fits = load_fits(args.fits)
    if len(overall_fits) != 100:
        raise ValueError(f"Marta's output requires exactly 100 children; got {len(overall_fits)}")
    fits = fit_each_variant(overall_fits, args.models, args.child_trials)
    output = args.output_dir or HERE.parent / "training_results" / "additional feature directional analysis" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "child_best_fits_per_variant.csv", fits)
    marta_dir = output / "for_marta"
    marta_dir.mkdir()
    metadata = {
        "fits_file": str(args.fits.resolve()), "fits": fits,
        "models_file": str(args.models.resolve()), "child_trials_file": str(args.child_trials.resolve()),
        "runs_per_child": args.runs, "max_trials": args.max_trials,
        "seed": args.seed, "opening_prob": 1.0, "include_inspect": False,
        "smc_config_example": make_smc_config(fits[0]["theta"]),
        "revisit_rate_undefined": "Blank/null for fewer than 2 attempts; excluded from child mean.",
        "aggregation": "Arithmetic mean across simulations per child within each variant; 100 children per variant.",
        "horizon": "Stop when solved or max_trials reached; color prefix is measured over observed attempts.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    seed_source = random.Random(args.seed)
    summaries = []
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as run_file, \
         (output / "attempts.csv").open("w", newline="", encoding="utf-8") as attempt_file:
        run_writer = attempt_writer = None
        for fit in tqdm(fits, desc="Child-variant fits", unit="fit"):
            slug = fit_slug(fit)
            child_dir = output / fit["SoC_variant"] / slug
            child_dir.mkdir(parents=True)
            child_runs = []
            identity = {name: fit[name] for name in ("Child_ID", "SoC_variant", "theta", "p_gen", "true_prior")}
            for run_number in tqdm(range(1, args.runs + 1), desc=fit["Child_ID"], leave=False):
                run_seed = seed_source.randrange(2**63)
                random.seed(run_seed)
                context = {**identity, "run_number": run_number, "seed": run_seed}
                run_name = f"{slug}__run_{run_number:03d}"
                with (child_dir / f"{run_name}.log").open("w", encoding="utf-8") as log:
                    logger = FileLogger(log)
                    logger.log(json.dumps(context))
                    env = Environment(opening_prob=1.0, include_inspect=False)
                    generator = Generator({"train": False, "prop_random": fit["p_gen"],
                                           "true_prior": fit["true_prior"]}, env)
                    engine = Engine(make_smc_config(fit["theta"]), env, generator, logger)
                    history = engine.run(max_trials=args.max_trials)
                    metrics, attempts = measure_attempts(engine.evidence)
                    logger.log(json.dumps(metrics))
                with (child_dir / f"{run_name}.json").open("w", encoding="utf-8") as handle:
                    json.dump({**context, "metrics": metrics, "attempts": attempts,
                               "history": history}, handle)
                row = {**context, **metrics, "boxes_opened": history[-1]["opened"],
                       "solved": env.is_solved(), "final_theta": history[-1]["theta"],
                       "trajectory_file": str((child_dir / f"{run_name}.json").relative_to(output))}
                if run_writer is None:
                    run_writer = csv.DictWriter(run_file, fieldnames=list(row))
                    run_writer.writeheader()
                run_writer.writerow(row)
                run_file.flush()
                for attempt in attempts:
                    attempt_row = {**context, **attempt}
                    if attempt_writer is None:
                        attempt_writer = csv.DictWriter(attempt_file, fieldnames=list(attempt_row))
                        attempt_writer.writeheader()
                    attempt_writer.writerow(attempt_row)
                attempt_file.flush()
                child_runs.append(row)
            summary = {**identity, "simulations": len(child_runs)}
            for metric in METRICS:
                values = [row[metric] for row in child_runs if row[metric] is not None]
                summary[f"mean_{metric}"] = sum(values) / len(values) if values else None
                summary[f"valid_runs_{metric}"] = len(values)
            summaries.append(summary)
            write_csv(child_dir / f"{slug}__runs.csv", child_runs)
            write_csv(child_dir / f"{slug}__means.csv", [summary])
            write_csv(output / "child_metric_means.csv", summaries)
    for variant in VARIANTS:
        rows = [row for row in summaries if row["SoC_variant"] == variant]
        assert len(rows) == 100 and len({row["Child_ID"] for row in rows}) == 100
        assert all(row[f"valid_runs_{metric}"] == args.runs for row in rows for metric in METRICS)
        write_csv(marta_dir / f"{variant}__100_child_predictions.csv", rows)
    print(f"Completed 100 children x {len(VARIANTS)} variants x {args.runs} simulations.")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
