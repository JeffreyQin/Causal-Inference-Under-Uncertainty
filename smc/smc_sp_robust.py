from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from environment import Environment, Key, Box
from llm.code import check_valid_program, execute_hypothesis_code
from smc.llm.llm_robust import LLM


@dataclass
class Particle:
    name: str
    hypothesis: str
    weight: float

    def is_executable(self) -> bool:
        return check_valid_program(self.hypothesis)

    def evaluate(self, key: Key, box: Box) -> bool:
        return execute_hypothesis_code(self.hypothesis, key, box)


class Engine:
    def __init__(self, config: Dict, env: Environment, llm: Optional[LLM] = None, logger=None):
        self.num_particles: int = config["num_particles"]
        self.act_mode: str = config["act_mode"]
        self.alpha0, self.beta0 = config["init_theta"]
        self.alpha, self.beta = config["init_theta"]
        self.ess_threshold: float = config.get("ess_threshold", 0.5)
        self.max_refine_attempts_per_trial: int = config.get("max_refine_attempts_per_trial", 3)
        self.csv_dir: str = config.get("csv_dir", "smc_sp_trial_csv")
        self.rejuvenate_on_low_ess_only: bool = config.get("rejuvenate_on_low_ess_only", True)

        self.env = env
        self.llm = llm
        self.logger = logger

        self.evidence: List[Tuple[Key, Box, bool]] = []
        self.succ_count = defaultdict(lambda: 0)
        self.fail_count = defaultdict(lambda: 0)
        self.history: List[Dict] = []
        self.run_aborted: bool = False
        self.run_abort_reason: str = ""
        self.run_number: int = -1
        self.csv_path: str = ""
        self.trial_rows: List[Dict] = []

        self.current_trial_refine_counts: List[int] = [0 for _ in range(self.num_particles)]
        self.current_trial_refine_status: List[str] = ["not_needed" for _ in range(self.num_particles)]
        self.current_trial_invalid_before_refine: List[bool] = [False for _ in range(self.num_particles)]
        self.current_trial_selected_particle_name: Optional[str] = None
        self.current_trial_selected_particle_index: Optional[int] = None
        self.current_trial_rejuvenated: bool = False
        self.current_trial_ess: float = 0.0

        self.particles = self._initialize_particles()

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.log(msg)

    def _initialize_particles(self) -> List[Particle]:
        particles: List[Particle] = []
        for i in range(self.num_particles):
            hypothesis, name = self.llm.generate_once(evidence=[])
            particles.append(Particle(name=name, hypothesis=hypothesis, weight=(1.0 / self.num_particles)))
        return particles

    def _reset_trial_flags(self) -> None:
        self.current_trial_refine_counts = [0 for _ in range(self.num_particles)]
        self.current_trial_refine_status = ["not_needed" for _ in range(self.num_particles)]
        self.current_trial_invalid_before_refine = [False for _ in range(self.num_particles)]
        self.current_trial_selected_particle_name = None
        self.current_trial_selected_particle_index = None
        self.current_trial_rejuvenated = False
        self.current_trial_ess = 0.0

    def _particle_probs_by_name(self) -> Dict[str, float]:
        probs: Dict[str, float] = {}
        for p in self.particles:
            probs[p.name] = probs.get(p.name, 0.0) + p.weight
        return probs

    def _compute_ess(self) -> float:
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            return 0.0
        return 1.0 / sum(w ** 2 for w in weights)

    def _compute_entropy(self, particle_weights: List[float]) -> float:
        weights = [w for w in particle_weights if w > 0]
        if len(weights) <= 1:
            return 0.0
        return -1.0 * sum(w * math.log2(w) for w in weights)

    def _compute_theta(self) -> None:
        self.alpha, self.beta = self.alpha0, self.beta0
        open_prob = defaultdict(lambda: 0.0)
        for (key, box, _) in self.evidence:
            if (key.id, box.id) in open_prob:
                continue
            open_prob[(key.id, box.id)] = sum(
                p.weight for p in self.particles if p.is_executable() and p.evaluate(key, box)
            )

        for kb_pair in open_prob.keys():
            self.alpha += open_prob[kb_pair] * self.succ_count[kb_pair]
            self.beta += open_prob[kb_pair] * self.fail_count[kb_pair]

        self.alpha = max(1e-9, self.alpha)
        self.beta = max(1e-9, self.beta)

    def _compute_likelihood(self, predict: bool, outcome: bool) -> float:
        assert self.alpha + self.beta > 0
        prob_success = self.alpha / (self.alpha + self.beta)
        if predict and outcome:
            return prob_success
        if predict and not outcome:
            return 1.0 - prob_success
        if not predict and outcome:
            return 0.0
        return 1.0

    def _compute_info_gain(self, key: Key, box: Box) -> float:
        current_entropy = self._compute_entropy([p.weight for p in self.particles])
        expected_entropy = 0.0

        for outcome in [True, False]:
            outcome_prob = 0.0
            updated_weights = []
            for particle in self.particles:
                if particle.is_executable():
                    pred_outcome = particle.evaluate(key, box)
                else:
                    pred_outcome = False
                likelihood = self._compute_likelihood(pred_outcome, outcome)
                outcome_prob += particle.weight * likelihood
                updated_weights.append(particle.weight * likelihood)

            if outcome_prob == 0:
                continue
            updated_weights = [w / outcome_prob for w in updated_weights if w > 0]
            new_entropy = self._compute_entropy(updated_weights)
            expected_entropy += outcome_prob * new_entropy

        return current_entropy - expected_entropy

    def _select_action_by_sample(self) -> Tuple[Key, Box]:
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            idx = random.randrange(len(self.particles))
        else:
            idx = random.choices(range(len(self.particles)), weights=weights, k=1)[0]
        particle = self.particles[idx]
        self.current_trial_selected_particle_index = idx
        self.current_trial_selected_particle_name = particle.name

        opened = {pair[1] for pair in self.env.success_pairs}
        candidate_actions = []
        fallback_actions = []
        for (key, box) in self.env.actions:
            if key == "inspect" or box.id in opened:
                continue
            pred = False
            if particle.is_executable():
                pred = particle.evaluate(key, box)
            if pred:
                candidate_actions.append((key, box))
            else:
                fallback_actions.append((key, box))

        if candidate_actions:
            return random.choice(candidate_actions)
        return random.choice(fallback_actions)

    def _select_action_by_bed(self) -> Tuple[Key, Box]:
        self.current_trial_selected_particle_index = None
        self.current_trial_selected_particle_name = None

        max_info_gain = float("-inf")
        best_actions = []
        for (key, box) in self.env.actions:
            if key == "inspect":
                continue
            if (key.id, box.id) in self.env.success_pairs:
                continue
            info_gain = self._compute_info_gain(key, box)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_actions = [(key, box)]
            elif info_gain == max_info_gain:
                best_actions.append((key, box))
        return random.choice(best_actions)

    def _resample(self) -> None:
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            indices = random.choices(range(self.num_particles), k=self.num_particles)
        else:
            indices = random.choices(range(self.num_particles), k=self.num_particles, weights=weights)

        resampled = []
        for i in indices:
            p = self.particles[i]
            resampled.append(Particle(name=p.name, hypothesis=p.hypothesis, weight=(1.0 / self.num_particles)))
        self.particles = resampled

    def _compute_h_likelihood(self, hypothesis: str) -> float:
        if not check_valid_program(hypothesis):
            return 0.0
        likelihood = 1.0
        for key, box, outcome in self.evidence:
            try:
                pred_match = execute_hypothesis_code(hypothesis, key, box)
            except Exception:
                return 0.0
            likelihood *= self._compute_likelihood(pred_match, outcome)
        return likelihood

    def _accept_h(self, hypothesis: str) -> bool:
        if not check_valid_program(hypothesis):
            return False
        for (key_id, box_id) in self.env.success_pairs:
            key = self.env.id_to_key[key_id]
            box = self.env.id_to_box[box_id]
            try:
                if execute_hypothesis_code(hypothesis, key, box) is False:
                    return False
            except Exception:
                return False
        return True

    def _rejuvenate(self) -> None:
        """ESS-triggered SMC rejuvenation only. Not code repair."""
        worst_idx = None
        worst_likelihood = float("inf")
        for i, particle in enumerate(self.particles):
            h_likelihood = self._compute_h_likelihood(particle.hypothesis)
            if h_likelihood < worst_likelihood:
                worst_likelihood = h_likelihood
                worst_idx = i

        if worst_idx is None:
            return

        old_h = self.particles[worst_idx].hypothesis
        new_h, new_name = self.llm.refine_once(self.evidence, old_h)
        if self._accept_h(new_h):
            self.particles[worst_idx] = Particle(
                name=new_name,
                hypothesis=new_h,
                weight=self.particles[worst_idx].weight,
            )
        self.current_trial_rejuvenated = True

    def _repair_invalid_particles_for_trial(self) -> None:
        """Always run every trial. Separate from ESS-gated rejuvenation."""
        invalid_indices = []
        for i, particle in enumerate(self.particles):
            invalid = not particle.is_executable()
            self.current_trial_invalid_before_refine[i] = invalid
            if invalid:
                invalid_indices.append(i)

        if not invalid_indices:
            return

        all_failed = True
        for i in invalid_indices:
            old_h = self.particles[i].hypothesis
            accepted = False

            for attempt in range(1, self.max_refine_attempts_per_trial + 1):
                self.current_trial_refine_counts[i] = attempt
                new_h, new_name = self.llm.refine_once(self.evidence, old_h)
                if self._accept_h(new_h):
                    self.particles[i] = Particle(
                        name=new_name,
                        hypothesis=new_h,
                        weight=self.particles[i].weight,
                    )
                    self.current_trial_refine_status[i] = "accepted_after_refine"
                    accepted = True
                    all_failed = False
                    break
                old_h = new_h

            if not accepted:
                self.current_trial_refine_status[i] = "refine_limit_exceeded"

        if invalid_indices and all_failed:
            self.run_aborted = True
            self.run_abort_reason = "all invalid particles exceeded 3 refine attempts in this trial"

    def _update_particle_weights(self, key: Key, box: Box, outcome: bool) -> None:
        for particle in self.particles:
            if particle.is_executable():
                pred_outcome = particle.evaluate(key, box)
            else:
                pred_outcome = False
            likelihood = self._compute_likelihood(pred_outcome, outcome)
            particle.weight *= likelihood

        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            for particle in self.particles:
                particle.weight = 1.0 / self.num_particles

        ess = self._compute_ess()
        self.current_trial_ess = ess
        if self.rejuvenate_on_low_ess_only and ess < self.num_particles * self.ess_threshold:
            self._resample()
            self._rejuvenate()

    def _snapshot_history(self, t: int, action_pair: Optional[Tuple[str, str]]) -> None:
        self.history.append({
            "t": t,
            "opened": len(self.env.success_pairs),
            "theta": self.alpha / (self.alpha + self.beta),
            "action": action_pair,
            "probs": self._particle_probs_by_name(),
        })

    def _append_trial_csv_row(self, trial_no: int, key: Key, box: Box, outcome: bool) -> None:
        selected_hypothesis = ""
        if self.current_trial_selected_particle_index is not None:
            selected_hypothesis = self.particles[self.current_trial_selected_particle_index].hypothesis
        row = {
            "run_number": self.run_number,
            "trial_no": trial_no,
            "action_key": key.id,
            "action_box": box.id,
            "action_pair": f"{key.id}->{box.id}",
            "outcome": outcome,
            "boxes_opened": len(self.env.success_pairs),
            "success_pairs": ";".join(sorted([f"{k}->{b}" for k, b in self.env.success_pairs])),
            "evidence_count": len(self.evidence),
            "theta": self.alpha / (self.alpha + self.beta),
            "ess": self.current_trial_ess,
            "selected_particle_index": self.current_trial_selected_particle_index,
            "selected_particle_name": self.current_trial_selected_particle_name,
            "selected_hypothesis": selected_hypothesis,
            "rejuvenated_this_trial": self.current_trial_rejuvenated,
            "run_aborted": self.run_aborted,
            "run_abort_reason": self.run_abort_reason,
        }

        for i, p in enumerate(self.particles):
            row[f"particle_{i}_name"] = p.name
            row[f"particle_{i}_weight"] = p.weight
            row[f"particle_{i}_invalid_before_refine"] = self.current_trial_invalid_before_refine[i]
            row[f"particle_{i}_refine_attempts"] = self.current_trial_refine_counts[i]
            row[f"particle_{i}_refine_status"] = self.current_trial_refine_status[i]

        self.trial_rows.append(row)

    def _write_run_csv(self) -> str:
        os.makedirs(self.csv_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.csv_dir, f"run_{self.run_number}_{stamp}.csv")
        if not self.trial_rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["run_number", "trial_no", "note"])
                writer.writerow([self.run_number, 0, "no_trial_rows_recorded"])
            return path

        fieldnames = list(self.trial_rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trial_rows)
        return path

    def run(self, max_trials: int, run_number: int = 1) -> Dict:
        self.run_number = run_number
        self.run_aborted = False
        self.run_abort_reason = ""
        self.trial_rows = []
        self.history = []
        self.trial_count = 0

        while not self.env.is_solved() and self.trial_count < max_trials:
            self._reset_trial_flags()

            if self.act_mode == "bed":
                key, box = self._select_action_by_bed()
            elif self.act_mode == "sample":
                key, box = self._select_action_by_sample()
            else:
                raise ValueError(f"Unsupported act_mode: {self.act_mode}")

            outcome = self.env.test_action(key, box)
            self.evidence.append((key, box, outcome))
            if outcome:
                self.succ_count[(key.id, box.id)] += 1
            else:
                self.fail_count[(key.id, box.id)] += 1

            self._compute_theta()
            self._update_particle_weights(key, box, outcome)

            # Separate trigger: always repair non-executable hypotheses every trial.
            self._repair_invalid_particles_for_trial()

            self.trial_count += 1
            self._snapshot_history(self.trial_count, (key.id, box.id))
            self._append_trial_csv_row(self.trial_count, key, box, outcome)

            if self.run_aborted:
                break

        self.csv_path = self._write_run_csv()
        return {
            "history": self.history,
            "csv_path": self.csv_path,
            "run_aborted": self.run_aborted,
            "run_abort_reason": self.run_abort_reason,
            "trials_completed": self.trial_count,
            "solved": self.env.is_solved(),
        }
