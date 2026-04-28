from collections import defaultdict
from environment import Environment, Key, Box
from typing import List, Dict
from llm.code import execute_hypothesis_code
from llm.llm_vs2 import LLM

import math
import random


class Particle:
    def __init__(self, name: str, hypothesis: str, weight: float):
        self.name = name
        self.hypothesis = hypothesis
        self.weight = weight

    def evaluate(self, key: Key, box: Box) -> bool:
        return execute_hypothesis_code(self.hypothesis, key, box)


class Engine:
    def __init__(self, config: Dict, env: Environment, llm: LLM = None, logger=None):
        self.num_particles: int = config['num_particles']

        # action selection mode:
        # bed = bayesian experimental design
        # sample = probabilistic sampling
        self.act_mode: str = config['act_mode']

        self.alpha0, self.beta0 = config['init_theta']
        self.alpha, self.beta = config['init_theta']

        self.ess_threshold: float = config['ess_threshold']

        self.env: Environment = env
        self.llm: LLM = llm
        self.logger = logger

        self.last_generate_invalid_count = 0
        self.last_refine_invalid_count = 0

        self.MAX_REJUVENATE_RETRIES = 5

        self.particles = self._initialize_particles()

        self.evidence = list()
        self.succ_count = defaultdict(lambda: 0)
        self.fail_count = defaultdict(lambda: 0)

        self.history = []
        self.aborted = False
        self.abort_reason = ""

    def _initialize_particles(self) -> List[Particle]:
        particles = list()
        for _ in range(self.num_particles):
            hypothesis, name = self.llm.generate(evidence=[])
            self.last_generate_invalid_count = getattr(self.llm, "last_generate_invalid_count", 0)
            particles.append(
                Particle(name=name, hypothesis=hypothesis, weight=(1.0 / self.num_particles))
            )
        return particles

    def _resample(self):
        weights = [p.weight for p in self.particles]

        if sum(weights) == 0:
            indices = random.choices(range(self.num_particles), k=self.num_particles)
        else:
            indices = random.choices(range(self.num_particles), k=self.num_particles, weights=weights)

        resampled = list()
        for i in indices:
            new_particle = Particle(
                name=self.particles[i].name,
                hypothesis=self.particles[i].hypothesis,
                weight=(1.0 / self.num_particles)
            )
            resampled.append(new_particle)
        self.particles = resampled

    def _rejuvenate(self):
        def _compute_h_likelihood(hypothesis: str) -> float:
            likelihood = 1.0
            for key, box, outcome in self.evidence:
                pred_match = execute_hypothesis_code(hypothesis, key, box)
                likelihood *= self._compute_likelihood(pred_match, outcome)
            return likelihood

        def _accept_h(hypothesis: str) -> bool:
            for (key_id, box_id) in self.env.success_pairs:
                key = self.env.id_to_key[key_id]
                box = self.env.id_to_box[box_id]
                if execute_hypothesis_code(hypothesis, key, box) is False:
                    return False
            return True

        likelihoods = [_compute_h_likelihood(p.hypothesis) for p in self.particles]
        worst_value = min(likelihoods)
        worst_indices = [i for i, v in enumerate(likelihoods) if v == worst_value]
        worst_h_idx = random.choice(worst_indices)
        worst_h = self.particles[worst_h_idx].hypothesis

        for rejuvenate_try in range(self.MAX_REJUVENATE_RETRIES):
            print(f"\n[REJUVENATE ATTEMPT {rejuvenate_try + 1}]")

            new_h, new_name = self.llm.refine(self.evidence, worst_h)
            self.last_refine_invalid_count = getattr(self.llm, "last_refine_invalid_count", 0)

            if _accept_h(new_h):
                self.particles[worst_h_idx] = Particle(
                    name=new_name,
                    hypothesis=new_h,
                    weight=self.particles[worst_h_idx].weight
                )
                return

            print("[REJUVENATE REJECTED] valid code but inconsistent with opened boxes")
            worst_h = new_h

        raise RuntimeError(
            f"REJUVENATE_ABORT: exceeded {self.MAX_REJUVENATE_RETRIES} rejuvenation attempts"
        )

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

    def _compute_inspect_info_gain(self, box: Box) -> float:
        pass

    def _compute_theta(self):
        self.alpha, self.beta = self.alpha0, self.beta0

        open_prob = defaultdict(lambda: 0.0)
        for (key, box, _) in self.evidence:
            if (key.id, box.id) not in open_prob:
                open_prob[(key.id, box.id)] = sum(
                    p.weight for p in self.particles if p.evaluate(key, box)
                )

        for kb_pair in open_prob.keys():
            self.alpha += open_prob[kb_pair] * self.succ_count[kb_pair]
            self.beta += open_prob[kb_pair] * self.fail_count[kb_pair]

        self.alpha = max(1e-9, self.alpha)
        self.beta = max(1e-9, self.beta)

    def _compute_likelihood(self, predict: bool, outcome: bool) -> float:
        assert (self.alpha + self.beta > 0)
        prob_success = self.alpha / (self.alpha + self.beta)

        if predict and outcome:
            return prob_success
        elif predict and not outcome:
            return 1.0 - prob_success
        elif not predict and outcome:
            return 0.0
        else:
            return 1.0

    def _compute_info_gain(self, key: Key, box: Box) -> float:
        current_entropy = self._compute_entropy([p.weight for p in self.particles])
        expected_entropy = 0.0

        for outcome in [True, False]:
            outcome_prob = 0.0
            updated_weights = list()

            for particle in self.particles:
                pred_outcome = particle.evaluate(key, box)
                likelihood = self._compute_likelihood(pred_outcome, outcome)
                outcome_prob += particle.weight * likelihood
                updated_weights.append(particle.weight * likelihood)

            if outcome_prob == 0:
                continue

            updated_weights = [w / outcome_prob for w in updated_weights if w > 0]
            new_entropy = self._compute_entropy(updated_weights)
            expected_entropy += outcome_prob * new_entropy

        return current_entropy - expected_entropy

    def _update_particle_weights(self, key: Key, box: Box, outcome: bool):
        for particle in self.particles:
            pred_outcome = particle.evaluate(key, box)
            likelihood = self._compute_likelihood(pred_outcome, outcome)
            particle.weight = particle.weight * likelihood

        total_weight = sum([p.weight for p in self.particles])
        for particle in self.particles:
            particle.weight = (particle.weight / total_weight) if total_weight > 0 else (1.0 / self.num_particles)

        ess = self._compute_ess()

        # keep current behavior: always resample + rejuvenate
        self._resample()
        self._rejuvenate()

    def _select_action_by_sample(self):
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            particle = random.choice(self.particles)
        else:
            particle = random.choices(self.particles, weights=weights, k=1)[0]

        opened = set([pair[1] for pair in self.env.success_pairs])

        candidate_actions = []
        fallback_actions = []

        for (key, box) in self.env.actions:
            if box.id not in opened:
                if particle.evaluate(key, box):
                    candidate_actions.append((key, box))
                else:
                    fallback_actions.append((key, box))

        if len(candidate_actions) > 0:
            action = random.choice(candidate_actions)
        else:
            action = random.choice(fallback_actions)

        return particle, action, candidate_actions, fallback_actions

    def _select_action_by_bed(self):
        max_info_gain = float('-inf')
        best_actions = list()

        for (key, box) in self.env.actions:
            if key == 'inspect':
                info_gain = self._compute_inspect_info_gain(box)
            else:
                if (key.id, box.id) in self.env.success_pairs:
                    continue
                info_gain = self._compute_info_gain(key, box)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_actions = [(key, box)]
            elif info_gain == max_info_gain:
                best_actions.append((key, box))

        chosen_action = random.choice(best_actions)
        return None, chosen_action, best_actions, []

    def run(self, max_trials: int):
        self.trial_count = 0
        self.history = []
        self.aborted = False
        self.abort_reason = ""

        while not self.env.is_solved() and self.trial_count < max_trials:
            self.logger.log(f"TRIAL {self.trial_count + 1}")

            state_before = {
                "opened_count": len(self.env.success_pairs),
                "opened_boxes": sorted([pair[1] for pair in self.env.success_pairs]),
                "success_pairs": sorted([[k, b] for (k, b) in self.env.success_pairs]),
                "theta": self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else None,
            }

            if self.act_mode == 'bed':
                selected_particle, (key, box), candidate_actions, fallback_actions = self._select_action_by_bed()
            elif self.act_mode == 'sample':
                selected_particle, (key, box), candidate_actions, fallback_actions = self._select_action_by_sample()
            else:
                raise ValueError(f"Unknown act_mode: {self.act_mode}")

            outcome = self.env.test_action(key, box)

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")

            self.evidence.append((key, box, outcome))
            if outcome is True:
                self.succ_count[(key.id, box.id)] += 1
            else:
                self.fail_count[(key.id, box.id)] += 1

            self._compute_theta()

            abort_reason = ""
            aborted_now = False

            try:
                self._update_particle_weights(key, box, outcome)
                self.logger.log(f"particle ids: {[p.name for p in self.particles]}")
                self.logger.log(f"number opened: {len(self.env.success_pairs)}")
            except RuntimeError as e:
                aborted_now = True
                abort_reason = str(e)
                self.aborted = True
                self.abort_reason = abort_reason
                self.logger.log(f"[RUN ABORTED INSIDE ENGINE] {abort_reason}")

            self.trial_count += 1

            probs = {}
            for p in self.particles:
                probs[p.name] = probs.get(p.name, 0.0) + p.weight

            state_after = {
                "opened_count": len(self.env.success_pairs),
                "opened_boxes": sorted([pair[1] for pair in self.env.success_pairs]),
                "success_pairs": sorted([[k, b] for (k, b) in self.env.success_pairs]),
                "theta": self.alpha / (self.alpha + self.beta),
            }

            self.history.append({
                "t": self.trial_count,
                "selected_particle_name": selected_particle.name if selected_particle is not None else None,
                "selected_particle_hypothesis": selected_particle.hypothesis if selected_particle is not None else None,
                "action_key": key.id,
                "action_box": box.id,
                "outcome": bool(outcome),
                "candidate_actions": [[k.id, b.id] for (k, b) in candidate_actions],
                "fallback_actions": [[k.id, b.id] for (k, b) in fallback_actions],
                "state_before_opened_count": state_before["opened_count"],
                "state_before_opened_boxes": "|".join(state_before["opened_boxes"]),
                "state_before_success_pairs": str(state_before["success_pairs"]),
                "state_before_theta": state_before["theta"],
                "state_after_opened_count": state_after["opened_count"],
                "state_after_opened_boxes": "|".join(state_after["opened_boxes"]),
                "state_after_success_pairs": str(state_after["success_pairs"]),
                "state_after_theta": state_after["theta"],
                "particle_probs": str(probs),
                "generate_invalid_count": self.last_generate_invalid_count if self.trial_count == 1 else 0,
                "refine_invalid_count": self.last_refine_invalid_count,
                "aborted": aborted_now,
                "abort_reason": abort_reason,
                "abort_trial": self.trial_count if aborted_now else "",
            })

            if aborted_now:
                break

        return self.history