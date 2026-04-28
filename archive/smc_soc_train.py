from collections import defaultdict
from environment import Environment, Key, Box
from typing import List, Dict, Any

from smc.generator import Generator

import math, random, copy


class Particle:

    def __init__(self, name: str, hypothesis: Dict, weight: float, prior: float):
        self.name = name
        self.hypothesis = hypothesis
        self.prior = prior
        self.weight = weight

    def evaluate(self, key: Key, box: Box) -> bool:
        return (self.hypothesis.get(key.id) == box.id)


class Engine:

    def __init__(self, config: Dict, env: Environment, proposal: Generator = None, logger=None):

        self.num_particles: int = config['num_particles']
        self.skill: bool = config['skill']

        # initialize theta distribution
        self.alpha0, self.beta0 = config['init_theta']
        self.alpha, self.beta = config['init_theta']

        self.ess_threshold: float = config['ess_threshold']

        self.env: Environment = env
        self.proposal: Generator = proposal
        self.logger = logger

        self.particles = self._initialize_particles()

        # for theta update
        self.succ_count = defaultdict(lambda: 0)
        self.fail_count = defaultdict(lambda: 0)
        self.evidence = list()

        self.train = bool(config.get("train", False))
        self.history = []
        self.trial_count = 0

    def _initialize_particles(self) -> List[Particle]:
        particles = list()

        sampled = self.proposal.sample_from_dist(self.num_particles)
        for i in range(self.num_particles):
            name, h_type, prior = sampled[i]
            if h_type == 'generator':
                hypothesis, name = self.proposal.generate()
            else:
                hypothesis = self.proposal.hypotheses[name]

            particles.append(
                Particle(
                    name=name,
                    hypothesis=hypothesis,
                    weight=(1.0 / self.num_particles),
                    prior=prior
                )
            )
        return particles

    def _history_snapshot(self) -> Dict:
        probs: Dict[str, float] = {}
        for p in self.particles:
            probs[p.name] = probs.get(p.name, 0.0) + p.weight
        return {
            "probs": probs,
            "weights_by_name": dict(probs),
            "particles_named": [
                {"name": p.name, "weight": float(p.weight)} for p in self.particles
            ],
            "particle_weights": [float(p.weight) for p in self.particles],
            "particle_names": [p.name for p in self.particles],
        }

    def _particle_weights_dict(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for i, p in enumerate(self.particles, start=1):
            out[f"particle_{i}"] = {
                "name": p.name,
                "weight": float(p.weight),
            }
        return out

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
                weight=(1.0 / self.num_particles),
                prior=self.particles[i].prior
            )
            resampled.append(new_particle)
        self.particles = resampled

    def _rejuvenate(self):

        def _compute_h_likelihood(hypothesis: Dict) -> float:
            likelihood = 1.0
            for key, box, outcome in self.evidence:
                pred_match = (hypothesis.get(key.id) == box.id)
                likelihood = likelihood * self._compute_likelihood(pred_match, outcome)
            return likelihood

        new_particles = copy.deepcopy(self.particles)

        for i in range(self.num_particles):
            orig_p = new_particles[i]

            name, h_type, prior = self.proposal.sample()
            if h_type == 'generator':
                new_h, name = self.proposal.generate()
            else:
                new_h = self.proposal.hypotheses[name]

            new_h_likelihood = _compute_h_likelihood(new_h)
            orig_h_likelihood = _compute_h_likelihood(orig_p.hypothesis)

            denom = orig_h_likelihood * orig_p.prior
            if denom == 0:
                accept_prob = 0
            else:
                accept_prob = min(1, (new_h_likelihood * prior) / denom)

            if random.random() <= accept_prob:
                new_particles[i] = Particle(
                    name=name,
                    hypothesis=new_h,
                    weight=new_particles[i].weight,
                    prior=prior
                )

        self.particles = new_particles

    def _compute_ess(self) -> float:
        weights = [p.weight for p in self.particles]
        if sum(weights) == 0:
            return 0.0
        return 1.0 / sum(w**2 for w in weights)

    def _compute_entropy(self, particle_weights: List[float]) -> float:
        weights = [w for w in particle_weights if w > 0]
        if len(weights) <= 1:
            return 0.0
        return -1.0 * sum(w * math.log2(w) for w in weights)

    def _compute_inspect_info_gain(self, box: Box) -> float:
        """
        calculate information gain by inspect box action
        """
        pass

    def _compute_theta(self):
        self.alpha, self.beta = self.alpha0, self.beta0

        open_prob = defaultdict(lambda: 0.0)
        for (key, box, _) in self.evidence:
            if (key.id, box.id) in open_prob:
                continue
            open_prob[(key.id, box.id)] = sum(
                p.weight for p in self.particles if p.evaluate(key, box)
            )

        for kb_pair in open_prob.keys():
            self.alpha += open_prob[kb_pair] * self.succ_count[kb_pair]
            self.beta += open_prob[kb_pair] * self.fail_count[kb_pair]

        self.alpha = max(1e-9, self.alpha)
        self.beta = max(1e-9, self.beta)

    def _compute_likelihood(self, predict: bool, outcome: bool) -> float:
        if self.skill:
            assert (self.alpha + self.beta > 0)
            prob_success = self.alpha / (self.alpha + self.beta)
        else:
            prob_success = 1.0

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
            particle.weight = (
                particle.weight / total_weight
                if total_weight > 0 else
                (1.0 / self.num_particles)
            )

        ess = self._compute_ess()
        if ess < self.num_particles * self.ess_threshold:
            self._resample()
            self._rejuvenate()

    def _get_candidate_actions(self):
        actions = self.env.actions
        opened = {box_id for _, box_id in self.env.success_pairs}

        candidate_actions = []
        for (key, box) in actions:
            if key != 'inspect' and box.id in opened:
                continue
            candidate_actions.append((key, box))
        return candidate_actions

    def _get_all_action_ig(self) -> Dict[str, float]:
        action_ig = {}
        for (key, box) in self._get_candidate_actions():
            if key == 'inspect':
                info_gain = self._compute_inspect_info_gain(box)
                action_name = f"inspect->{box.id}"
            else:
                info_gain = self._compute_info_gain(key, box)
                action_name = f"{key.id}->{box.id}"
            action_ig[action_name] = float(info_gain)
        return action_ig

    def _select_action(self):
        actions = self.env.actions
        opened = {box_id for _, box_id in self.env.success_pairs}

        h_weights = defaultdict(lambda: 0.0)
        for p in self.particles:
            h_weights[p.name] += p.weight

        if sum(h_weights.values()) > 0:
            best_name, _ = max(h_weights.items(), key=lambda kv: kv[1])

            masses = [m for m in h_weights.values() if m > 0]
            total_mass = sum(masses)
            entropy = self._compute_entropy([float(m / total_mass) for m in masses])

            if entropy <= 1e-2 and best_name in self.proposal.hypotheses:
                hyp = self.proposal.hypotheses[best_name]
                greedy_candidates = []
                for key in self.env.keys:
                    box_id = hyp.get(key.id, None)
                    if box_id is None:
                        continue
                    if box_id in opened:
                        continue
                    box = self.env.id_to_box[box_id]
                    greedy_candidates.append((key, box))
                if greedy_candidates:
                    return random.choice(greedy_candidates)

        max_info_gain = float('-inf')
        best_actions = list()

        candidate_actions = []
        for (key, box) in actions:
            if key != 'inspect' and box.id in opened:
                continue
            candidate_actions.append((key, box))

        for (key, box) in candidate_actions:
            if key == 'inspect':
                info_gain = self._compute_inspect_info_gain(box)
            else:
                info_gain = self._compute_info_gain(key, box)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_actions = [(key, box)]
            elif info_gain == max_info_gain:
                best_actions.append((key, box))

        eps = 1e-9
        if not best_actions:
            best_actions = candidate_actions

        if max_info_gain <= eps and candidate_actions:
            best_open_prob = float('-inf')
            exploit_actions = []
            for (key, box) in candidate_actions:
                if key == 'inspect':
                    continue
                open_prob = sum(
                    p.weight for p in self.particles if p.evaluate(key, box)
                )
                if open_prob > best_open_prob:
                    best_open_prob = open_prob
                    exploit_actions = [(key, box)]
                elif open_prob == best_open_prob:
                    exploit_actions.append((key, box))
            if exploit_actions:
                return random.choice(exploit_actions)

        return random.choice(best_actions)

    def _get_chosen_action_name(self) -> str:
        key, box = self._select_action()
        if key == "inspect":
            return f"inspect->{box.id}"
        return f"{key.id}->{box.id}"

    def step(self, key: Key, box: Box, outcome: bool):
        """
        Replay one externally supplied observation into the model.
        This updates the same belief state pieces used by run().
        """

        # Keep environment aligned with observed successful openings.
        if outcome:
            self.env.success_pairs.add((key.id, box.id))
            self.env.opened.add(box.id)

        self.evidence.append((key, box, outcome))

        if outcome:
            self.proposal.prune_proposal_dist(key, box)
            self.succ_count[(key.id, box.id)] += 1
        else:
            self.fail_count[(key.id, box.id)] += 1

        if self.skill:
            self._compute_theta()

        self._update_particle_weights(key, box, outcome)

        self.trial_count += 1

        theta = self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.0
        snap = self._history_snapshot()
        self.history.append({
            "t": self.trial_count,
            "opened": len(self.env.success_pairs),
            "theta": theta,
            "action": (key.id, box.id),
            **snap,
        })

    def replay_actions(self, action_sequence: List[Dict[str, Any]], run_name: str = "run_1") -> Dict[str, Any]:
        """
        Replay child actions through the engine.

        Expected action_sequence format:
        [
            {"key": "red", "box": "red", "outcome": True},
            {"key": "grey2", "box": "pink", "outcome": False},
            ...
        ]

        Snapshot for trial_t is recorded BEFORE applying that child's action,
        so it reflects the model state used to evaluate the child's next move.
        """

        self.trial_count = 0
        self.history = []

        trial_major_trace: Dict[str, Any] = {}

        for t, step_info in enumerate(action_sequence, start=1):
            key_id = step_info["key"]
            box_id = step_info["box"]
            outcome = bool(step_info["outcome"])

            key = self.env.id_to_key[key_id]
            box = self.env.id_to_box[box_id]

            all_action_ig = self._get_all_action_ig()
            chosen_action = self._get_chosen_action_name()
            particle_weights = self._particle_weights_dict()
            theta = self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.0

            trial_name = f"trial_{t}"
            trial_major_trace.setdefault(trial_name, {})
            trial_major_trace[trial_name][run_name] = {
                "particle_weights": particle_weights,
                "all_action_ig": all_action_ig,
                "chosen_action": chosen_action,
                "kid_action": f"{key_id}->{box_id}",
                "kid_outcome": outcome,
                "theta": float(theta),
                "opened_boxes": sorted(list(self.env.opened)),
            }

            self.step(key, box, outcome)

        return trial_major_trace

    def run(self, max_trials: int) -> bool:
        self.trial_count = 0
        self.history = []

        opened = len(self.env.success_pairs)
        theta = self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.0
        snap0 = self._history_snapshot()
        self.history.append({
            "t": 0,
            "opened": opened,
            "theta": theta,
            "action": None,
            **snap0,
        })

        while not self.env.is_solved() and self.trial_count < max_trials:

            self.logger.log(f"TRIAL {self.trial_count + 1}")

            (key, box) = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))
            if outcome is True:
                self.proposal.prune_proposal_dist(key, box)
                self.succ_count[(key.id, box.id)] += 1
            else:
                self.fail_count[(key.id, box.id)] += 1

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")

            if self.skill:
                self._compute_theta()

            self._update_particle_weights(key, box, outcome)

            self.logger.log(f"partcle ids: {[p.name for p in self.particles]}")
            self.logger.log(f"number opened: {len(self.env.success_pairs)}")

            self.trial_count += 1

            t = self.trial_count
            opened = len(self.env.success_pairs)
            theta = self.alpha / (self.alpha + self.beta)
            snap = self._history_snapshot()
            if key == "inspect":
                action_pair = ("inspect", box.id)
            else:
                action_pair = (key.id, box.id)

            self.history.append({
                "t": t,
                "opened": opened,
                "theta": theta,
                "action": action_pair,
                **snap,
            })

        return self.history