import random
from typing import List, Optional

from environment import Environment
from llm.llm import LLM
from llm.code import execute_hypothesis_code


class LlmPS:

    def __init__(
        self,
        env: Environment,
        llm: LLM,
        logger,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        self.env = env
        self.llm = llm
        self.logger = logger

        self.temperature = temperature
        self.max_tokens = max_tokens

        self.hypothesis: Optional[str] = None
        self.evidence = list()
        self.evidence_lines: List[str] = []

    def _is_executable_hypothesis(self, hypothesis: Optional[str]) -> bool:
        """
        Local robustness check only inside llm_ps.py.
        It rejects LLM outputs that do not define callable predict(key, box).
        """
        if not hypothesis:
            return False

        namespace = {}
        try:
            exec(hypothesis, namespace)
            return callable(namespace.get("predict"))
        except Exception:
            return False

    def _select_action(self):
        """
        Select next opening action.
        Randomly select a key-box action consistent with the hypothesis.
        If none exists, randomly select an action.
        """
        opened = set([pair[1] for pair in self.env.success_pairs])
        candidate_actions = []
        fallback_actions = []

        for (key, box) in self.env.actions:
            if box.id not in opened:
                if execute_hypothesis_code(self.hypothesis, key, box) is True:
                    candidate_actions.append((key, box))
                else:
                    fallback_actions.append((key, box))

        if candidate_actions:
            return random.choice(candidate_actions)
        return random.choice(fallback_actions)

    def _accept_h(self) -> bool:
        # Reject bad LLM output before execute_hypothesis_code can crash.
        if not self._is_executable_hypothesis(self.hypothesis):
            return False

        # Check consistency with opened boxes.
        for (key_id, box_id) in self.env.success_pairs:
            key, box = self.env.id_to_key[key_id], self.env.id_to_box[box_id]
            if execute_hypothesis_code(self.hypothesis, key, box) is False:
                return False

        # Failure evidence intentionally disabled for stochastic oracle.
        return True

    def _bad_hypothesis_result(self, reason: str) -> dict:
        return {
            "solved": self.env.is_solved(),
            "trials": self.trial_count,
            "opened": len(self.env.success_pairs),
            "success_pairs": list(self.env.success_pairs),
            "history": self.history,
            "aborted": True,
            "abort_reason": reason,
            "last_hypothesis": self.hypothesis,
        }

    def run(self, max_trials: int) -> dict:
        self.trial_count = 0
        self._interaction_seq = 0
        self.history = []

        MAX_REFINE_RETRIES = 3

        while not self.env.is_solved() and self.trial_count < max_trials:
            self.logger.log(f"TRIAL {self.trial_count}")

            if self.evidence_lines:
                self.logger.log("Evidence lines (included in prompt):")
                for i, line in enumerate(self.evidence_lines, start=1):
                    self.logger.log(f"{i}. {line}")
            else:
                self.logger.log("Evidence lines (included in prompt): (none)")

            refine_attempts = 0

            if self.trial_count == 0:
                self.hypothesis, h_name = self.llm.generate([])

                while not self._accept_h():
                    if refine_attempts >= MAX_REFINE_RETRIES:
                        return self._bad_hypothesis_result(
                            "invalid_hypothesis_after_generate_refine_cap"
                        )

                    self.logger.log(
                        f"Generated hypothesis rejected; refining "
                        f"{refine_attempts + 1}/{MAX_REFINE_RETRIES}"
                    )

                    self.hypothesis, h_name = self.llm.refine(
                        self.evidence,
                        self.hypothesis or "",
                    )
                    refine_attempts += 1

            else:
                while True:
                    self.hypothesis, new_name = self.llm.refine(
                        self.evidence,
                        self.hypothesis or "",
                    )

                    if self._accept_h():
                        break

                    if refine_attempts >= MAX_REFINE_RETRIES:
                        return self._bad_hypothesis_result(
                            "invalid_hypothesis_after_refine_cap"
                        )

                    self.logger.log(
                        f"Refined hypothesis rejected; retrying "
                        f"{refine_attempts + 1}/{MAX_REFINE_RETRIES}"
                    )

                    refine_attempts += 1

            self.logger.log(f"LLM response:\n{self.hypothesis}")
            self.logger.log(f"Hypothesis:\n{self.hypothesis}")

            key, box = self._select_action()
            outcome = self.env.test_action(key, box)

            self.evidence.append((key, box, outcome))
            status = "success" if outcome is True else "failure"
            self.evidence_lines.append(
                f"Open box {box.id} with key {key.id}: {status}"
            )

            self.logger.log(f"Action chosen: ({key.id}, {box.id})")
            self.logger.log(f"Outcome: {outcome}")
            self.logger.log(f"Boxes opened: {self.env.success_pairs}")

            self.trial_count += 1
            self._interaction_seq += 1

            self.history.append(
                {
                    "t": self._interaction_seq,
                    "opened": len(self.env.success_pairs),
                    "hypothesis": self.hypothesis,
                    "action": [key.id, box.id],
                    "outcome": bool(outcome),
                    "llm_response": self.hypothesis,
                    "evidence_lines": list(self.evidence_lines),
                    "refine_attempts": refine_attempts,
                    "accepted_after_refine": True,
                }
            )

        return {
            "solved": self.env.is_solved(),
            "trials": self.trial_count,
            "opened": len(self.env.success_pairs),
            "success_pairs": list(self.env.success_pairs),
            "history": self.history,
            "aborted": False,
            "abort_reason": None,
        }