from environment import Environment
from llm.llm import LLM
from smc_sp import Engine
from utils.plotter2 import Plotter2


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


smc_config = {
    "num_particles": 30,
    "init_theta": (19, 1),
    "ess_threshold": 0.5,
    # action selection mode:
    # - "bed" = Bayesian experimental design (info gain)
    # - "sample" = sample a particle then act greedily
    "act_mode": "bed",
}


if __name__ == "__main__":
    num_runs = 10
    max_trials = 70

    trial_counts = []
    target_history = None
    max_n_steps = -1

    logger = Logger(logging=False)
    llm = LLM()

    for _ in range(num_runs):
        env = Environment(opening_prob=0.5, include_inspect=False)
        engine = Engine(smc_config, env, llm, logger)
        history = engine.run(max_trials=max_trials)

        n_steps = len(history)
        trial_counts.append(n_steps)
        if n_steps > max_n_steps:
            max_n_steps = n_steps
            target_history = history

    alpha0, beta0 = smc_config["init_theta"]
    Plotter2.plot_trials_to_solve_histogram(
        trial_counts,
        title=f"SMC-SP: trials to solve across {num_runs} runs (max_trials={max_trials})",
        alpha0=alpha0,
        beta0=beta0,
        show=True,
    )

    if False: #$target_history is not None:
        n_steps = len(target_history)
        plotter = Plotter2(target_history)
        plotter.plot_weights_by_name_over_trials(
            title=(
                "SMC-SP: weight per hypothesis name — run with most trials "
                f"({n_steps} steps; max_trials={max_trials})"
            ),
            alpha0=alpha0,
            beta0=beta0,
            show=True,
        )

