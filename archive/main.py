from smc_soc import Engine
from environment import Environment
from smc.generator import Generator
from utils.plotter2 import Plotter2


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        pass


gen_config = {
    "omega": 2.0,
    "prop_random": 0.8,
    "true_prior": 0.2,
    "train": False,
    # Chosen so initial P(number_match) = 0.32 when prop_random = 0.1.
    # With these values: prior_sum=90, so 0.9*(32/90)=0.32.
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 8,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (1, 1),
    "ess_threshold": 0.5,
    'skill': True,
    "mode": "soc",
    "prior": "uniform",
    "model": "gpt-4o", #or "deepseek-chat"
}

max_trials = 70


if __name__ == '__main__':
    num_runs = 100
    trial_counts = []
    target_history = None
    max_n_steps = -1
    logger = Logger(logging=False)

    for _ in range(num_runs):
        environment = Environment(opening_prob=0.9, include_inspect=False)
        generator = Generator(gen_config, environment)
        smc_engine = Engine(smc_config, environment, generator, logger)
        history = smc_engine.run(max_trials=max_trials)
        n_steps = len(history)
        trial_counts.append(n_steps)
        if n_steps > max_n_steps:
            max_n_steps = n_steps
            target_history = history

    alpha0, beta0 = smc_config["init_theta"]
    Plotter2.plot_trials_to_solve_histogram(
        trial_counts,
        title=f"Trials to solve across {num_runs} runs (max_trials={max_trials})",
        alpha0=alpha0,
        beta0=beta0,
        prop_random=gen_config["prop_random"],
        true_prior=gen_config.get("true_prior"),
        show=True,
    )

    if False:# target_history is not None:
        n_steps = len(target_history)
        plotter = Plotter2(target_history)
        plotter.plot_weights_by_name_over_trials(
            title=(
                "Weight per hypothesis name — run with most trials "
                f"({n_steps} steps incl. t=0; max_trials={max_trials})"
            ),
            alpha0=alpha0,
            beta0=beta0,
            prop_random=gen_config["prop_random"],
            true_prior=gen_config.get("true_prior"),
            show=True,
        )