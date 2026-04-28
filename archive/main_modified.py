"""
Single SMC run: plot E[theta] over trials and per-particle weights over trials.
"""
from smc_soc import Engine
from environment import Environment
from smc.generator import Generator
from utils.plotter import Plotter2


class Logger:
    def __init__(self, logging: bool = True):
        self.logging = logging

    def log(self, log_str: str):
        if self.logging:
            print(log_str)


gen_config = {
    "omega": 2.0,
    "prop_random": 0.1,
    "true_prior": 0.2,
    "train": False,
    "prior_color": 20,
    "prior_order": 20,
    "prior_shape": 10,
    "prior_number": 32,
    "prior_sim_color_total": 8,
}

smc_config = {
    "num_particles": 30,
    "init_theta": (19, 1),
    "ess_threshold": 0.5,
    "skill": True,
    "mode": "soc",
    "prior": "uniform",
    "model": "gpt-4o",
}

max_trials = 70

if __name__ == "__main__":
    environment = Environment(include_inspect=False)
    generator = Generator(gen_config, environment)
    logger = Logger(logging=False)

    smc_engine = Engine(smc_config, environment, generator, logger)
    history = smc_engine.run(max_trials=max_trials)

    plotter = Plotter2(history)
    plotter.plot_theta_over_trials(
        title="E[theta] over trials (single run; t=0 = before first action)",
        show=True,
    )
    # Total mass per hypothesis name (sum of particle weights); t=0 is initial draw.
    plotter.plot_weights_by_name_over_trials(
        title="Weight per hypothesis name (single run; t=0 = initial, before first action)",
        show=True,
    )
    # Optional: per particle slot (index), not aggregated by name.
    plotter.plot_particle_weights_over_trials(
        title="Particle weights by slot index (single run; t=0 = initial)",
        show=True,
    )
