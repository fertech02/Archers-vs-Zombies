"""
Run all Task-2 experiments and save the figures.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from games import STAG_HUNT, SUBSIDY, PRISONERS_DILEMMA, BIASED_RPS, ALL_GAMES
from algorithms import (EpsilonGreedyQLearner, BoltzmannQLearner,
                        LenientBoltzmannQLearner)
from training import run_many_trajectories
from plotting import plot_2x2, plot_simplex
from dynamics import build_2x2_vector_field


# Experiment configuration

# Per-algorithm hyper-parameters.
CONFIG = {
    "epsilon_greedy": dict(alpha=0.05, epsilon=0.1),
    "boltzmann":      dict(alpha=0.05, temperature=0.5),
    "lenient":        dict(alpha=0.05, temperature=1.0,
                           kappa=5, tau_decay=0.999, tau_min=0.1),
}

N_TRAJ      = 12          # number of empirical trajectories per algo
N_STEPS     = 30_000      # length of each trajectory in interactions
RECORD_EVERY = 50         # snapshot the policy every this many steps

ALG_FACTORIES = {
    "epsilon_greedy":
        lambda n_actions, rng: EpsilonGreedyQLearner(
            n_actions, rng=rng, **CONFIG["epsilon_greedy"]),
    "boltzmann":
        lambda n_actions, rng: BoltzmannQLearner(
            n_actions, rng=rng, **CONFIG["boltzmann"]),
    "lenient":
        lambda n_actions, rng: LenientBoltzmannQLearner(
            n_actions, rng=rng, **CONFIG["lenient"]),
}

ALG_TITLES = {
    "epsilon_greedy": "epsilon-greedy QL",
    "boltzmann":      "Boltzmann QL",
    "lenient":        "Lenient Boltzmann QL",
}

# Which dynamics to overlay for each algorithm.
DYN_OVERLAY = {
    "epsilon_greedy": None,
    "boltzmann":      "boltzmann",
    "lenient":        "lenient",
}


# Helper

def run_one_game(game, fname_stem: str, outdir: str):
    """Run all three algorithms on `game` and produce a 1x3 figure of policy
    traces, plus a 1x2 figure of analytical vector fields for Boltzmann and
    Lenient Boltzmann.
    """
    print(f"\n=== {game.name} ===")

    # Run experiments
    runs_per_alg = {}
    for alg_name in ALG_FACTORIES:
        print(f"  running {alg_name}...", flush=True)
        factory = lambda rng, name=alg_name: ALG_FACTORIES[name](
            game.n_actions, rng)
        runs_per_alg[alg_name] = run_many_trajectories(
            game=game,
            make_agent=factory,
            n_trajectories=N_TRAJ,
            n_steps=N_STEPS,
            record_every=RECORD_EVERY,
            seed=42,
        )

    # Plot
    if game.n_actions == 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        for ax, alg in zip(axes, ALG_FACTORIES):
            dyn = DYN_OVERLAY[alg]
            dyn_kw = None
            if dyn == "lenient":
                dyn_kw = dict(tau=CONFIG["lenient"]["temperature"],
                              kappa=CONFIG["lenient"]["kappa"])
            elif dyn == "boltzmann":
                dyn_kw = dict(tau=CONFIG["boltzmann"]["temperature"])
            plot_2x2(game, runs_per_alg[alg],
                     title=f"{game.name}: {ALG_TITLES[alg]}",
                     dynamics=dyn, dyn_kwargs=dyn_kw, ax=ax)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        for ax, alg in zip(axes, ALG_FACTORIES):
            plot_simplex(game, runs_per_alg[alg],
                         title=f"{game.name}: {ALG_TITLES[alg]}",
                         player="row", ax=ax)

    fig.suptitle(f"Empirical policy traces, {game.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{fname_stem}.png"), dpi=130)
    plt.close(fig)


def plot_pure_dynamics(outdir: str):
    """Plot just the analytical vector fields for both
    Boltzmann and Lenient Boltzmann dynamics on the three 2x2 games.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    games_2x2 = [STAG_HUNT, SUBSIDY, PRISONERS_DILEMMA]

    for col, game in enumerate(games_2x2):
        for row, (dyn_name, kw) in enumerate(
            [("boltzmann", dict(tau=0.5)),
             ("lenient",   dict(tau=0.5, kappa=5))]):
            ax = axes[row, col]
            X, Y, U, V = build_2x2_vector_field(
                game.A, game.B, grid_size=22, dynamics=dyn_name, **kw)
            mag = np.hypot(U, V)
            max_mag = mag.max() if mag.max() > 0 else 1.0
            ax.quiver(X, Y, U / (mag + 1e-9), V / (mag + 1e-9),
                      mag / max_mag, cmap="Blues", pivot="mid",
                      scale=30, width=0.0045)
            for x_star, y_star in game.nash_equilibria:
                ax.plot(x_star[0], y_star[0], "P", ms=11, color="red",
                        mec="black", mew=0.7)
            kind = "Boltzmann" if dyn_name == "boltzmann" else "Lenient"
            ax.set_title(f"{game.name} -- {kind} dynamics")
            ax.set_xlabel(f"P(row plays {game.action_names[0]})")
            ax.set_ylabel(f"P(col plays {game.action_names[0]})")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Replicator-dynamics vector fields ", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dynamics_only.png"), dpi=130)
    plt.close(fig)


# Entry-point

if __name__ == "__main__":
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)

    run_one_game(STAG_HUNT, "stag_hunt", outdir)
    run_one_game(SUBSIDY, "subsidy", outdir)
    run_one_game(PRISONERS_DILEMMA, "prisoners_dilemma", outdir)
    run_one_game(BIASED_RPS, "biased_rps", outdir)

    plot_pure_dynamics(outdir)

    print("\nAll figures saved in", outdir)
