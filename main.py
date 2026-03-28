"""
main.py — Task 2: ε-greedy, Boltzmann, and Lenient Boltzmann Q-learning
on Stag Hunt. Produces learning trajectory plots and replicator field plots.

Usage:
    python main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from MatrixGame import StagHunt
from EpsilonAgent import EpsilonGreedyAgent
from BoltzmannAgent import BoltzmannAgent
from LenientAgent import LenientBoltzmannAgent
from Training import TrainConfig, train, convergence_stats
from Plot import (
    plot_learning_trajectories,
    plot_q_values,
    plot_replicator_with_traces,
    print_summary,
)

N_EPISODES = 10_000
SEED = 42


def make_agents(kind: str, n_actions: int):
    """Factory: return (agent1, agent2) for the given algorithm."""
    if kind == "epsilon_greedy":
        return (
            EpsilonGreedyAgent(n_actions, alpha=0.1, epsilon=0.5,
                               epsilon_decay=0.9995, epsilon_min=0.01, name="P1"),
            EpsilonGreedyAgent(n_actions, alpha=0.1, epsilon=0.5,
                               epsilon_decay=0.9995, epsilon_min=0.01, name="P2"),
        )
    elif kind == "boltzmann":
        return (
            BoltzmannAgent(n_actions, alpha=0.1, tau=1.0,
                           tau_decay=0.9995, tau_min=0.01, name="P1"),
            BoltzmannAgent(n_actions, alpha=0.1, tau=1.0,
                           tau_decay=0.9995, tau_min=0.01, name="P2"),
        )
    elif kind == "lenient_boltzmann":
        return (
            LenientBoltzmannAgent(n_actions, alpha=0.1, tau=1.0,
                                  tau_decay=0.9995, tau_min=0.01,
                                  kappa=0.5, kappa_decay=0.9995,
                                  kappa_min=0.0, name="P1"),
            LenientBoltzmannAgent(n_actions, alpha=0.1, tau=1.0,
                                  tau_decay=0.9995, tau_min=0.01,
                                  kappa=0.5, kappa_decay=0.9995,
                                  kappa_min=0.0, name="P2"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {kind}")


if __name__ == "__main__":
    game = StagHunt()
    print(game)
    print()

    algorithms = {
        "ε-greedy":          "epsilon_greedy",
        "Boltzmann":         "boltzmann",
        "Lenient Boltzmann": "lenient_boltzmann",
    }

    all_histories = {}

    for label, kind in algorithms.items():
        print(f"Running: {label}")
        a1, a2 = make_agents(kind, game.n_actions)
        cfg = TrainConfig(n_episodes=N_EPISODES, seed=SEED)
        history = train(game, a1, a2, cfg)
        stats = convergence_stats(history, game, window=200)
        print_summary(stats, game)
        all_histories[label] = history

        # Per-algorithm plots
        plot_q_values(
            history, game, agent_idx=1, save=True,
            filename=f"q_values_{kind}.png",
        )
        plot_replicator_with_traces(
            {label: history}, game, n_grid=20, save=True,
            filename=f"replicator_{kind}.png",
        )

    # Combined trajectory plot (all 3 algorithms, one subplot each)
    plot_learning_trajectories(
        all_histories, game, window=200, save=True,
        filename="all_algorithms_trajectories.png",
    )

    # Combined replicator plot (all 3 traces on one field)
    plot_replicator_with_traces(
        all_histories, game, n_grid=20, save=True,
        filename="replicator_all_algorithms.png",
    )

    print("Done. Plots saved in results/plots/")