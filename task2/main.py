"""
main.py — Task 2: ε-greedy, Boltzmann, and Lenient Boltzmann Q-learning
on Stag Hunt. Produces learning trajectory plots and replicator field plots.

Usage:
    python main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from MatrixGame import StagHunt, SubsidyGame, PrisonerDilemma, BiasedRockPaperScissor
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

GAMES = [StagHunt(), SubsidyGame(), PrisonerDilemma(), BiasedRockPaperScissor()]


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


def slug(name: str) -> str:
    return name.lower().replace("'", "").replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    algorithms = {
        "ε-greedy":          "epsilon_greedy",
        "Boltzmann":         "boltzmann",
        "Lenient Boltzmann": "lenient_boltzmann",
    }

    for game in GAMES:
        print("\n" + "#" * 60)
        print(f"# Game: {game.name}")
        print("#" * 60)
        print(game)
        print()

        all_histories = {}
        gslug = slug(game.name)

        for label, kind in algorithms.items():
            print(f"Running: {label}")
            a1, a2 = make_agents(kind, game.n_actions)
            cfg = TrainConfig(n_episodes=N_EPISODES, seed=SEED)
            history = train(game, a1, a2, cfg)
            stats = convergence_stats(history, game, window=200)
            print_summary(stats, game)
            all_histories[label] = history

            plot_q_values(
                history, game, agent_idx=1, save=True,
                filename=f"q_values_{gslug}_{kind}.png",
            )
            if game.n_actions == 2:
                plot_replicator_with_traces(
                    {label: history}, game, n_grid=20, save=True,
                    filename=f"replicator_{gslug}_{kind}.png",
                )

        plot_learning_trajectories(
            all_histories, game, window=200, save=True,
            filename=f"trajectories_{gslug}.png",
        )

        if game.n_actions == 2:
            plot_replicator_with_traces(
                all_histories, game, n_grid=20, save=True,
                filename=f"replicator_{gslug}_all.png",
            )

    print("Done. Plots saved in results/plots/")