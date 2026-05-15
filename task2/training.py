"""
Self-play training loop for two independent Q-learners in a matrix game.
The main function is `train_self_play()`, which runs one trajectory of self-play
training and returns the policy snapshots and final Q-values. 
"""

from __future__ import annotations
import numpy as np
from typing import Callable

from games import MatrixGame
from algorithms import QLearnerBase


# Single trajectory

def train_self_play(game: MatrixGame,
                    make_agent: Callable[[np.random.Generator], QLearnerBase],
                    n_steps: int = 20_000,
                    record_every: int = 50,
                    init_Q: tuple[np.ndarray, np.ndarray] | None = None,
                    seed: int = 0) -> dict:
    """Run two independent Q-learners for n_steps rounds of self-play.
    Records policy snapshots every record_every steps.
    Returns policy trajectories for both players."""
    rng_row = np.random.default_rng(seed)
    rng_col = np.random.default_rng(seed + 10_000)

    row = make_agent(rng_row)
    col = make_agent(rng_col)

    if init_Q is not None:
        row.Q[:] = init_Q[0]
        col.Q[:] = init_Q[1]

    snapshots_row = []
    snapshots_col = []
    snapshots_row.append(row.policy().copy())
    snapshots_col.append(col.policy().copy())

    for t in range(1, n_steps + 1):
        a_row = row.select_action()
        a_col = col.select_action()
        r_row = float(game.A[a_row, a_col])
        r_col = float(game.B[a_row, a_col])
        row.update(a_row, r_row)
        col.update(a_col, r_col)

        if t % record_every == 0:
            snapshots_row.append(row.policy().copy())
            snapshots_col.append(col.policy().copy())

    return {
        "policies_row": np.asarray(snapshots_row),
        "policies_col": np.asarray(snapshots_col),
        "final_row_Q": row.Q.copy(),
        "final_col_Q": col.Q.copy(),
    }


# Many trajectories from many initial conditions

def sample_initial_Q_on_simplex(n_actions: int, rng: np.random.Generator,
                                scale: float = 1.0) -> np.ndarray:
    """Sample Q-values such that the implied softmax policy is
    uniformly distributed over the probability simplex."""
    p = rng.dirichlet(np.ones(n_actions))
    # clamp to avoid log(0)
    p = np.clip(p, 1e-3, None)
    p /= p.sum()
    return scale * np.log(p)


def run_many_trajectories(game: MatrixGame,
                          make_agent: Callable[[np.random.Generator], QLearnerBase],
                          n_trajectories: int = 15,
                          n_steps: int = 20_000,
                          record_every: int = 50,
                          seed: int = 0,
                          q_init_scale: float = 1.0) -> list[dict]:
    """Run n_trajectories independent self-play training runs,
    each starting from a different random initial policy.
    """
    rng = np.random.default_rng(seed)
    runs = []
    for k in range(n_trajectories):
        # Generate random initial policies on the simplex via initial Q-values
        Q_row = sample_initial_Q_on_simplex(game.n_actions, rng, q_init_scale)
        Q_col = sample_initial_Q_on_simplex(game.n_actions, rng, q_init_scale)
        run = train_self_play(
            game=game,
            make_agent=make_agent,
            n_steps=n_steps,
            record_every=record_every,
            init_Q=(Q_row, Q_col),
            seed=int(rng.integers(1, 10**9)),
        )
        runs.append(run)
    return runs
