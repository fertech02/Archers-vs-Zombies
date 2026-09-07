"""
Analytical replicator-dynamics vector fields for Boltzmann and Lenient
Boltzmann Q-learning 
Used to overlay the theoretical flow on empirical trajectories.
"""

from __future__ import annotations
import numpy as np


# (1) Standard Boltzmann replicator dynamics

def _entropy_term(x: np.ndarray) -> np.ndarray:
    """Return the vector  ln x_i - sum_k x_k ln x_k  (entrywise)."""
    x_safe = np.clip(x, 1e-12, 1.0)
    log_x = np.log(x_safe)
    avg_log = np.sum(x * log_x)
    return log_x - avg_log


def boltzmann_dynamics(x: np.ndarray, y: np.ndarray,
                       A: np.ndarray, B: np.ndarray,
                       alpha: float = 1.0, tau: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Boltzmann replicator vector field (Tuyls et al. Equations 7 8; 
    restated as Equation 10 in Bloembergen et al.)"""
    # Row player payoff vector: (A y)_i  for i = 1..n_row.
    Ay = A @ y
    avg_x = x @ Ay
    selection_x = (alpha / tau) * x * (Ay - avg_x)
    mutation_x = -alpha * x * _entropy_term(x)
    dx = selection_x + mutation_x

    # Column player: with row strategy x, payoffs for col-action j are
    # sum_i B[i,j] * x_i = (B^T x)_j.
    BTx = B.T @ x
    avg_y = y @ BTx
    selection_y = (alpha / tau) * y * (BTx - avg_y)
    mutation_y = -alpha * y * _entropy_term(y)
    dy = selection_y + mutation_y
    return dx, dy


# (2) Lenient utility (Bloembergen et al. 2015, Eq. 11)

def lenient_utility(player_payoff_matrix: np.ndarray,
                    opponent_strategy: np.ndarray,
                    kappa: int) -> np.ndarray:
    """Compute the lenient expected utility u_i for each action i. 
    u_i is the expected maximum payoff over kappa i.i.d. draws
    from the opponent's strategy. 
    """
    n_i, n_j = player_payoff_matrix.shape
    y = opponent_strategy
    u = np.zeros(n_i)
    for i in range(n_i):
        row = player_payoff_matrix[i]                           # length n_j
        total = 0.0
        for j in range(n_j):
            a_ij = row[j]
            # Set of opponent actions k giving a_ik <= a_ij, < a_ij, = a_ij
            mask_le = row <= a_ij + 1e-12
            mask_lt = row <  a_ij - 1e-12
            mask_eq = np.abs(row - a_ij) <= 1e-12
            P_le = float(y[mask_le].sum())
            P_lt = float(y[mask_lt].sum())
            P_eq = float(y[mask_eq].sum())
            if P_eq <= 1e-12:
                continue
            contribution = a_ij * y[j] * ((P_le ** kappa) - (P_lt ** kappa)) / P_eq
            total += contribution
        u[i] = total
    return u


def lenient_boltzmann_dynamics(x: np.ndarray, y: np.ndarray,
                               A: np.ndarray, B: np.ndarray,
                               alpha: float = 1.0, tau: float = 0.5,
                               kappa: int = 5
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Vector field of Lenient Boltzmann replicator dynamics.
    Same as boltzmann_dynamics but with expected payoff replaced
    by lenient utility """

    u = lenient_utility(A, y, kappa)
    avg_x = x @ u
    selection_x = (alpha / tau) * x * (u - avg_x)
    mutation_x = -alpha * x * _entropy_term(x)
    dx = selection_x + mutation_x

    # For the column player we view its payoffs as a row player would, by
    # transposing B (col-player's payoff for its own action j against
    # opponent action i is B[i, j] so the matrix from the col player's POV
    # is B.T of shape (n_col_actions, n_row_actions))
    w = lenient_utility(B.T, x, kappa)
    avg_y = y @ w
    selection_y = (alpha / tau) * y * (w - avg_y)
    mutation_y = -alpha * y * _entropy_term(y)
    dy = selection_y + mutation_y
    return dx, dy


# Helpers to build a vector field on a 2D grid for 2-action games

def build_2x2_vector_field(A: np.ndarray, B: np.ndarray,
                           grid_size: int = 21,
                           dynamics: str = "boltzmann",
                           alpha: float = 1.0, tau: float = 0.5,
                           kappa: int = 5
                           ) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Evaluate the vector field on a grid_size x grid_size grid
    in (x1, y1) in [0,1]^2. Returns (X, Y, U, V) ready for quiver."""
    assert A.shape == (2, 2) and B.shape == (2, 2)
    xs = np.linspace(0.02, 0.98, grid_size)
    ys = np.linspace(0.02, 0.98, grid_size)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            x = np.array([X[i, j], 1.0 - X[i, j]])
            y = np.array([Y[i, j], 1.0 - Y[i, j]])
            if dynamics == "boltzmann":
                dx, dy = boltzmann_dynamics(x, y, A, B, alpha=alpha, tau=tau)
            elif dynamics == "lenient":
                dx, dy = lenient_boltzmann_dynamics(x, y, A, B,
                                                   alpha=alpha, tau=tau,
                                                   kappa=kappa)
            else:
                raise ValueError(f"unknown dynamics '{dynamics}'")
            U[i, j] = dx[0]
            V[i, j] = dy[0]
    return X, Y, U, V


if __name__ == "__main__":
    # sanity check : Boltzmann dynamics evaluated at uniform strategy in
    # the stag hunt should point towards the Pareto-optimal (S,S) corner.
    from games import STAG_HUNT
    x = np.array([0.5, 0.5])
    y = np.array([0.5, 0.5])
    dx, dy = boltzmann_dynamics(x, y, STAG_HUNT.A, STAG_HUNT.B, tau=0.5)
    print(f"Stag Hunt @ uniform: dx={dx},  dy={dy}")

    # Lenient version
    dx_l, dy_l = lenient_boltzmann_dynamics(x, y, STAG_HUNT.A, STAG_HUNT.B,
                                             tau=0.5, kappa=5)
    print(f"Stag Hunt (kappa=5) @ uniform: dx={dx_l},  dy={dy_l}")
