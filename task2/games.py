"""
Payoff matrices, Nash equilibria, and Pareto-optimal states (pre-computed, they were 
derived analytically in the report section 2.1) 
for the 4 matrix games: Stag Hunt, Subsidy Game, Prisoner's Dilemma, and 
Biased Rock-Paper-Scissors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MatrixGame:

    name: str  #name of the game
    #row-player payoff matrix (n_Actions_row, n_actions_col)
    A: np.ndarray

    #col-player payoff matrix, same shape as A              
    B: np.ndarray

    action_names: list[str] #list of action labels
    #list of (x*, y*) pure or mixed equilibria, where x* is the row-player's mixed strategy as a numpy array and y* is the column-player's.
    nash_equilibria: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # list of (i, j) joint action indices that are Pareto-optimal.
    pareto_optimal: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_actions(self) -> int:
        return self.A.shape[0]


# Stag Hunt
#       S        H
# S   1, 1    0, 2/3
# H  2/3, 0   2/3, 2/3
#
# Pure NEs: (S,S) and (H,H). Mixed NE: each player plays S with probability 2/5 (see report).
# Pareto-optimal: (S,S) gives (1,1) which Pareto-dominates (H,H) at (2/3, 2/3).
STAG_HUNT = MatrixGame(
    name="Stag Hunt",
    A=np.array([[1.0, 0.0], [2/3, 2/3]]),
    B=np.array([[1.0, 2/3], [0.0, 2/3]]),
    action_names=["Stag", "Hare"],
    nash_equilibria=[
        (np.array([1.0, 0.0]), np.array([1.0, 0.0])),     # (S,S)
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),     # (H,H)
        (np.array([2/5, 3/5]), np.array([2/5, 3/5])),     # mixed
    ],
    pareto_optimal=[(0, 0)],                              # only (S,S)
)


# Subsidy Game
#       S1       S2
# S1  12, 12   0, 11
# S2  11, 0    10, 10
#
# Pure NEs: (S1,S1) and (S2,S2). Mixed NE: (10/11, 1/11). PO: (S1,S1) only.
# (S2,S2) is risk-dominant but Pareto-suboptimal.
SUBSIDY = MatrixGame(
    name="Subsidy Game",
    A=np.array([[12.0, 0.0], [11.0, 10.0]]),
    B=np.array([[12.0, 11.0], [0.0, 10.0]]),
    action_names=["S1", "S2"],
    nash_equilibria=[
        (np.array([1.0, 0.0]), np.array([1.0, 0.0])),       # (S1,S1)
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),       # (S2,S2)
        (np.array([10/11, 1/11]), np.array([10/11, 1/11])), # mixed
    ],
    pareto_optimal=[(0, 0)],                                # only (S1,S1)
)


# Prisoner's Dilemma
#        C        D
# C   -1, -1   -4, 0
# D    0, -4   -3, -3
#
# D strictly dominates C for both players. Unique NE: (D,D).
# PO profiles: (C,C), (C,D), (D,C). (D,D) is the only non-PO outcome.
PRISONERS_DILEMMA = MatrixGame(
    name="Prisoner's Dilemma",
    A=np.array([[-1.0, -4.0], [0.0, -3.0]]),
    B=np.array([[-1.0, 0.0], [-4.0, -3.0]]),
    action_names=["Cooperate", "Defect"],
    nash_equilibria=[
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),     # (D,D)
    ],
    pareto_optimal=[(0, 0), (0, 1), (1, 0)],              # all except (D,D)
)


# Biased Rock-Paper-Scissors
# Row-player payoff matrix:
#       R       P       S
# R     0    -0.05    0.25
# P   0.05     0     -0.5
# S  -0.25   0.5      0
#
# This is a zero-sum game 
# No pure NE. Unique mixed NE: (10/16, 5/16, 1/16). PO is degenerate. (see report)
A_BRPS = np.array([
    [ 0.00, -0.05,  0.25],
    [ 0.05,  0.00, -0.50],
    [-0.25,  0.50,  0.00],
])

BIASED_RPS = MatrixGame(
    name="Biased Rock-Paper-Scissors",
    A=A_BRPS,
    B=-A_BRPS,                                            # zero-sum
    action_names=["Rock", "Paper", "Scissors"],
    nash_equilibria=[
        (np.array([10/16, 5/16, 1/16]),
         np.array([10/16, 5/16, 1/16])),
    ],
    pareto_optimal=[],
)


ALL_GAMES = [STAG_HUNT, SUBSIDY, PRISONERS_DILEMMA, BIASED_RPS]


def sample_payoff(game: MatrixGame, action_row: int, action_col: int
                  ) -> tuple[float, float]:
    """Return the joint payoff (r_row, r_col) for the joint action."""
    return float(game.A[action_row, action_col]), float(game.B[action_row, action_col])


if __name__ == "__main__":
    # Sanity check
    for g in ALL_GAMES:
        print(f"\n=== {g.name} ===")
        print(f"A =\n{g.A}")
        print(f"B =\n{g.B}")
        print(f"Nash equilibria:")
        for k, (x, y) in enumerate(g.nash_equilibria):
            print(f"  NE{k}: x*={np.round(x, 4).tolist()}  "
                  f"y*={np.round(y, 4).tolist()}")
        print(f"Pareto-optimal pure profiles (row-idx, col-idx): "
              f"{g.pareto_optimal}")
