import numpy as np

class MatrixGame:
    """Base class for two-player matrix games."""

    def __init__(self, payoff_matrix: np.ndarray, action_names: list[str], name: str):

        self.payoff = payoff_matrix
        self.action_names = action_names
        self.n_actions = len(action_names)
        self.name = name

    def step(self, a1: int, a2: int) -> tuple[float, float]:
        r1, r2 = self.payoff[a1, a2]
        return float(r1), float(r2)

    def action_name(self, idx: int) -> str:
        return self.action_names[idx]

    def __repr__(self):
        lines = [f"Game: {self.name}", "Payoff matrix (r1, r2):"]
        header = "       " + "  ".join(f"P2:{n:>4}" for n in self.action_names)
        lines.append(header)
        for i, n in enumerate(self.action_names):
            row = f"P1:{n:>3}  " + "  ".join(
                f"({self.payoff[i,j,0]:.2f},{self.payoff[i,j,1]:.2f})"
                for j in range(self.n_actions)
            )
            lines.append(row)
        return "\n".join(lines)


class StagHunt(MatrixGame):

    def __init__(self):
        payoff = np.array([
            [(1.0,   1.0 ), (0.0,   2/3)],
            [(2/3,   0.0 ), (2/3,   2/3)],
        ])
        super().__init__(payoff, action_names=["S", "H"], name="Stag Hunt")

class SubsidyGame(MatrixGame):

    def __init__(self):
        payoff = np.array([
            [(12.0, 12.0), (0.0, 11.0)],
            [(11.0, 0.0), (10.0, 10.0)],
        ])
        super().__init__(payoff, action_names=["S1","S2"], name="Subsidy Game")

class PrisonerDilemma(MatrixGame):

    def __init__(self):
        payoff = np.array([
            [(-1.0, -1.0), (-4.0, 0.0)],
            [(0.0, -4.0), (-3.0, -3.0)]
        ])
        super().__init__(payoff, action_names=["C", "D"], name="Prisoner's Dilemma")

class BiasedRockPaperScissor(MatrixGame):

    def __init__(self):
        payoff = np.array([
            [(0.0,0.0), (-0.05,0.05), (0.25, -0.25)],
            [(0.05,-0.05), (0.0,0.0), (-0.5, 0.5)],
            [(-0.25, 0.25), (0.5,-0.5), (0.0, 0.0)]
        ]
        )
        super().__init__(payoff, action_names=["R","P","S"], name="Biased Rock-Paper-Scissor")

