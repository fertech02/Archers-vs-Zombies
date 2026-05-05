import numpy as np
from dataclasses import dataclass, field
import MatrixGame
from EpsilonAgent import EpsilonGreedyAgent


@dataclass
class TrainConfig:
    n_episodes: int = 10_000
    window: int = 100
    seed: int = 42


@dataclass
class RunHistory:

    actions1: list = field(default_factory=list)
    actions2: list = field(default_factory=list)
    rewards1: list = field(default_factory=list)
    rewards2: list = field(default_factory=list)
    q1: list = field(default_factory=list)   # snapshot of full Q-table per ep
    q2: list = field(default_factory=list)
    eps: list = field(default_factory=list)

    def append(self, a1, a2, r1, r2, Q1, Q2, eps):
        self.actions1.append(a1)
        self.actions2.append(a2)
        self.rewards1.append(r1)
        self.rewards2.append(r2)
        self.q1.append(Q1.copy())
        self.q2.append(Q2.copy())
        self.eps.append(eps)

    def to_numpy(self):
        return {
            "actions1": np.array(self.actions1),
            "actions2": np.array(self.actions2),
            "rewards1": np.array(self.rewards1),
            "rewards2": np.array(self.rewards2),
            "q1": np.array(self.q1),
            "q2": np.array(self.q2),
            "eps": np.array(self.eps),
        }


def run_episode(
    game: MatrixGame,
    agent1: EpsilonGreedyAgent,
    agent2: EpsilonGreedyAgent,
) -> tuple[int, int, float, float]:
    """Run a single episode: select actions, get rewards, update Q-tables."""
    a1 = agent1.select_action()
    a2 = agent2.select_action()
    r1, r2 = game.step(a1, a2)
    agent1.update(a1, r1)
    agent2.update(a2, r2)
    return a1, a2, r1, r2


def _exploration_param(agent) -> float:
    """Read the agent's current exploration parameter (ε for ε-greedy, τ for Boltzmann)."""
    if hasattr(agent, "eps_current"):
        return agent.eps_current
    if hasattr(agent, "tau_current"):
        return agent.tau_current
    return 0.0


def _decay(agent) -> None:
    """Call whichever decay method the agent exposes."""
    if hasattr(agent, "decay_epsilon"):
        agent.decay_epsilon()
    elif hasattr(agent, "decay_tau"):
        agent.decay_tau()
    elif hasattr(agent, "decay"):
        agent.decay()


def train(
    game: MatrixGame,
    agent1,
    agent2,
    cfg: TrainConfig = TrainConfig(),
) -> RunHistory:

    np.random.seed(cfg.seed)
    agent1.reset()
    agent2.reset()

    history = RunHistory()

    for ep in range(cfg.n_episodes):
        a1, a2, r1, r2 = run_episode(game, agent1, agent2)
        history.append(a1, a2, r1, r2, agent1.Q, agent2.Q, _exploration_param(agent1))
        _decay(agent1)
        _decay(agent2)

    return history


def convergence_stats(history: RunHistory, game: MatrixGame, window: int = 100) -> dict:
    """
    Compute rolling joint-action frequencies and time-averaged Q-values.
    Useful to check convergence to Nash / Pareto.
    """
    data = history.to_numpy()
    n = len(data["actions1"])

    # Rolling frequency of each joint action in last `window` episodes
    joint = data["actions1"] * game.n_actions + data["actions2"]
    n_joint = game.n_actions ** 2
    freqs = np.zeros((n, n_joint))
    for ep in range(n):
        start = max(0, ep - window + 1)
        window_slice = joint[start:ep+1]
        for j in range(n_joint):
            freqs[ep, j] = np.mean(window_slice == j)

    # Time-averaged Q-values
    avg_q1 = np.cumsum(data["q1"], axis=0) / np.arange(1, n+1)[:, None]
    avg_q2 = np.cumsum(data["q2"], axis=0) / np.arange(1, n+1)[:, None]

    return {
        "joint_freqs": freqs,        # shape (n_episodes, n_actions^2)
        "avg_q1": avg_q1,
        "avg_q2": avg_q2,
        "final_q1": data["q1"][-1],
        "final_q2": data["q2"][-1],
        "final_eps": data["eps"][-1],
    }