"""
3 independent multi-agent RL algorithms for stateless (matrix) games:

    1- EpsilonGreedyQLearner       : epsilon-greedy Q-learning
    2- BoltzmannQLearner           : softmax/Boltzmann Q-learning
    3- LenientBoltzmannQLearner    : lenient Boltzmann Q-learning

All algorithms implement the same minimal interface:
    - select_action() -> int        : sample an action from the current policy
    - update(action, reward)        : update internal state given the observed reward for the action taken
    - policy() -> np.ndarray        : return the current policy as a probability  vector over actions

All share the same stateless Q-update: Q(a) <- Q(a) + alpha * (r - Q(a))
See report Section 2.2 for details and citations
"""

from __future__ import annotations
import numpy as np


# ------- Base class

class QLearnerBase:
    """Tabular Q-learner for a stateless game with `n_actions` actions."""

    def __init__(self, n_actions: int, alpha: float = 0.05,
                 q_init: float = 0.0, rng: np.random.Generator | None = None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.Q = np.full(n_actions, q_init, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        # Stateless Q-update: Q(a) <- Q(a) + alpha * (r - Q(a))
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def policy(self) -> np.ndarray:
        raise NotImplementedError


# -------- (a) epsilon-greedy Q-learning

class EpsilonGreedyQLearner(QLearnerBase):
    """Q-learner with epsilon-greedy action selection.
    Plays argmax Q(a) with prob 1-eps, uniform random with prob eps.""" 

    def __init__(self, n_actions: int, alpha: float = 0.05,
                 epsilon: float = 0.1, q_init: float = 0.0,
                 rng: np.random.Generator | None = None):
        super().__init__(n_actions, alpha, q_init, rng)
        self.epsilon = epsilon

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # break ties uniformly at random
        max_q = self.Q.max()
        candidates = np.flatnonzero(self.Q >= max_q - 1e-12)
        return int(self.rng.choice(candidates))

    def policy(self) -> np.ndarray:
        pi = np.full(self.n_actions, self.epsilon / self.n_actions)
        max_q = self.Q.max()
        candidates = np.flatnonzero(self.Q >= max_q - 1e-12)
        pi[candidates] += (1.0 - self.epsilon) / len(candidates)
        return pi


# -------- (b) Boltzmann (softmax) Q-learning

class BoltzmannQLearner(QLearnerBase):
    """Q-learner with Boltzmann (softmax) action selection.
    pi(a) = exp(Q(a)/tau) / sum_b exp(Q(b)/tau). High tau = more exploration."""

    def __init__(self, n_actions: int, alpha: float = 0.05,
                 temperature: float = 0.5, q_init: float = 0.0,
                 rng: np.random.Generator | None = None):
        super().__init__(n_actions, alpha, q_init, rng)
        self.tau = temperature

    def policy(self) -> np.ndarray:
        # Numerically stable softmax
        z = self.Q / self.tau
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def select_action(self) -> int:
        return int(self.rng.choice(self.n_actions, p=self.policy()))


# -------- (c) Lenient Boltzmann Q-learning

class LenientBoltzmannQLearner(BoltzmannQLearner):
    """Boltzmann Q-learner with kappa-reward buffer (Panait et al. 2008).
    Q is updated with the max of the last kappa rewards, ignoring
    miscoordination penalties caused by a still-exploring partner."""

    def __init__(self, n_actions: int, alpha: float = 0.05,
                 temperature: float = 1.0, kappa: int = 5,
                 tau_decay: float = 1.0,
                 tau_min: float = 0.05,
                 q_init: float = 0.0,
                 rng: np.random.Generator | None = None):
        super().__init__(n_actions, alpha, temperature, q_init, rng)
        self.kappa = kappa
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        # Per-action reward buffers
        self._buffers: list[list[float]] = [[] for _ in range(n_actions)]

    def update(self, action: int, reward: float) -> None:
        self._buffers[action].append(reward)
        if len(self._buffers[action]) >= self.kappa:
            r_max = max(self._buffers[action])
            self.Q[action] += self.alpha * (r_max - self.Q[action])
            self._buffers[action] = []
            # Cool down temperature one step
            self.tau = max(self.tau * self.tau_decay, self.tau_min)
