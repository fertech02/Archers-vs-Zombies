import numpy as np

class EpsilonGreedyAgent:

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,   # 1.0 = no decay; try 0.9995 for decay
        epsilon_min: float = 0.01,
        init_q: float = 0.0,
        name: str = "Agent",
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.name = name

        self.Q = np.full(n_actions, init_q, dtype=float)
        self._eps_current = epsilon
        self.episode = 0

    def select_action(self) -> int:
        """ε-greedy action selection."""
        if np.random.rand() < self._eps_current:
            return np.random.randint(self.n_actions)          # explore
        return int(np.argmax(self.Q))                          # exploit

    def update(self, action: int, reward: float) -> None:
        """Update Q-value for the taken action."""
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def decay_epsilon(self) -> None:
        """Call once per episode to decay ε."""
        self._eps_current = max(
            self.epsilon_min,
            self._eps_current * self.epsilon_decay
        )
        self.episode += 1

    @property
    def policy(self) -> np.ndarray:
        """
        Return the greedy policy probabilities (for reporting).
        The true policy is ε-greedy, but we report the greedy part.
        """
        probs = np.full(self.n_actions, self._eps_current / self.n_actions)
        probs[np.argmax(self.Q)] += 1.0 - self._eps_current
        return probs

    @property
    def eps_current(self) -> float:
        return self._eps_current

    def reset(self) -> None:
        self.Q[:] = 0.0
        self._eps_current = self.epsilon
        self.episode = 0

    def __repr__(self):
        q_str = ", ".join(f"Q[{i}]={v:.4f}" for i, v in enumerate(self.Q))
        return f"{self.name}(ε={self._eps_current:.4f}, α={self.alpha}, {q_str})"