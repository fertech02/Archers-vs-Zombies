import numpy as np


class BoltzmannAgent:

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        tau: float = 1.0,
        tau_decay: float = 1.0,       # 1.0 = no decay; try 0.9995
        tau_min: float = 0.01,
        init_q: float = 0.0,
        name: str = "Agent",
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.name = name

        self.Q = np.full(n_actions, init_q, dtype=float)
        self._tau_current = tau
        self.episode = 0

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def policy(self) -> np.ndarray:
        """Softmax policy probabilities."""
        # Subtract max for numerical stability (doesn't change the result)
        q = self.Q - np.max(self.Q)
        exp_q = np.exp(q / self._tau_current)
        return exp_q / exp_q.sum()

    def select_action(self) -> int:
        """Sample action from Boltzmann distribution."""
        probs = self.policy()
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, action: int, reward: float) -> None:
        """Update Q-value for the taken action."""
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def decay_tau(self) -> None:
        """Call once per episode to decay temperature."""
        self._tau_current = max(
            self.tau_min,
            self._tau_current * self.tau_decay
        )
        self.episode += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tau_current(self) -> float:
        return self._tau_current

    def reset(self) -> None:
        self.Q[:] = 0.0
        self._tau_current = self.tau
        self.episode = 0

    def __repr__(self):
        q_str = ", ".join(f"Q[{i}]={v:.4f}" for i, v in enumerate(self.Q))
        return f"{self.name}(τ={self._tau_current:.4f}, α={self.alpha}, {q_str})"