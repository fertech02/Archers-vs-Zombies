import numpy as np


class LenientBoltzmannAgent:

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        tau: float = 1.0,
        tau_decay: float = 1.0,
        tau_min: float = 0.01,
        kappa: float = 0.5,           # leniency parameter
        kappa_decay: float = 1.0,     # 1.0 = no decay; try 0.9995
        kappa_min: float = 0.0,
        init_q: float = 0.0,
        name: str = "Agent",
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.kappa = kappa
        self.kappa_decay = kappa_decay
        self.kappa_min = kappa_min
        self.name = name

        self.Q = np.full(n_actions, init_q, dtype=float)
        self._tau_current = tau
        self._kappa_current = kappa
        self.episode = 0
        self._n_updates = np.zeros(n_actions, dtype=int)   # for diagnostics
        self._n_skipped = np.zeros(n_actions, dtype=int)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def policy(self) -> np.ndarray:
        """Softmax policy (same as standard Boltzmann)."""
        q = self.Q - np.max(self.Q)
        exp_q = np.exp(q / self._tau_current)
        return exp_q / exp_q.sum()

    def select_action(self) -> int:
        """Sample action from Boltzmann distribution."""
        probs = self.policy()
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, action: int, reward: float) -> None:
        """
        Lenient update: only apply if reward passes the leniency threshold.
        θ ~ Exp(κ), update only if r >= Q[a] - θ.
        """
        if self._kappa_current > 0:
            theta = np.random.exponential(self._kappa_current)
        else:
            theta = 0.0   # κ=0 → always update (standard Boltzmann)

        if reward >= self.Q[action] - theta:
            self.Q[action] += self.alpha * (reward - self.Q[action])
            self._n_updates[action] += 1
        else:
            self._n_skipped[action] += 1

    def decay(self) -> None:
        """Decay both temperature τ and leniency κ. Call once per episode."""
        self._tau_current = max(self.tau_min, self._tau_current * self.tau_decay)
        self._kappa_current = max(self.kappa_min, self._kappa_current * self.kappa_decay)
        self.episode += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tau_current(self) -> float:
        return self._tau_current

    @property
    def kappa_current(self) -> float:
        return self._kappa_current

    @property
    def skip_rate(self) -> np.ndarray:
        """Fraction of updates skipped per action (diagnostic)."""
        total = self._n_updates + self._n_skipped
        return np.where(total > 0, self._n_skipped / total, 0.0)

    def reset(self) -> None:
        self.Q[:] = 0.0
        self._tau_current = self.tau
        self._kappa_current = self.kappa
        self.episode = 0
        self._n_updates[:] = 0
        self._n_skipped[:] = 0

    def __repr__(self):
        q_str = ", ".join(f"Q[{i}]={v:.4f}" for i, v in enumerate(self.Q))
        return (f"{self.name}(τ={self._tau_current:.4f}, "
                f"κ={self._kappa_current:.4f}, α={self.alpha}, {q_str})")