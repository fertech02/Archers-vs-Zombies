import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "results" / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_trajectories(
    history_dict: dict,          # {"run_name": RunHistory}
    game,
    window: int = 100,
    save: bool = True,
    filename: str = "trajectories.png",
):
    """
    Plot time-averaged joint action frequencies for multiple runs.
    This is the "empirical policy trace" asked by the project.
    """
    n_joint = game.n_actions ** 2
    action_labels = [
        f"({game.action_name(i)},{game.action_name(j)})"
        for i in range(game.n_actions)
        for j in range(game.n_actions)
    ]
    colors = ["#378ADD", "#D85A30", "#3B6D11", "#993556"]

    fig, axes = plt.subplots(1, len(history_dict), figsize=(6 * len(history_dict), 4), squeeze=False)

    for col, (run_name, history) in enumerate(history_dict.items()):
        ax = axes[0][col]
        data = history.to_numpy()
        n = len(data["actions1"])
        joint = data["actions1"] * game.n_actions + data["actions2"]

        for j in range(n_joint):
            freq = np.array([
                np.mean(joint[max(0, ep-window):ep+1] == j)
                for ep in range(n)
            ])
            ax.plot(freq, color=colors[j % len(colors)], label=action_labels[j], linewidth=1.5)

        ax.set_title(run_name, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Freq. (rolling {window} ep.)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(f"{game.name} — Learning trajectories", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR / filename}")
    plt.show()


def plot_q_values(
    history,
    game,
    agent_idx: int = 1,
    save: bool = True,
    filename: str = "q_values.png",
):
    """Plot Q-value evolution over training."""
    data = history.to_numpy()
    q_key = f"q{agent_idx}"
    Q = data[q_key]   # shape (n_episodes, n_actions)
    n = Q.shape[0]
    episodes = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#378ADD", "#D85A30", "#3B6D11", "#993556", "#7B4FBF"]
    for a in range(game.n_actions):
        ax.plot(episodes, Q[:, a], color=colors[a % len(colors)],
                label=f"Q[{game.action_name(a)}]", linewidth=1.5)

    ax.set_title(f"{game.name} — Q-values Player {agent_idx}", fontsize=11)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR / filename}")
    plt.show()


def plot_replicator_with_traces(
    histories: dict,
    game,
    n_grid: int = 20,
    save: bool = True,
    filename: str = "replicator_field.png",
):
    """
    Plot replicator dynamics vector field overlaid with empirical policy traces.
    Matches the style of Figure 1e in the project PDF.

    Works for 2-action games. x1 = prob(action 0) for P1, x2 = prob(action 0) for P2.
    Replicator equation: dx_i/dt = x_i * (f_i - f_avg)
    where f_i = fitness of action i given opponent's mixed strategy.
    """
    A = game.payoff[:, :, 0]   # P1 payoff matrix, shape (n_actions, n_actions)
    B = game.payoff[:, :, 1]   # P2 payoff matrix

    def replicator(x1, x2):
        """
        x1, x2 in [0,1]: prob of playing action 0 for P1 and P2.
        Returns (dx1, dx2) — replicator dynamics direction.
        """
        p1 = np.array([x1, 1 - x1])
        p2 = np.array([x2, 1 - x2])

        f1 = A @ p2          # fitness of each action for P1
        f1_avg = p1 @ f1
        dx1 = x1 * (f1[0] - f1_avg)

        f2 = B.T @ p1        # fitness of each action for P2
        f2_avg = p2 @ f2
        dx2 = x2 * (f2[0] - f2_avg)

        return dx1, dx2

    # --- Build grid ---
    eps = 1e-6
    xs = np.linspace(eps, 1 - eps, n_grid)
    X1, X2 = np.meshgrid(xs, xs)
    DX1, DX2 = np.vectorize(replicator)(X1, X2)

    # Normalize arrows (direction only, like the reference figure)
    norm = np.sqrt(DX1**2 + DX2**2) + 1e-12
    UX = DX1 / norm
    UY = DX2 / norm

    fig, ax = plt.subplots(figsize=(5, 5))

    # Quiver field (dashed-style arrows, gray like the reference)
    ax.quiver(
        X1, X2, UX, UY,
        color="gray", alpha=0.5,
        scale=28, width=0.003,
        headwidth=3, headlength=3,
    )

    # --- Empirical policy traces ---
    colors = ["black", "#378ADD", "#D85A30", "#3B6D11"]
    for (run_name, history), color in zip(histories.items(), colors):
        data = history.to_numpy()
        # Time-averaged policy: prob(action 0) at each episode
        traj1 = np.cumsum(data["actions1"] == 0) / np.arange(1, len(data["actions1"]) + 1)
        traj2 = np.cumsum(data["actions2"] == 0) / np.arange(1, len(data["actions2"]) + 1)

        # Subsample for clean plot (every 10 episodes after warmup)
        warmup = 50
        step = max(1, len(traj1) // 300)
        idx = np.arange(warmup, len(traj1), step)

        ax.plot(traj1[idx], traj2[idx], color=color, linewidth=1.2,
                label=run_name, alpha=0.9)
        # Mark start and end
        ax.plot(traj1[warmup], traj2[warmup], "o", color=color, markersize=4)
        ax.plot(traj1[-1], traj2[-1], "s", color=color, markersize=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"$x_1$ — P('{game.action_name(0)}') — Player 1", fontsize=10)
    ax.set_ylabel(f"$x_2$ — P('{game.action_name(0)}') — Player 2", fontsize=10)
    ax.set_title(f"{game.name} — Replicator dynamics + policy traces", fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {SAVE_DIR / filename}")
    plt.show()


def print_summary(stats: dict, game, agent_names=("Player 1", "Player 2")):
    """Print a readable convergence summary."""
    n_actions = game.n_actions
    action_labels = [
        f"({game.action_name(i)},{game.action_name(j)})"
        for i in range(n_actions)
        for j in range(n_actions)
    ]
    freqs = stats["joint_freqs"][-1]   # final rolling frequencies

    print(f"\n{'='*50}")
    print(f"  {game.name} — Convergence summary")
    print(f"{'='*50}")
    print(f"  Final ε:  {stats['final_eps']:.4f}")
    print()
    print(f"  Final Q-values {agent_names[0]}: " +
          ", ".join(f"{game.action_name(a)}={v:.4f}" for a, v in enumerate(stats["final_q1"])))
    print(f"  Final Q-values {agent_names[1]}: " +
          ", ".join(f"{game.action_name(a)}={v:.4f}" for a, v in enumerate(stats["final_q2"])))
    print()
    print("  Joint action frequencies (last window):")
    for label, freq in zip(action_labels, freqs):
        bar = "█" * int(freq * 30)
        print(f"    {label}  {freq:.2%}  {bar}")
    print(f"{'='*50}\n")