"""
Plot empirical learning trajectories overlaid on (analytical) replicator-
dynamics vector fields.
Two kinds of plot are supported (for 2 action and 3 action (biased rps) games)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from games import MatrixGame
from dynamics import build_2x2_vector_field


# 2x2 plots

def plot_2x2(game: MatrixGame, runs: list[dict], title: str,
             dynamics: str | None = None,
             dyn_kwargs: dict | None = None,
             time_average: bool = True,
             ax: plt.Axes | None = None) -> plt.Axes:
    """Plot 12 time-averaged policy traces in the (x1, y1) unit square.
    Optionally overlays the replicator-dynamics vector field as a quiver.
    time_average=True shows cumulative-mean policy"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Vector field
    if dynamics is not None:
        kw = dict(grid_size=21, alpha=1.0, tau=0.5, kappa=5)
        if dyn_kwargs:
            kw.update(dyn_kwargs)
        X, Y, U, V = build_2x2_vector_field(game.A, game.B,
                                            dynamics=dynamics, **kw)
        # Normalise arrow lengths 
        magnitude = np.hypot(U, V)
        max_mag = magnitude.max() if magnitude.max() > 0 else 1.0
        U_n = U / (magnitude + 1e-9)
        V_n = V / (magnitude + 1e-9)
        ax.quiver(X, Y, U_n, V_n, magnitude / max_mag,
                  cmap="Blues", pivot="mid",
                  scale=30, width=0.0035, alpha=0.85)

    # Empirical traces
    for r, run in enumerate(runs):
        px = run["policies_row"][:, 0]   # P(action 1) for row
        py = run["policies_col"][:, 0]   # P(action 1) for col
        if time_average:
            cum_x = np.cumsum(px) / np.arange(1, len(px) + 1)
            cum_y = np.cumsum(py) / np.arange(1, len(py) + 1)
        else:
            cum_x, cum_y = px, py
        ax.plot(cum_x, cum_y, lw=1.2, alpha=0.85,
                color=plt.cm.tab20(r % 20))
        ax.plot(cum_x[0], cum_y[0], "o", ms=4,
                color=plt.cm.tab20(r % 20), mec="black", mew=0.4)
        ax.plot(cum_x[-1], cum_y[-1], "*", ms=10,
                color=plt.cm.tab20(r % 20), mec="black", mew=0.4)

    # Mark Nash equilibria
    for x_star, y_star in game.nash_equilibria:
        ax.plot(x_star[0], y_star[0], "P", ms=12, color="red",
                mec="black", mew=0.7, zorder=10)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"P(row plays {game.action_names[0]})")
    ax.set_ylabel(f"P(col plays {game.action_names[0]})")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    return ax


# 3-action simplex plot (for Biased RPS)

def _project_simplex(p: np.ndarray) -> np.ndarray:
    """Project a length-3 probability vector to 2D coordinates
    of an equilateral triangle: v0=(Rock), v1=(Paper), v2=(Scissors)."""
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    return p[0] * v0 + p[1] * v1 + p[2] * v2


def plot_simplex(game: MatrixGame, runs: list[dict], title: str,
                 player: str = "row",
                 time_average: bool = True,
                 ax: plt.Axes | None = None) -> plt.Axes:
    """Plot policy traces on the 2-simplex triangle for a 3-action game.
    Nash equilibria are marked with a red cross."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the triangle
    v0, v1, v2 = (np.array([0.0, 0.0]),
                  np.array([1.0, 0.0]),
                  np.array([0.5, np.sqrt(3.0) / 2.0]))
    tri = np.stack([v0, v1, v2, v0])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1)
    labels = game.action_names
    ax.text(v0[0] - 0.05, v0[1] - 0.06, labels[0], fontsize=11, ha="right")
    ax.text(v1[0] + 0.05, v1[1] - 0.06, labels[1], fontsize=11, ha="left")
    ax.text(v2[0], v2[1] + 0.04, labels[2], fontsize=11, ha="center")

    key = "policies_row" if player == "row" else "policies_col"

    for r, run in enumerate(runs):
        traj = run[key]
        if time_average:
            cum = np.cumsum(traj, axis=0) / np.arange(1, len(traj) + 1)[:, None]
        else:
            cum = traj
        pts = np.array([_project_simplex(p) for p in cum])
        ax.plot(pts[:, 0], pts[:, 1], lw=1.2, alpha=0.85,
                color=plt.cm.tab20(r % 20))
        ax.plot(pts[0, 0], pts[0, 1], "o", ms=4,
                color=plt.cm.tab20(r % 20), mec="black", mew=0.4)
        ax.plot(pts[-1, 0], pts[-1, 1], "*", ms=10,
                color=plt.cm.tab20(r % 20), mec="black", mew=0.4)

    # Mark Nash equilibrium
    for x_star, y_star in game.nash_equilibria:
        ne_pt = _project_simplex(x_star if player == "row" else y_star)
        ax.plot(ne_pt[0], ne_pt[1], "P", ms=14, color="red",
                mec="black", mew=0.7, zorder=10, label="Nash equilibrium")

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title)
    return ax
