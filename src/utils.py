"""Utilities: plotting, bookkeeping, schedules, simple helpers."""
from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# global plotting style
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 14
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

def create_results_folder(base_dir="results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path

def visualize_pso_convergence(obj_values, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(obj_values, marker="o", linestyle="-")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.title("PSO Convergence Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def visualize_particle_trajectory(X_history, save_path_prefix):
    """X_history: list/array of shape (n_iter, n_particles, 3) in log10-space."""
    X_history = np.asarray(X_history)
    init = X_history[0]
    final = X_history[-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = ["log10(phi)", "log10(alpha)", "log10(alpha_EC)"]
    titles = [
        "log10(phi) vs log10(alpha)",
        "log10(phi) vs log10(alpha_EC)",
        "log10(alpha) vs log10(alpha_EC)",
    ]
    for ax, (i, j), title in zip(axes, pairs, titles):
        for snap in X_history[::max(1, len(X_history) // 10)]:
            ax.scatter(snap[:, i], snap[:, j], alpha=0.1, s=5, color="gray", zorder=2)
        ax.scatter(final[:, i], final[:, j], alpha=0.8, color="red", label="Final", s=10, marker="x", zorder=3)
        ax.scatter(init[:, i], init[:, j], alpha=0.4, color="blue", label="Initial", s=5, marker="o", zorder=1)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_trajectory.png", dpi=150)
    plt.close()

def reflect_boundary(log_x, bounds):
    """Reflect with small jitter if out of bounds. bounds: [(lo,hi),(lo,hi),(lo,hi)]"""
    for k in range(3):
        lo, hi = bounds[k]
        if log_x[k] <= lo:
            log_x[k] = lo + abs(np.random.normal(0, 0.1))
        elif log_x[k] >= hi:
            log_x[k] = hi - abs(np.random.normal(0, 0.1))
    return log_x

def adaptive_temperature(t, T0=1.0, min_T=0.1):
    return max(T0 / np.log(2 + 0.001 * t), min_T)
