"""
visualization/plots.py
──────────────────────
Production-grade plotting for drone telemetry and optimization results.

Contains:
    plot_evolution  — GA convergence and diversity over generations.
    plot_trajectory — PID pitch response and control effort.
    compare_pids    — Overlay multiple PID runs for comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulator import Trajectory


# ─────────────────────────────────────────────
# Plot Defaults
# ─────────────────────────────────────────────

_STYLE      = "seaborn-v0_8-darkgrid"
_FIGSIZE    = (12, 6)
_DPI        = 120
_REPORT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"


def _ensure_dir():
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────

class Visualizer:
    """Utility for generating research-grade figures."""

    def __init__(
        self,
        save : bool = True,
        show : bool = False,
        dpi  : int  = _DPI,
    ):
        self.save = save
        self.show = show
        self.dpi  = dpi
        plt.style.use(_STYLE)

    # ── GA Evolution ──────────────────────────

    def plot_evolution(
        self,
        best_scores      : list[float],
        mean_scores      : list[float],
        diversity_scores : list[float],
        title            : str = "GA Optimization Progress",
    ):
        """Visualizes convergence (best/mean fitness) and population diversity."""
        _ensure_dir()

        gens = np.arange(1, len(best_scores) + 1)

        fig, ax1 = plt.subplots(figsize=_FIGSIZE, dpi=self.dpi)

        # Fitness lines
        ax1.plot(gens, best_scores, "b-",  lw=2,   label="Best Fitness")
        ax1.plot(gens, mean_scores, "b--", alpha=0.5, label="Mean Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness (Lower is Better)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(True, alpha=0.3)

        # Diversity on twin axis
        ax2 = ax1.twinx()
        ax2.fill_between(gens, diversity_scores, color="gray", alpha=0.1)
        ax2.plot(gens, diversity_scores, "k:", alpha=0.4, label="Diversity")
        ax2.set_ylabel("Population Diversity (Std)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.title(title)
        fig.tight_layout()

        if self.save:
            path = _REPORT_DIR / "ga_evolution.png"
            plt.savefig(path)
            print(f"[Visualizer] Saved → {path}")

        if self.show:
            plt.show()

        plt.close()

    # ── Trajectory ────────────────────────────

    def plot_trajectory(
        self,
        trj   : Trajectory,
        kp    : float = 0.0,
        ki    : float = 0.0,
        kd    : float = 0.0,
        title : str   = "PID Controller Response",
    ):
        """Plots Pitch Angle and Control Effort over time."""
        _ensure_dir()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 8), dpi=self.dpi
        )

        # Pitch
        ax1.plot(trj.time, trj.pitch, "r-", lw=2, label="Current Pitch")
        ax1.axhline(0,                      color="k",      linestyle="--", alpha=0.5, label="Setpoint")
        ax1.axhline( trj.cfg.stable_zone_deg, color="orange", linestyle=":",            label="Stable Zone")
        ax1.axhline(-trj.cfg.stable_zone_deg, color="orange", linestyle=":")
        ax1.set_ylabel("Angle (deg)")
        ax1.set_title(f"{title} | Kp={kp:.2f} Ki={ki:.2f} Kd={kd:.2f}")
        ax1.legend(loc="upper right")

        # Control effort
        ax2.plot(trj.time, trj.control, "g-", label="Control Effort (u)")
        ax2.set_ylabel("Actuator Output")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper right")

        fig.tight_layout()

        if self.save:
            path = _REPORT_DIR / "trajectory_run.png"
            plt.savefig(path)
            print(f"[Visualizer] Saved → {path}")

        if self.show:
            plt.show()

        plt.close()

    # ── PID Comparison ────────────────────────

    def compare_pids(
        self,
        trajectories : list[Trajectory],
        labels       : list[str],
        title        : str = "PID Comparison",
    ):
        """Overlays multiple PID runs on one chart."""
        _ensure_dir()

        plt.figure(figsize=_FIGSIZE, dpi=self.dpi)

        for trj, label in zip(trajectories, labels):
            plt.plot(trj.time, trj.pitch, label=label)

        plt.axhline(0, color="k", ls="--", label="Setpoint")
        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (deg)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        if self.save:
            path = _REPORT_DIR / "pid_comparison.png"
            plt.savefig(path)
            print(f"[Visualizer] Saved → {path}")

        if self.show:
            plt.show()

        plt.close()