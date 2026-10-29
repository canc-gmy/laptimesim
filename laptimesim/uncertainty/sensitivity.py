"""Sensitivity visualization generators (Tornado and distribution histogram)."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from laptimesim.uncertainty.monte_carlo import UncertaintyResult


class SensitivityPlotter:
    """Plots Monte Carlo sensitivity tornado charts and confidence distributions."""

    def __init__(self, uq_result: UncertaintyResult):
        self.res = uq_result

    def plot_lap_time_distribution(self, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot histogram of predicted lap times with confidence interval bands."""
        fig, ax = plt.subplots(figsize=(8, 5))

        times = np.array(self.res.lap_times)
        ax.hist(times, bins=15, color="skyblue", edgecolor="black", alpha=0.7, density=True)

        ax.axvline(self.res.mean_lap_time, color="red", linestyle="-", linewidth=1.5, label=f"Mean: {self.res.mean_lap_time:.2f} s")
        ax.axvline(self.res.ci_lower, color="green", linestyle="--", linewidth=1.2, label=f"{self.res.confidence_level*100:.0f}% CI Lower: {self.res.ci_lower:.2f} s")
        ax.axvline(self.res.ci_upper, color="green", linestyle="--", linewidth=1.2, label=f"{self.res.confidence_level*100:.0f}% CI Upper: {self.res.ci_upper:.2f} s")

        ax.set_xlabel("Lap Time [s]")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Predicted Lap Time Uncertainty Distribution (N={len(times)})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_tornado_sensitivity(self, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot Tornado sensitivity chart showing correlation between parameters and lap time."""
        fig, ax = plt.subplots(figsize=(8, 5))

        items = sorted(self.res.parameter_sensitivities.items(), key=lambda kv: abs(kv[1]))
        params = [k for k, _ in items]
        corrs = [v for _, v in items]

        colors = ["salmon" if c > 0 else "lightgreen" for c in corrs]
        y_pos = np.arange(len(params))

        ax.barh(y_pos, corrs, color=colors, edgecolor="black", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.axvline(0.0, color="black", linewidth=0.8)

        ax.set_xlabel("Sensitivity (Pearson Correlation with Lap Time)")
        ax.set_title("Parameter Sensitivity Ranking (Tornado Chart)")
        ax.grid(True, linestyle=":", alpha=0.6)

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig
