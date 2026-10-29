"""Plotting and visualization generators: GG diagram, Telemetry, and Track Map."""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/server execution
import matplotlib.pyplot as plt
import pandas as pd


class SimulationPlotter:
    """Generates standard engineering plots for vehicle dynamics analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_gg_diagram(self, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Generate GG Diagram (longitudinal vs. lateral acceleration)."""
        fig, ax = plt.subplots(figsize=(7, 7))

        ax_g = self.df.get("ax_g", np.zeros(len(self.df)))
        ay_g = self.df.get("ay_g", np.zeros(len(self.df)))
        speed = self.df.get("speed_km_h", np.zeros(len(self.df)))

        scatter = ax.scatter(ay_g, ax_g, c=speed, cmap="plasma", alpha=0.7, edgecolors="none", s=25)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Speed [km/h]")

        # Draw reference friction circles
        theta = np.linspace(0, 2 * np.pi, 200)
        for r in [0.5, 1.0, 1.5, 2.0]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), "--", color="gray", alpha=0.5, linewidth=0.8)

        ax.set_xlabel("Lateral Acceleration $a_y$ [g] (Left < 0 < Right)")
        ax.set_ylabel("Longitudinal Acceleration $a_x$ [g] (Braking < 0 < Accel)")
        ax.set_title("G-G Acceleration Diagram")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_aspect("equal", adjustable="datalim")

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_telemetry_channels(self, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot stacked telemetry channels vs. track distance s."""
        s = self.df["s_m"]
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # 1. Speed Profile
        axes[0].plot(s, self.df["speed_km_h"], color="crimson", linewidth=1.5, label="Speed")
        axes[0].set_ylabel("Speed [km/h]")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right")

        # 2. Driver Controls
        axes[1].plot(s, self.df.get("ctrl_throttle", np.zeros_like(s)), color="forestgreen", label="Throttle", linewidth=1.2)
        axes[1].plot(s, self.df.get("ctrl_brake_pressure", np.zeros_like(s)), color="red", label="Brake", linewidth=1.2)
        axes[1].set_ylabel("Pedal [-]")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right")

        # 3. Steering Angle
        delta_deg = np.rad2deg(self.df.get("ctrl_delta_steer", np.zeros_like(s)))
        axes[2].plot(s, delta_deg, color="royalblue", label="Steering Angle", linewidth=1.2)
        axes[2].set_ylabel("Steer [deg]")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right")

        # 4. Energy Profile & Power
        if "telem_P_elec" in self.df:
            p_kw = self.df["telem_P_elec"] / 1000.0
            axes[3].plot(s, p_kw, color="darkorange", label="Battery Power [kW]", linewidth=1.2)
            axes[3].set_ylabel("Power [kW]")
        if "state_soc" in self.df:
            ax_soc = axes[3].twinx()
            ax_soc.plot(s, self.df["state_soc"] * 100.0, "--", color="purple", label="SoC [%]")
            ax_soc.set_ylabel("SoC [%]")
        axes[3].set_xlabel("Track Distance $s$ [m]")
        axes[3].grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    def plot_track_racing_line(self, save_path: Optional[str | Path] = None) -> plt.Figure:
        """Plot track centerline vs. optimal vehicle racing line."""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(self.df["track_x_m"], self.df["track_y_m"], "k--", linewidth=1.0, alpha=0.5, label="Track Centerline")
        scatter = ax.scatter(
            self.df["vehicle_x_m"], self.df["vehicle_y_m"],
            c=self.df["speed_km_h"], cmap="turbo", s=15, label="Optimal Racing Line"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Speed [km/h]")

        ax.set_xlabel("Global X [m]")
        ax.set_ylabel("Global Y [m]")
        ax.set_title("Optimal Racing Line & Track Layout")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best")

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig
