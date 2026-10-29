"""Telemetry extraction and processing for simulation results."""

from typing import Dict
import numpy as np
import pandas as pd
from laptimesim.solver.ocp import OCPResult
from laptimesim.track.canonical import CanonicalTrack


class TelemetryProcessor:
    """Processes raw solver state/control arrays into comprehensive telemetry dataframes."""

    def __init__(self, result: OCPResult, track: CanonicalTrack):
        self.result = result
        self.track = track

    def to_dataframe(self) -> pd.DataFrame:
        """Combine all spatial, temporal, state, control, and telemetry channels into a DataFrame."""
        N = len(self.result.s)
        resampled_track = self.track.resample(N)

        data: Dict[str, np.ndarray] = {
            "s_m": self.result.s,
            "time_s": self.result.t,
            "track_x_m": resampled_track.x,
            "track_y_m": resampled_track.y,
            "track_curvature_rad_m": resampled_track.kappa,
        }

        # Calculate vehicle global trajectory from track centerline + lateral deviation n
        n_dev = self.result.states["n"]
        psi_err = self.result.states["psi"]

        # Approximate vehicle Cartesian coordinates (normal offset from track centerline)
        dx = np.gradient(resampled_track.x, self.result.s)
        dy = np.gradient(resampled_track.y, self.result.s)
        ds = np.hypot(dx, dy)
        ds = np.where(ds < 1e-6, 1e-6, ds)
        # Normal unit vector points to left of travel
        nx = -dy / ds
        ny = dx / ds

        veh_x = resampled_track.x + n_dev * nx
        veh_y = resampled_track.y + n_dev * ny
        data["vehicle_x_m"] = veh_x
        data["vehicle_y_m"] = veh_y

        # Velocity in km/h
        vx = self.result.states["vx"]
        vy = self.result.states["vy"]
        v_total = np.hypot(vx, vy)
        data["speed_m_s"] = v_total
        data["speed_km_h"] = v_total * 3.6

        # Accelerations in G
        g = 9.81
        if "ax" in self.result.telemetry:
            data["ax_g"] = self.result.telemetry["ax"] / g
        if "ay" in self.result.telemetry:
            data["ay_g"] = self.result.telemetry["ay"] / g

        # Controls
        for ctrl_name, ctrl_arr in self.result.controls.items():
            data[f"ctrl_{ctrl_name}"] = ctrl_arr

        # Key States
        for st_name in ["n", "psi", "vx", "vy", "psi_dot", "phi", "theta", "soc", "t_batt"]:
            if st_name in self.result.states:
                data[f"state_{st_name}"] = self.result.states[st_name]

        # Key Telemetry
        for telem_name, telem_arr in self.result.telemetry.items():
            if telem_name not in ["ax", "ay"]:
                data[f"telem_{telem_name}"] = telem_arr

        return pd.DataFrame(data)
