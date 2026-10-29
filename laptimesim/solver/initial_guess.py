"""Initial guess generator for curvilinear optimal control lap time problem."""

from typing import Dict
import numpy as np
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel


class InitialGuessGenerator:
    """Generates physically consistent initial guesses for states and controls."""

    def __init__(self, vehicle: VehicleModel, track: CanonicalTrack):
        self.vehicle = vehicle
        self.track = track

    def generate(self, num_nodes: int) -> Dict[str, np.ndarray]:
        """Generate initial trajectory guesses along the track nodes.

        Returns:
            Dictionary containing:
                - 's': Arc-length coordinates (N,)
                - 'X': State matrix of shape (n_states_ocp, N)
                - 'U': Control matrix of shape (n_controls, N)
        """
        resampled_track = self.track.resample(num_nodes)
        s = resampled_track.s
        kappa = resampled_track.kappa

        # Point mass / friction circle estimate of speed limit in corners
        mu = float(self.vehicle.tire_front.D_lat)
        g = 9.81
        v_max_straight = 35.0  # ~126 km/h typical FSAE speed
        v_min_corner = 8.0

        # Max cornering speed from lateral acceleration limit: v^2 * kappa <= mu * g
        abs_kappa = np.maximum(np.abs(kappa), 1e-4)
        v_corner = np.sqrt((mu * g) / abs_kappa)
        v_profile = np.clip(v_corner, v_min_corner, v_max_straight)

        # Smooth velocity profile
        for _ in range(3):
            v_profile = np.convolve(v_profile, [0.2, 0.6, 0.2], mode="same")

        r_wheel = self.vehicle.r_wheel
        omega_profile = v_profile / r_wheel

        # Ackermann-approximate steering angle for centerline: delta ≈ L * kappa
        delta_profile = self.vehicle.L * kappa

        # Yaw rate for centerline tracking: r ≈ v * kappa
        yaw_rate = v_profile * kappa

        # Total OCP states:
        # [n, psi, vx, vy, vz, phi, theta, psi_dot, phi_dot, theta_dot,
        #  omega_fl, omega_fr, omega_rl, omega_rr,
        #  zs_fl, zs_fr, zs_rl, zs_rr,
        #  dzs_fl, dzs_fr, dzs_rl, dzs_rr,
        #  delta_compl, theta_dt, soc, t_batt]
        n_ocp_states = 2 + len(self.vehicle.state_names())
        X = np.zeros((n_ocp_states, num_nodes))

        # States 0: n (lateral deviation), 1: psi (heading error relative to track)
        X[0, :] = 0.0  # Centerline
        X[1, :] = 0.0  # Aligned with track tangent

        # Vehicle states
        X[2, :] = v_profile            # vx
        X[3, :] = 0.0                  # vy
        X[4, :] = 0.0                  # vz
        X[5, :] = 0.0                  # phi (roll)
        X[6, :] = 0.0                  # theta (pitch)
        X[7, :] = yaw_rate             # psi_dot (yaw rate)
        X[8, :] = 0.0                  # phi_dot
        X[9, :] = 0.0                  # theta_dot

        # Wheel speeds
        X[10, :] = omega_profile       # omega_fl
        X[11, :] = omega_profile       # omega_fr
        X[12, :] = omega_profile       # omega_rl
        X[13, :] = omega_profile       # omega_rr

        # Compliances & Battery
        X[22, :] = 0.0                 # delta_compl
        X[23, :] = 0.0                 # theta_dt
        X[24, :] = np.linspace(0.95, 0.85, num_nodes)  # SoC gradually discharging
        X[25, :] = np.linspace(25.0, 35.0, num_nodes)  # Battery temp gradually rising

        # Controls: [delta_cmd, throttle, brake_cmd, regen_cmd]
        n_controls = len(self.vehicle.control_names())
        U = np.zeros((n_controls, num_nodes))
        U[0, :] = delta_profile        # Steering command
        U[1, :] = 0.3                  # Moderate throttle guess
        U[2, :] = 0.0                  # No brake
        U[3, :] = 0.0                  # No regen

        return {"s": s, "X": X, "U": U, "track": resampled_track}
