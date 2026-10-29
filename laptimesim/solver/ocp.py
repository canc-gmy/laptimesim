"""Curvilinear Optimal Control Problem (OCP) formulation and IPOPT solver using Direct Collocation."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import casadi as ca
import numpy as np

from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.initial_guess import InitialGuessGenerator


@dataclass
class OCPResult:
    """Output results from optimal control problem solver."""
    success: bool
    lap_time: float
    message: str
    s: np.ndarray
    t: np.ndarray
    states: Dict[str, np.ndarray]
    controls: Dict[str, np.ndarray]
    telemetry: Dict[str, np.ndarray]
    iterations: int
    solver_time: float


class CurvilinearLaptimeSolver:
    """Solves the minimum lap time problem using direct collocation and IPOPT.

    Independent variable is arc-length s along track centerline.
    Objective:
        min integral_0^s_lap (dt/ds) ds = min integral_0^s_lap ( (1 - n*kappa) / (vx*cos(psi) - vy*sin(psi)) ) ds
    """

    def __init__(self, vehicle: VehicleModel, track: CanonicalTrack, config: FullSimulationConfig):
        self.vehicle = vehicle
        self.track = track
        self.config = config
        self.num_nodes = config.simulation.num_nodes

    def _build_spatial_dynamics(self) -> ca.Function:
        """Constructs spatial ODE dx/ds = (dx/dt) * (dt/ds) with Curvilinear frame kinematics."""
        # Vehicle state names
        veh_state_names = self.vehicle.state_names()
        n_veh_states = len(veh_state_names)
        n_controls = len(self.vehicle.control_names())

        # Total OCP state: [n, psi] + veh_states
        # n: lateral distance from track centerline [m]
        # psi: heading error angle relative to track tangent [rad]
        n_ocp_states = 2 + n_veh_states

        x_ocp = ca.SX.sym("x_ocp", n_ocp_states)
        u_ocp = ca.SX.sym("u_ocp", n_controls)
        kappa = ca.SX.sym("kappa")  # Track curvature at current arc-length s

        n_dev = x_ocp[0]
        psi_rel = x_ocp[1]
        x_veh = x_ocp[2:]

        # Vehicle velocities
        vx = x_veh[0]
        vy = x_veh[1]
        r_yaw = x_veh[5]

        # Time-domain vehicle dynamics
        x_veh_dot, telem = self.vehicle.forward_dynamics(x_veh, u_ocp)

        # Curvilinear transformation: dt/ds
        # Progress velocity along track: s_dot = (vx*cos(psi) - vy*sin(psi)) / (1 - n*kappa)
        cos_psi = ca.cos(psi_rel)
        sin_psi = ca.sin(psi_rel)
        v_progress = (vx * cos_psi - vy * sin_psi) / ca.fmax(1.0 - n_dev * kappa, 0.05)
        dt_ds = 1.0 / ca.fmax(v_progress, 1.0)

        # Curvilinear state kinematics
        # dn/ds = (vx*sin(psi) + vy*cos(psi)) * dt/ds
        dn_ds = (vx * sin_psi + vy * cos_psi) * dt_ds
        # dpsi/ds = r_yaw * dt/ds - kappa
        dpsi_ds = r_yaw * dt_ds - kappa

        # Spatial vehicle derivatives: dx_veh/ds = dx_veh/dt * dt/ds
        dx_veh_ds = x_veh_dot * dt_ds

        # Assemble full spatial derivative dx_ocp/ds
        dx_ds = ca.vertcat(dn_ds, dpsi_ds, dx_veh_ds)

        # Create callable CasADi function
        # Returns: dx_ds, dt_ds, telem expressions
        telem_keys = list(telem.keys())
        telem_vals = [telem[k] for k in telem_keys]

        return ca.Function(
            "spatial_dynamics",
            [x_ocp, u_ocp, kappa],
            [dx_ds, dt_ds] + telem_vals,
            ["x", "u", "kappa"],
            ["dx_ds", "dt_ds"] + telem_keys,
        )

    def solve(self) -> OCPResult:
        """Set up and solve the NLP using CasADi Opti stack."""
        start_time = time.time()
        N = self.num_nodes
        resampled_track = self.track.resample(N)
        s_grid = resampled_track.s
        kappa_grid = resampled_track.kappa
        w_l = resampled_track.width_left
        w_r = resampled_track.width_right

        # Generate initial guess
        guess_gen = InitialGuessGenerator(self.vehicle, self.track)
        guess = guess_gen.generate(N)

        opti = ca.Opti()

        n_veh_states = len(self.vehicle.state_names())
        n_ocp_states = 2 + n_veh_states
        n_controls = len(self.vehicle.control_names())

        # Decision variables
        X = opti.variable(n_ocp_states, N)
        U = opti.variable(n_controls, N)

        # Spatial dynamics function
        spatial_dyn = self._build_spatial_dynamics()

        # Build Objective: minimize total lap time
        # Lap time = sum_{k=0}^{N-2} 0.5 * (dt_ds_k + dt_ds_{k+1}) * ds_k
        lap_time_expr = 0.0

        for k in range(N - 1):
            ds_k = float(s_grid[k + 1] - s_grid[k])
            kappa_k = float(kappa_grid[k])
            kappa_k1 = float(kappa_grid[k + 1])

            # Collocation / Trapezoidal integration
            dyn_k = spatial_dyn(X[:, k], U[:, k], kappa_k)
            dyn_k1 = spatial_dyn(X[:, k + 1], U[:, k + 1], kappa_k1)

            dx_ds_k = dyn_k[0]
            dt_ds_k = dyn_k[1]

            dx_ds_k1 = dyn_k1[0]
            dt_ds_k1 = dyn_k1[1]

            # Trapezoidal state continuity constraint: X_{k+1} - X_k = 0.5 * ds * (f_k + f_{k+1})
            opti.subject_to(X[:, k + 1] - X[:, k] == 0.5 * ds_k * (dx_ds_k + dx_ds_k1))

            # Accumulate time
            lap_time_expr += 0.5 * ds_k * (dt_ds_k + dt_ds_k1)

        # Regularization on control effort to improve numerical conditioning and smooth inputs
        reg_control = 1e-4 * ca.sumsqr(U)
        reg_derivs = 1e-3 * ca.sumsqr(U[:, 1:] - U[:, :-1])
        opti.minimize(lap_time_expr + reg_control + reg_derivs)

        # Path constraints at each node
        for k in range(N):
            # Track limit constraints on lateral deviation n: -width_right <= n <= width_left
            # Allow margin for half-car width (~0.7m)
            margin = 0.7
            opti.subject_to(opti.bounded(-float(w_r[k]) + margin, X[0, k], float(w_l[k]) - margin))

            # Heading error psi relative to track: keep within [-pi/3, pi/3]
            opti.subject_to(opti.bounded(-1.0, X[1, k], 1.0))

            # Forward speed bounds: vx >= 5.0 m/s
            opti.subject_to(opti.bounded(5.0, X[2, k], 45.0))

            # Control bounds:
            # Steering angle: [-0.4 rad, 0.4 rad] (~23 deg)
            opti.subject_to(opti.bounded(-0.4, U[0, k], 0.4))
            # Throttle: [0.0, 1.0]
            opti.subject_to(opti.bounded(0.0, U[1, k], 1.0))
            # Brake: [0.0, 1.0]
            opti.subject_to(opti.bounded(0.0, U[2, k], 1.0))
            # Regen: [0.0, 1.0]
            opti.subject_to(opti.bounded(0.0, U[3, k], 1.0))

            # Exclusive throttle and braking (soft/slack bounded)
            opti.subject_to(U[1, k] * U[2, k] <= 0.05)

        # Periodicity / Closed lap constraints: X[0] == X[N-1]
        # For non-cyclic states like battery SoC and temperature, allow terminal values <= initial values
        cyclic_state_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        for idx in cyclic_state_indices:
            opti.subject_to(X[idx, 0] == X[idx, N - 1])

        # Set initial guesses
        opti.set_initial(X, guess["X"])
        opti.set_initial(U, guess["U"])

        # Configure IPOPT solver options
        p_opts = {
            "expand": True,
            "print_time": 0,
        }
        s_opts = {
            "max_iter": self.config.simulation.solver_max_iter,
            "tol": self.config.simulation.solver_tolerance,
            "print_level": 5 if self.config.logging.solver_iterations else 0,
            "linear_solver": "mumps",
            "mu_strategy": "adaptive",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # Solve
        success = False
        message = ""
        iterations = 0
        sol = None

        try:
            sol = opti.solve()
            success = True
            message = "Optimal solution found."
            iterations = sol.stats()["iter_count"]
        except Exception as e:
            # Even on non-convergence or max-iters, opti provides debug access to the last iterate
            success = False
            message = str(e)
            sol = opti.debug

        # Extract values
        X_val = sol.value(X)
        U_val = sol.value(U)
        total_time_val = float(sol.value(lap_time_expr))

        # Reconstruct time array along track
        t_arr = np.zeros(N)
        for k in range(N - 1):
            ds_k = float(s_grid[k + 1] - s_grid[k])
            dyn_k = spatial_dyn(X_val[:, k], U_val[:, k], float(kappa_grid[k]))
            dyn_k1 = spatial_dyn(X_val[:, k + 1], U_val[:, k + 1], float(kappa_grid[k + 1]))
            dt_k = 0.5 * ds_k * (float(dyn_k[1]) + float(dyn_k1[1]))
            t_arr[k + 1] = t_arr[k] + dt_k

        # Map state names to arrays
        all_state_names = ["n", "psi"] + self.vehicle.state_names()
        states_dict = {all_state_names[i]: X_val[i, :] for i in range(len(all_state_names))}

        control_names = self.vehicle.control_names()
        controls_dict = {control_names[i]: U_val[i, :] for i in range(len(control_names))}

        # Extract telemetry across all nodes
        telemetry_dict = {}
        for k in range(N):
            out_k = spatial_dyn(X_val[:, k], U_val[:, k], float(kappa_grid[k]))
            # out_k[0] is dx_ds, out_k[1] is dt_ds, subsequent outputs are telemetry keys
            # Record ax, ay, P_elec
            if k == 0:
                for idx, key in enumerate(spatial_dyn.name_out()[2:]):
                    telemetry_dict[key] = np.zeros(N)
            for idx, key in enumerate(spatial_dyn.name_out()[2:]):
                telemetry_dict[key][k] = float(out_k[2 + idx])

        solver_elapsed = time.time() - start_time

        return OCPResult(
            success=success,
            lap_time=total_time_val,
            message=message,
            s=s_grid,
            t=t_arr,
            states=states_dict,
            controls=controls_dict,
            telemetry=telemetry_dict,
            iterations=iterations,
            solver_time=solver_elapsed,
        )
