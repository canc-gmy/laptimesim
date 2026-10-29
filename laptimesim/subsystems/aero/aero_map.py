"""Multi-dimensional Aerodynamic Map Model using CasADi symbolic interpolation."""

from typing import Tuple, Dict, Any
import numpy as np
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.aero.base import AeroModelBase


@model_registry.register("aero", "aero_map")
class AeroMapModel(AeroModelBase):
    """Aerodynamic model utilizing multi-dimensional ride height and pitch maps.

    Interpolates Cl, Cd, and front aero balance (% front downforce) as functions
    of front ride height (rh_f) and rear ride height (rh_r) or pitch angle.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.frontal_area = float(params.get("frontal_area", 1.0))
        self.air_density = float(params.get("air_density", 1.225))

        # Define 2D grid: front ride height and rear ride height [m]
        self.rh_f_grid = np.array(params.get("rh_f_grid", [0.020, 0.030, 0.040, 0.050]), dtype=float)
        self.rh_r_grid = np.array(params.get("rh_r_grid", [0.025, 0.035, 0.045, 0.055]), dtype=float)

        # Default representative aerodynamic map tables (Cl, Cd, balance)
        cl_table = np.array(
            params.get(
                "cl_map",
                [
                    [3.4, 3.3, 3.1, 2.9],
                    [3.5, 3.4, 3.2, 3.0],
                    [3.3, 3.2, 3.0, 2.8],
                    [3.1, 3.0, 2.8, 2.6],
                ],
            ),
            dtype=float,
        )

        cd_table = np.array(
            params.get(
                "cd_map",
                [
                    [1.35, 1.32, 1.28, 1.25],
                    [1.40, 1.36, 1.30, 1.26],
                    [1.38, 1.34, 1.29, 1.24],
                    [1.33, 1.30, 1.25, 1.21],
                ],
            ),
            dtype=float,
        )

        bal_table = np.array(
            params.get(
                "bal_map",
                [
                    [0.46, 0.45, 0.43, 0.41],
                    [0.48, 0.46, 0.44, 0.42],
                    [0.47, 0.45, 0.43, 0.40],
                    [0.45, 0.43, 0.41, 0.39],
                ],
            ),
            dtype=float,
        )

        # Build CasADi B-spline interpolants for smooth, differentiable lookup
        self.cl_interp = ca.interpolant("cl_interp", "bspline", [self.rh_f_grid.tolist(), self.rh_r_grid.tolist()], cl_table.flatten(order="F"))
        self.cd_interp = ca.interpolant("cd_interp", "bspline", [self.rh_f_grid.tolist(), self.rh_r_grid.tolist()], cd_table.flatten(order="F"))
        self.bal_interp = ca.interpolant("bal_interp", "bspline", [self.rh_f_grid.tolist(), self.rh_r_grid.tolist()], bal_table.flatten(order="F"))

    def forces_and_moments(
        self,
        vx: ca.SX,
        vy: ca.SX,
        vz: ca.SX,
        ride_height_f: ca.SX,
        ride_height_r: ca.SX,
        pitch: ca.SX,
        roll: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
        # Clip ride heights smoothly within grid bounds
        rh_f_clipped = ca.fmin(ca.fmax(ride_height_f, self.rh_f_grid[0]), self.rh_f_grid[-1])
        rh_r_clipped = ca.fmin(ca.fmax(ride_height_r, self.rh_r_grid[0]), self.rh_r_grid[-1])

        # Evaluate interpolated coefficients
        query_pt = ca.vertcat(rh_f_clipped, rh_r_clipped)
        Cl = self.cl_interp(query_pt)
        Cd = self.cd_interp(query_pt)
        balance = self.bal_interp(query_pt)

        v_squared = vx**2 + vy**2
        q = 0.5 * self.air_density * v_squared * self.frontal_area

        drag = Cd * q
        total_downforce = Cl * q
        df_front = total_downforce * balance
        df_rear = total_downforce * (1.0 - balance)
        side_force = 0.0

        return drag, df_front, df_rear, side_force
