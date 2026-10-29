"""Constant aerodynamic coefficients model."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.aero.base import AeroModelBase


@model_registry.register("aero", "constant")
class ConstantAeroModel(AeroModelBase):
    """Aerodynamic model with constant Cl, Cd, and front aero distribution."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.Cl = float(params.get("Cl", 3.0))
        self.Cd = float(params.get("Cd", 1.2))
        self.frontal_area = float(params.get("frontal_area", 1.0))
        self.air_density = float(params.get("air_density", 1.225))  # kg/m^3
        self.cop_longitudinal = float(params.get("cop_longitudinal", 0.45))  # Fraction on front axle

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
        v_squared = vx**2 + vy**2
        q = 0.5 * self.air_density * v_squared * self.frontal_area

        drag = self.Cd * q
        total_downforce = self.Cl * q
        df_front = total_downforce * self.cop_longitudinal
        df_rear = total_downforce * (1.0 - self.cop_longitudinal)
        side_force = 0.0

        return drag, df_front, df_rear, side_force
