"""Pacejka Magic Formula Tire Model (MF 5.2 simplified / standard form)."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.tire.base import TireModelBase


@model_registry.register("tire", "pacejka_mf52")
class PacejkaMF52(TireModelBase):
    """Pacejka Magic Formula tire model: y = D * sin(C * arctan(B * x - E * (B * x - arctan(B * x)))).

    Parameters expected in params:
        - B: Stiffness factor
        - C: Shape factor
        - D: Peak factor (friction coefficient or scaled by Fz)
        - E: Curvature factor
        - radius: Loaded tire radius [m]
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.B_lat = float(params.get("B_lat", params.get("B", 10.0)))
        self.C_lat = float(params.get("C_lat", params.get("C", 1.6)))
        self.D_lat = float(params.get("D_lat", params.get("D", 1.5)))
        self.E_lat = float(params.get("E_lat", params.get("E", -0.5)))

        self.B_long = float(params.get("B_long", params.get("B", 11.0)))
        self.C_long = float(params.get("C_long", params.get("C", 1.65)))
        self.D_long = float(params.get("D_long", params.get("D", 1.55)))
        self.E_long = float(params.get("E_long", params.get("E", -0.4)))

        self.radius = float(params.get("radius", 0.2286))

    def _magic_formula(self, slip: ca.SX, B: float, C: float, D: float, E: float, Fz: ca.SX) -> ca.SX:
        """Core Magic Formula equation evaluated symbolically."""
        # Fz must be non-negative (clipped smoothly or via fmax)
        Fz_pos = ca.fmax(Fz, 1.0)
        peak = D * Fz_pos
        bx = B * slip
        bx_atan = ca.atan(bx)
        inner = bx - E * (bx - bx_atan)
        force = peak * ca.sin(C * ca.atan(inner))
        # If normal load was negative or zero, force vanishes smoothly
        return ca.if_else(Fz > 0, force, 0.0)

    def lateral_force(self, alpha: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0) -> ca.SX:
        # Camber thrust can be added linearly: Fy += C_gamma * gamma * Fz
        Fy0 = self._magic_formula(alpha, self.B_lat, self.C_lat, self.D_lat, self.E_lat, Fz)
        return Fy0

    def longitudinal_force(self, kappa: ca.SX, Fz: ca.SX) -> ca.SX:
        Fx0 = self._magic_formula(kappa, self.B_long, self.C_long, self.D_long, self.E_long, Fz)
        return Fx0

    def combined_forces(
        self, alpha: ca.SX, kappa: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0
    ) -> Tuple[ca.SX, ca.SX]:
        """Combined slip calculation using friction ellipse weighting."""
        Fx0 = self.longitudinal_force(kappa, Fz)
        Fy0 = self.lateral_force(alpha, Fz, gamma)

        # Friction ellipse interaction
        eps = 1e-6
        denom = ca.sqrt(alpha**2 + kappa**2 + eps)
        cos_theta = ca.fabs(kappa) / denom
        sin_theta = ca.fabs(alpha) / denom

        Fx_comb = Fx0 * cos_theta
        Fy_comb = Fy0 * sin_theta
        return Fx_comb, Fy_comb
