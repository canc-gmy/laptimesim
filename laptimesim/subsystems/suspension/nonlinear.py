"""Nonlinear suspension model with bump stops and velocity-sensitive asymmetric damping."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.suspension.base import SuspensionModelBase


@model_registry.register("suspension", "nonlinear")
class NonlinearSuspension(SuspensionModelBase):
    """Nonlinear spring, bump stop engagement, asymmetric damping, and anti-roll bars."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        front = params.get("front", {})
        rear = params.get("rear", {})

        # Front linear + bump stop parameters
        self.k_main_f = float(front.get("spring_rate", 35000.0))
        self.k_bump_f = float(front.get("bump_stop_rate", 120000.0))
        self.clearance_bump_f = float(front.get("bump_stop_clearance", 0.015))  # 15 mm bump clearance
        self.c_comp_f = float(front.get("damping_comp", 1600.0))
        self.c_reb_f = float(front.get("damping_reb", 2200.0))
        self.k_arb_f = float(front.get("arb_rate", 5000.0))

        # Rear linear + bump stop parameters
        self.k_main_r = float(rear.get("spring_rate", 40000.0))
        self.k_bump_r = float(rear.get("bump_stop_rate", 140000.0))
        self.clearance_bump_r = float(rear.get("bump_stop_clearance", 0.018))
        self.c_comp_r = float(rear.get("damping_comp", 1800.0))
        self.c_reb_r = float(rear.get("damping_reb", 2500.0))
        self.k_arb_r = float(rear.get("arb_rate", 4000.0))

    def _corner_force(
        self,
        zs: ca.SX,
        dzs: ca.SX,
        k_main: float,
        k_bump: float,
        bump_clearance: float,
        c_comp: float,
        c_reb: float,
    ) -> ca.SX:
        """Compute nonlinear spring and asymmetric damping force at one corner."""
        # Main coil spring
        f_spring = k_main * zs

        # Bump stop: activates progressively when deflection zs exceeds bump_clearance
        bump_deflection = ca.fmax(0.0, zs - bump_clearance)
        f_bump = k_bump * (bump_deflection**1.5)  # Progressive elastomeric bump stop curve

        # Asymmetric damping: compression (dzs > 0) vs. rebound (dzs < 0)
        c_eff = ca.if_else(dzs >= 0.0, c_comp, c_reb)
        f_damper = c_eff * dzs

        return f_spring + f_bump + f_damper

    def suspension_forces(
        self,
        z_s_fl: ca.SX,
        z_s_fr: ca.SX,
        z_s_rl: ca.SX,
        z_s_rr: ca.SX,
        dz_s_fl: ca.SX,
        dz_s_fr: ca.SX,
        dz_s_rl: ca.SX,
        dz_s_rr: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
        # Corner forces
        f_fl = self._corner_force(z_s_fl, dz_s_fl, self.k_main_f, self.k_bump_f, self.clearance_bump_f, self.c_comp_f, self.c_reb_f)
        f_fr = self._corner_force(z_s_fr, dz_s_fr, self.k_main_f, self.k_bump_f, self.clearance_bump_f, self.c_comp_f, self.c_reb_f)
        f_rl = self._corner_force(z_s_rl, dz_s_rl, self.k_main_r, self.k_bump_r, self.clearance_bump_r, self.c_comp_r, self.c_reb_r)
        f_rr = self._corner_force(z_s_rr, dz_s_rr, self.k_main_r, self.k_bump_r, self.clearance_bump_r, self.c_comp_r, self.c_reb_r)

        # Anti-roll bars
        f_arb_f = self.k_arb_f * (z_s_fl - z_s_fr)
        f_arb_r = self.k_arb_r * (z_s_rl - z_s_rr)

        return (
            f_fl + f_arb_f,
            f_fr - f_arb_f,
            f_rl + f_arb_r,
            f_rr - f_arb_r,
        )
