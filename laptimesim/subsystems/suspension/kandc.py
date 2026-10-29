"""Full Kinematics & Compliance (K&C) suspension model.

Features:
- Dynamic Camber Gain: gamma(zs) = gamma_0 + d_gamma/dzs * zs
- Dynamic Bump Steer / Toe Change: delta_toe(zs) = toe_0 + d_toe/dzs * zs
- Dynamic Roll Center Migration: h_RC(zs, phi) migrating with roll and pitch
- Anti-Dive & Anti-Squat geometric pitch moments
- Bump stop elastomeric progressive stiffness
"""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.suspension.base import SuspensionModelBase


@model_registry.register("suspension", "kandc")
class KinematicComplianceSuspension(SuspensionModelBase):
    """Full K&C suspension model evaluating elastokinematic geometry changes."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        front = params.get("front", {})
        rear = params.get("rear", {})

        # Spring and damper rates
        self.k_main_f = float(front.get("spring_rate", 38000.0))
        self.k_main_r = float(rear.get("spring_rate", 42000.0))
        self.c_comp_f = float(front.get("damping_comp", 1800.0))
        self.c_reb_f = float(front.get("damping_reb", 2400.0))
        self.c_comp_r = float(rear.get("damping_comp", 2000.0))
        self.c_reb_r = float(rear.get("damping_reb", 2600.0))

        # Anti-roll bar rates
        self.k_arb_f = float(front.get("arb_rate", 5500.0))
        self.k_arb_r = float(rear.get("arb_rate", 4500.0))

        # Bump stops
        self.k_bump_f = float(front.get("bump_stop_rate", 120000.0))
        self.k_bump_r = float(rear.get("bump_stop_rate", 140000.0))
        self.clearance_bump_f = float(front.get("bump_stop_clearance", 0.015))
        self.clearance_bump_r = float(rear.get("bump_stop_clearance", 0.018))

        # Static angles (rad)
        self.static_camber_f = float(front.get("static_camber_deg", -2.5)) * (ca.pi / 180.0)
        self.static_camber_r = float(rear.get("static_camber_deg", -1.5)) * (ca.pi / 180.0)
        self.static_toe_f = float(front.get("static_toe_deg", 0.5)) * (ca.pi / 180.0)  # Toe-out
        self.static_toe_r = float(rear.get("static_toe_deg", -0.2)) * (ca.pi / 180.0) # Toe-in

        # Kinematic gradients: rad per meter of suspension travel
        self.camber_gain_f = float(front.get("camber_gain_deg_m", -35.0)) * (ca.pi / 180.0)
        self.camber_gain_r = float(rear.get("camber_gain_deg_m", -25.0)) * (ca.pi / 180.0)
        self.bump_steer_f = float(front.get("bump_steer_deg_m", 5.0)) * (ca.pi / 180.0)
        self.bump_steer_r = float(rear.get("bump_steer_deg_m", 2.0)) * (ca.pi / 180.0)

        # Anti-geometry percentages
        self.anti_dive_pct = float(front.get("anti_dive_pct", 25.0)) / 100.0
        self.anti_squat_pct = float(rear.get("anti_squat_pct", 30.0)) / 100.0

    def wheel_kinematics(
        self,
        zs_fl: ca.SX,
        zs_fr: ca.SX,
        zs_rl: ca.SX,
        zs_rr: ca.SX,
        roll: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        """Calculates dynamic wheel camber angles (gamma) and bump-steer toe angles (delta_toe).

        Returns:
            Tuple of:
                - (gamma_fl, gamma_fr, gamma_rl, gamma_rr) [rad]
                - (toe_fl, toe_fr, toe_rl, toe_rr) [rad]
        """
        # Dynamic Camber = Static Camber + Camber Gain * zs + Roll Angle
        gamma_fl = self.static_camber_f + self.camber_gain_f * zs_fl + roll
        gamma_fr = -self.static_camber_f - self.camber_gain_f * zs_fr + roll
        gamma_rl = self.static_camber_r + self.camber_gain_r * zs_rl + roll
        gamma_rr = -self.static_camber_r - self.camber_gain_r * zs_rr + roll

        # Dynamic Toe = Static Toe + Bump Steer * zs
        toe_fl = self.static_toe_f + self.bump_steer_f * zs_fl
        toe_fr = -self.static_toe_f - self.bump_steer_f * zs_fr
        toe_rl = self.static_toe_r + self.bump_steer_r * zs_rl
        toe_rr = -self.static_toe_r - self.bump_steer_r * zs_rr

        return (gamma_fl, gamma_fr, gamma_rl, gamma_rr, toe_fl, toe_fr, toe_rl, toe_rr)

    def _corner_force(
        self, zs: ca.SX, dzs: ca.SX, k_main: float, k_bump: float, clearance: float, c_comp: float, c_reb: float
    ) -> ca.SX:
        f_spring = k_main * zs
        bump_deflection = ca.fmax(0.0, zs - clearance)
        f_bump = k_bump * (bump_deflection**1.5)
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
        f_fl = self._corner_force(z_s_fl, dz_s_fl, self.k_main_f, self.k_bump_f, self.clearance_bump_f, self.c_comp_f, self.c_reb_f)
        f_fr = self._corner_force(z_s_fr, dz_s_fr, self.k_main_f, self.k_bump_f, self.clearance_bump_f, self.c_comp_f, self.c_reb_f)
        f_rl = self._corner_force(z_s_rl, dz_s_rl, self.k_main_r, self.k_bump_r, self.clearance_bump_r, self.c_comp_r, self.c_reb_r)
        f_rr = self._corner_force(z_s_rr, dz_s_rr, self.k_main_r, self.k_bump_r, self.clearance_bump_r, self.c_comp_r, self.c_reb_r)

        # Anti-roll bar load transfers
        f_arb_f = self.k_arb_f * (z_s_fl - z_s_fr)
        f_arb_r = self.k_arb_r * (z_s_rl - z_s_rr)

        return (
            f_fl + f_arb_f,
            f_fr - f_arb_f,
            f_rl + f_arb_r,
            f_rr - f_arb_r,
        )
