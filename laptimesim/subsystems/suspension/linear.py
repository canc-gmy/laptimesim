"""Linear spring-damper suspension model with Anti-Roll Bars (ARB)."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.suspension.base import SuspensionModelBase


@model_registry.register("suspension", "linear")
class LinearSuspension(SuspensionModelBase):
    """Linear spring, damper, and anti-roll bar model for each axle."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        front = params.get("front", {})
        rear = params.get("rear", {})

        self.k_spring_f = float(front.get("spring_rate", 35000.0))  # N/m
        self.c_damper_f = float(front.get("damping", 1800.0))       # N*s/m
        self.k_arb_f = float(front.get("arb_rate", 5000.0))         # N/m

        self.k_spring_r = float(rear.get("spring_rate", 40000.0))
        self.c_damper_r = float(rear.get("damping", 2000.0))
        self.k_arb_r = float(rear.get("arb_rate", 4000.0))

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
        # Front ARB force: transfers load from compressed side to extended side
        f_arb_f = self.k_arb_f * (z_s_fl - z_s_fr)
        f_arb_r = self.k_arb_r * (z_s_rl - z_s_rr)

        # Total vertical force per corner (positive pushes chassis upward)
        F_fl = self.k_spring_f * z_s_fl + self.c_damper_f * dz_s_fl + f_arb_f
        F_fr = self.k_spring_f * z_s_fr + self.c_damper_f * dz_s_fr - f_arb_f
        F_rl = self.k_spring_r * z_s_rl + self.c_damper_r * dz_s_rl + f_arb_r
        F_rr = self.k_spring_r * z_s_rr + self.c_damper_r * dz_s_rr - f_arb_r

        return F_fl, F_fr, F_rl, F_rr
