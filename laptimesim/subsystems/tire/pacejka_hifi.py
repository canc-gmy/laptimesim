"""High-fidelity Pacejka Magic Formula tire model with load sensitivity, camber thrust, and thermal degradation.

Features:
- Normal load sensitivity: friction coefficient mu decays at high Fz (mu(Fz) = D0 - dFz * Fz)
- Camber thrust: Fy += C_gamma * gamma * Fz
- Thermal friction curve: bell-shaped friction multiplier around optimal tire operating window (75-95 deg C)
- Tire surface thermal state equations: slip friction heating, convective air cooling, and carcass conduction
"""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.tire.base import TireModelBase


@model_registry.register("tire", "pacejka_hifi")
class PacejkaHiFiTire(TireModelBase):
    """High-fidelity Pacejka model with thermal and load sensitivities."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # Base Magic Formula parameters
        self.B_lat = float(params.get("B_lat", params.get("B", 10.5)))
        self.C_lat = float(params.get("C_lat", params.get("C", 1.65)))
        self.D0_lat = float(params.get("D0_lat", params.get("D", 1.70)))  # Zero-load peak friction
        self.E_lat = float(params.get("E_lat", params.get("E", -0.4)))

        self.B_long = float(params.get("B_long", params.get("B", 11.0)))
        self.C_long = float(params.get("C_long", params.get("C", 1.68)))
        self.D0_long = float(params.get("D0_long", params.get("D", 1.75)))
        self.E_long = float(params.get("E_long", params.get("E", -0.4)))

        # Load sensitivity coefficient (degradation of mu with vertical load)
        self.dFz_lat = float(params.get("dFz_lat", 0.00015))   # 1/N
        self.dFz_long = float(params.get("dFz_long", 0.00012))

        # Camber stiffness (N/rad per Newton of Fz)
        self.C_gamma = float(params.get("C_gamma", 1.8))

        # Thermal grip window parameters (optimal around 85 deg C)
        self.T_opt = float(params.get("T_opt", 85.0))         # deg C
        self.T_width = float(params.get("T_width", 25.0))     # deg C
        self.mu_cold_factor = float(params.get("mu_cold_factor", 0.65))

        # Thermal mass and cooling parameters
        self.thermal_mass_surf = float(params.get("thermal_mass_surf", 250.0))  # J/K
        self.h_conv_base = float(params.get("h_conv_base", 15.0))              # W/K
        self.radius = float(params.get("radius", 0.2032))

    def thermal_grip_multiplier(self, T_surf: ca.SX) -> ca.SX:
        """Gaussian/bell-shaped thermal efficiency curve centered at T_opt."""
        delta_T = (T_surf - self.T_opt) / self.T_width
        eta_thermal = self.mu_cold_factor + (1.0 - self.mu_cold_factor) * ca.exp(-0.5 * (delta_T**2))
        return eta_thermal

    def _magic_formula_hifi(
        self,
        slip: ca.SX,
        B: float,
        C: float,
        D0: float,
        dFz: float,
        E: float,
        Fz: ca.SX,
        T_surf: ca.SX,
    ) -> ca.SX:
        """Evaluates load-sensitive and thermally-degraded friction peak."""
        Fz_pos = ca.fmax(Fz, 1.0)
        # Load sensitivity: peak friction decays linearly with normal load
        mu_load = ca.fmax(0.8, D0 - dFz * Fz_pos)
        # Thermal scaling
        mu_eff = mu_load * self.thermal_grip_multiplier(T_surf)

        peak_force = mu_eff * Fz_pos
        bx = B * slip
        inner = bx - E * (bx - ca.atan(bx))
        force = peak_force * ca.sin(C * ca.atan(inner))

        return ca.if_else(Fz > 0, force, 0.0)

    def lateral_force(self, alpha: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0, T_surf: ca.SX = 85.0) -> ca.SX:
        Fy0 = self._magic_formula_hifi(alpha, self.B_lat, self.C_lat, self.D0_lat, self.dFz_lat, self.E_lat, Fz, T_surf)
        # Camber thrust component: F_gamma = C_gamma * gamma * Fz
        F_camber = self.C_gamma * gamma * Fz
        return Fy0 + F_camber

    def longitudinal_force(self, kappa: ca.SX, Fz: ca.SX, T_surf: ca.SX = 85.0) -> ca.SX:
        Fx0 = self._magic_formula_hifi(kappa, self.B_long, self.C_long, self.D0_long, self.dFz_long, self.E_long, Fz, T_surf)
        return Fx0

    def combined_forces(
        self, alpha: ca.SX, kappa: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0, T_surf: ca.SX = 85.0
    ) -> Tuple[ca.SX, ca.SX]:
        Fx0 = self.longitudinal_force(kappa, Fz, T_surf)
        Fy0 = self.lateral_force(alpha, Fz, gamma, T_surf)

        eps = 1e-6
        denom = ca.sqrt(alpha**2 + kappa**2 + eps)
        cos_theta = ca.fabs(kappa) / denom
        sin_theta = ca.fabs(alpha) / denom

        return Fx0 * cos_theta, Fy0 * sin_theta

    def thermal_step(
        self,
        Fx: ca.SX,
        Fy: ca.SX,
        v_slip_x: ca.SX,
        v_slip_y: ca.SX,
        vx: ca.SX,
        T_surf: ca.SX,
        T_ambient: float = 25.0,
    ) -> ca.SX:
        """Compute rate of change of tire surface temperature dT_surf/dt [K/s]."""
        # Friction power generation = F_slip * v_slip
        Q_friction = ca.fabs(Fx * v_slip_x) + ca.fabs(Fy * v_slip_y)

        # Convective cooling scales with vehicle forward velocity: h = h0 * (1 + 0.1 * vx)
        h_eff = self.h_conv_base * (1.0 + 0.08 * vx)
        Q_cooling = h_eff * (T_surf - T_ambient)

        # Carcass conduction
        Q_conduction = 5.0 * (T_surf - (T_ambient + 15.0))

        dT_dt = (Q_friction - Q_cooling - Q_conduction) / self.thermal_mass_surf
        return dT_dt
