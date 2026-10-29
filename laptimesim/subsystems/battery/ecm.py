"""Equivalent Circuit Model (ECM) for Formula Student electric vehicle battery."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.battery.base import BatteryModelBase


@model_registry.register("battery", "ecm")
class EquivalentCircuitBattery(BatteryModelBase):
    """Rint / ECM model with open-circuit voltage, internal resistance, and thermal mass."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.capacity_kwh = float(params.get("capacity_kwh", 6.5))
        self.capacity_joules = self.capacity_kwh * 3.6e6
        self.voltage_nominal = float(params.get("voltage_nominal", 400.0))
        self.r_int = float(params.get("internal_resistance", 0.05))  # Ohms
        self.thermal_mass = float(params.get("thermal_mass", 5000.0))  # J/K
        self.ambient_temp = float(params.get("ambient_temp", 25.0))
        self.cooling_h = float(params.get("cooling_coefficient", 25.0))  # W/K

    def step(
        self, P_elec: ca.SX, soc: ca.SX, temp: ca.SX, dt: ca.SX
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        # Approximate Voc as linear function of SoC around nominal voltage
        Voc = self.voltage_nominal * (0.85 + 0.3 * soc)

        # I = (Voc - sqrt(Voc^2 - 4*R*P)) / (2*R)
        # Using first-order approximation for numerical robustness: I ≈ P / Voc
        I = P_elec / Voc
        V_term = Voc - I * self.r_int

        # Rate of change of SoC: dSoC/dt = - (I * Voc) / Q_total
        d_soc_dt = -P_elec / self.capacity_joules

        # Heat generation: Q_heat = I^2 * R_int
        Q_heat = (I**2) * self.r_int
        Q_cooling = self.cooling_h * (temp - self.ambient_temp)
        d_temp_dt = (Q_heat - Q_cooling) / self.thermal_mass

        return d_soc_dt, d_temp_dt, V_term
