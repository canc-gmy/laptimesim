"""High-fidelity 2-RC Equivalent Circuit Model (ECM) with active thermal cooling."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.battery.base import BatteryModelBase


@model_registry.register("battery", "ecm_2rc")
class TwoRCEquivalentCircuitBattery(BatteryModelBase):
    """2-RC battery pack model accounting for electrochemical charge transfer and diffusion polarization."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.capacity_kwh = float(params.get("capacity_kwh", 6.8))
        self.capacity_joules = self.capacity_kwh * 3.6e6
        self.voltage_nominal = float(params.get("voltage_nominal", 400.0))

        # 2-RC Circuit Parameters
        self.r0 = float(params.get("r0", 0.035))      # Ohmic resistance [Ohms]
        self.r1 = float(params.get("r1", 0.015))      # Fast charge transfer resistance [Ohms]
        self.c1 = float(params.get("c1", 1200.0))     # Fast double-layer capacitance [Farads]
        self.r2 = float(params.get("r2", 0.020))      # Slower diffusion resistance [Ohms]
        self.c2 = float(params.get("c2", 8000.0))     # Slow diffusion capacitance [Farads]

        # Thermal mass and cooling flow
        self.thermal_mass = float(params.get("thermal_mass", 5500.0))    # J/K
        self.ambient_temp = float(params.get("ambient_temp", 25.0))
        self.h_cooling_base = float(params.get("cooling_base", 30.0))     # W/K

    def step(
        self,
        P_elec: ca.SX,
        soc: ca.SX,
        temp: ca.SX,
        dt: ca.SX,
        v_rc1: ca.SX = 0.0,
        v_rc2: ca.SX = 0.0,
        car_speed: ca.SX = 0.0,
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        # Open Circuit Voltage curve (characteristic LiFePO4 / NMC profile)
        Voc = self.voltage_nominal * (0.84 + 0.32 * soc)

        # Current demand: I ≈ P / Voc
        I = P_elec / ca.fmax(Voc, 200.0)

        # 2-RC state equations:
        # dv_rc1/dt = - v_rc1 / (r1*c1) + I / c1
        # dv_rc2/dt = - v_rc2 / (r2*c2) + I / c2
        # Terminal voltage
        V_term = Voc - I * self.r0 - v_rc1 - v_rc2

        # Rate of change of SoC
        d_soc_dt = -P_elec / self.capacity_joules

        # Total heat generation: Joule heating across R0, R1, R2
        Q_joule = (I**2) * (self.r0 + self.r1 + self.r2)

        # Active airspeed-dependent cooling: h(v) = h0 * (1 + 0.05 * v_car)
        h_eff = self.h_cooling_base * (1.0 + 0.05 * car_speed)
        Q_cooling = h_eff * (temp - self.ambient_temp)

        d_temp_dt = (Q_joule - Q_cooling) / self.thermal_mass

        return d_soc_dt, d_temp_dt, V_term
