"""Electric Powertrain Model for Formula Student vehicle."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.powertrain.base import PowertrainModelBase


@model_registry.register("powertrain", "electric")
class ElectricPowertrain(PowertrainModelBase):
    """Electric powertrain with motor torque-speed curve, gear reduction, and rear differential."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.motor_peak_torque = float(params.get("motor_peak_torque", 230.0))  # Nm
        self.motor_peak_power = float(params.get("motor_peak_power", 80000.0))  # W (80 kW FSAE limit)
        self.motor_max_rpm = float(params.get("motor_max_rpm", 6000.0))
        self.gear_ratio = float(params.get("gear_ratio", 3.5))
        self.eta_drivetrain = float(params.get("drivetrain_efficiency", 0.92))
        self.regen_enabled = bool(params.get("regen_enabled", True))
        self.regen_max_torque = float(params.get("regen_max_torque", 60.0))  # Nm

    def motor_torque_limit(self, omega_motor: ca.SX) -> ca.SX:
        """Hyperbolic power limit curve: T_max(omega) = min(T_peak, P_peak / omega)."""
        eps = 1e-3
        omega_eff = ca.fmax(ca.fabs(omega_motor), eps)
        t_power_limited = self.motor_peak_power / omega_eff
        return ca.fmin(self.motor_peak_torque, t_power_limited)

    def wheel_torques(
        self,
        throttle: ca.SX,
        regen_cmd: ca.SX,
        omega_fl: ca.SX,
        omega_fr: ca.SX,
        omega_rl: ca.SX,
        omega_rr: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        # Rear-wheel drive setup typical of FSAE EV
        omega_axle_rear = 0.5 * (omega_rl + omega_rr)
        omega_motor = omega_axle_rear * self.gear_ratio

        # Drive torque requested
        t_max = self.motor_torque_limit(omega_motor)
        t_drive_motor = throttle * t_max

        # Regen braking torque
        t_regen_motor = ca.if_else(
            self.regen_enabled,
            regen_cmd * ca.fmin(self.regen_max_torque, t_max),
            0.0
        )
        t_net_motor = t_drive_motor - t_regen_motor

        # Net mechanical wheel torque (split 50:50 across rear open/spool differential)
        t_wheel_rear = 0.5 * t_net_motor * self.gear_ratio * self.eta_drivetrain

        # Mechanical and electrical power
        p_mech = t_net_motor * omega_motor
        p_elec = ca.if_else(
            p_mech >= 0,
            p_mech / self.eta_drivetrain,  # Discharging losses
            p_mech * self.eta_drivetrain   # Charging / regen losses
        )

        t_wheel_fl = 0.0
        t_wheel_fr = 0.0
        t_wheel_rl = t_wheel_rear
        t_wheel_rr = t_wheel_rear

        return t_wheel_fl, t_wheel_fr, t_wheel_rl, t_wheel_rr, p_elec
