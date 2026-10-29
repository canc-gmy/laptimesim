"""4-Motor Independent All-Wheel Drive with Active Direct Yaw Moment Torque Vectoring."""

from typing import Tuple, Dict, Any
import casadi as ca
from laptimesim.core.registry import model_registry
from laptimesim.subsystems.powertrain.base import PowertrainModelBase


@model_registry.register("powertrain", "four_wheel_drive_tv")
class FourWheelDriveTorqueVectoring(PowertrainModelBase):
    """4-Motor independent AWD with active electronic torque vectoring (TV) and 80 kW rules limit."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # 4 in-hub / outboard motors
        self.motor_peak_torque_corner = float(params.get("motor_peak_torque_corner", 65.0))  # 4x 65 Nm = 260 Nm
        self.total_power_ceiling = float(params.get("total_power_ceiling", 80000.0))         # 80 kW rule ceiling
        self.motor_max_rpm = float(params.get("motor_max_rpm", 20000.0))                    # AMK / Fischer motor
        self.gear_ratio = float(params.get("gear_ratio", 11.5))                             # Planetary in-hub ratio
        self.eta_drivetrain = float(params.get("drivetrain_efficiency", 0.94))
        self.regen_max_torque_corner = float(params.get("regen_max_torque_corner", 35.0))

        # Torque Vectoring Control Gains
        self.tv_yaw_gain = float(params.get("tv_yaw_gain", 450.0))  # Nm yaw moment per rad/s yaw error

    def wheel_torques(
        self,
        throttle: ca.SX,
        regen_cmd: ca.SX,
        omega_fl: ca.SX,
        omega_fr: ca.SX,
        omega_rl: ca.SX,
        omega_rr: ca.SX,
        yaw_rate: ca.SX = 0.0,
        yaw_rate_target: ca.SX = 0.0,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        """Calculates 4 individual motor torques with Active Torque Vectoring and 80 kW ceiling.

        Returns:
            (T_wheel_fl, T_wheel_fr, T_wheel_rl, T_wheel_rr, P_elec)
        """
        # Average wheel speeds for front and rear
        omega_avg = 0.25 * (ca.fabs(omega_fl) + ca.fabs(omega_fr) + ca.fabs(omega_rl) + ca.fabs(omega_rr))
        omega_motor_avg = omega_avg * self.gear_ratio

        # 80 kW power ceiling hyperbolic scaling
        eps = 1e-3
        p_cap_per_corner = (self.total_power_ceiling / 4.0)
        t_power_cap = p_cap_per_corner / ca.fmax(omega_motor_avg, eps)
        t_max_motor = ca.fmin(self.motor_peak_torque_corner, t_power_cap)

        # Base forward drive torque demand per motor
        t_base_motor = throttle * t_max_motor - regen_cmd * ca.fmin(self.regen_max_torque_corner, t_max_motor)

        # Active Torque Vectoring (TV) differential torque:
        # Yaw rate tracking error: e_r = r_target - r
        # When target > actual (need yaw acceleration into turn), boost outer/left wheels and reduce inner/right wheels
        e_yaw = yaw_rate_target - yaw_rate
        delta_torque_tv = self.tv_yaw_gain * e_yaw / self.gear_ratio
        # Clip delta torque to maintain total power within limits
        delta_torque_tv = ca.fmin(ca.fmax(delta_torque_tv, -0.4 * t_max_motor), 0.4 * t_max_motor)

        # 4 Individual Motor Torques (Left gets +delta, Right gets -delta)
        t_m_fl = t_base_motor + delta_torque_tv
        t_m_fr = t_base_motor - delta_torque_tv
        t_m_rl = t_base_motor + delta_torque_tv
        t_m_rr = t_base_motor - delta_torque_tv

        # Output Wheel Torques (after planetary gear reduction)
        t_w_fl = t_m_fl * self.gear_ratio * self.eta_drivetrain
        t_w_fr = t_m_fr * self.gear_ratio * self.eta_drivetrain
        t_w_rl = t_m_rl * self.gear_ratio * self.eta_drivetrain
        t_w_rr = t_m_rr * self.gear_ratio * self.eta_drivetrain

        # Net Electrical Power draw (sum of all 4 corners)
        p_mech_fl = t_m_fl * (omega_fl * self.gear_ratio)
        p_mech_fr = t_m_fr * (omega_fr * self.gear_ratio)
        p_mech_rl = t_m_rl * (omega_rl * self.gear_ratio)
        p_mech_rr = t_m_rr * (omega_rr * self.gear_ratio)

        p_mech_total = p_mech_fl + p_mech_fr + p_mech_rl + p_mech_rr

        p_elec = ca.if_else(
            p_mech_total >= 0,
            p_mech_total / self.eta_drivetrain,
            p_mech_total * self.eta_drivetrain,
        )

        return t_w_fl, t_w_fr, t_w_rl, t_w_rr, p_elec
