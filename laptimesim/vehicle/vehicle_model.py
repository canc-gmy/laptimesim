"""Full 18+ DOF Vehicle Dynamics Model in CasADi symbolic framework.

Includes:
- 6-DOF Chassis translation & rotation (vx, vy, vz, omega_x, omega_y, omega_z)
- 4 Wheel rotational degrees of freedom (omega_fl, omega_fr, omega_rl, omega_rr)
- 4 Suspension deflections and velocities (z_s_*, dz_s_*)
- Steering column compliance (delta_compl)
- Drivetrain compliance (theta_dt)
- Battery State of Charge (SoC) and temperature (T_batt)
"""

from typing import Dict, Any, Tuple
import casadi as ca
from laptimesim.core.config import FullSimulationConfig
from laptimesim.core.registry import model_registry
import laptimesim.subsystems.tire
import laptimesim.subsystems.aero
import laptimesim.subsystems.suspension
import laptimesim.subsystems.powertrain
import laptimesim.subsystems.battery


class VehicleModel:
    """Combines chassis multi-body dynamics with all registered subsystems."""

    def __init__(self, config: FullSimulationConfig):
        self.config = config
        v = config.vehicle
        self.mass = v.mass + v.driver_mass
        self.Ixx = v.inertia.Ixx
        self.Iyy = v.inertia.Iyy
        self.Izz = v.inertia.Izz
        self.L = v.wheelbase
        self.a = v.wheelbase * (1.0 - v.cg_longitudinal)  # Distance from CG to front axle
        self.b = v.wheelbase * v.cg_longitudinal           # Distance from CG to rear axle
        self.tf = v.track_front
        self.tr = v.track_rear
        self.h_cg = v.cg_height
        self.g = 9.81

        # Instantiate subsystems from registry
        self.tire_front = model_registry.create(
            "tire", config.tire.get("model", "pacejka_mf52"), config.tire.get("front", config.tire)
        )
        self.tire_rear = model_registry.create(
            "tire", config.tire.get("model", "pacejka_mf52"), config.tire.get("rear", config.tire)
        )
        self.aero = model_registry.create(
            "aero", config.aero.get("model", "constant"), config.aero
        )
        self.suspension = model_registry.create(
            "suspension", config.suspension.get("model", "linear"), config.suspension
        )
        self.powertrain = model_registry.create(
            "powertrain", config.powertrain.get("model", "electric"), config.powertrain
        )
        battery_params = config.battery or {"model": "ecm"}
        self.battery = model_registry.create(
            "battery", battery_params.get("model", "ecm"), battery_params
        )

        # Wheel parameters
        self.r_wheel = self.tire_front.radius
        self.I_wheel = 0.8  # kg*m^2 per wheel assembly

        # Steering compliance parameters
        self.k_steer = 5000.0   # Nm/rad
        self.c_steer = 80.0     # Nms/rad
        # Drivetrain compliance parameters
        self.k_dt = 15000.0     # Nm/rad
        self.c_dt = 120.0       # Nms/rad

    def state_names(self) -> list[str]:
        """Names of all states in the vehicle model."""
        return [
            # Chassis velocities & body orientation
            "vx", "vy", "vz", "phi", "theta", "psi_dot", "phi_dot", "theta_dot",
            # 4 Wheel spin rates
            "omega_fl", "omega_fr", "omega_rl", "omega_rr",
            # 4 Suspension deflections
            "zs_fl", "zs_fr", "zs_rl", "zs_rr",
            # 4 Suspension deflection rates
            "dzs_fl", "dzs_fr", "dzs_rl", "dzs_rr",
            # Compliances (steering column & drivetrain torsion)
            "delta_compl", "theta_dt",
            # Battery states
            "soc", "t_batt",
        ]

    def control_names(self) -> list[str]:
        """Names of all control inputs."""
        return [
            "delta_steer",   # Steering wheel / rack angle command [rad]
            "throttle",      # Throttle command [0.0 - 1.0]
            "brake_pressure",# Brake line pressure [0.0 - 1.0]
            "regen_cmd",     # Regen braking fraction [0.0 - 1.0]
        ]

    def forward_dynamics(
        self, x: ca.SX, u: ca.SX, dt: ca.SX = 0.01
    ) -> Tuple[ca.SX, Dict[str, ca.SX]]:
        """Compute the full system state derivative dx/dt given state x and control u.

        Returns:
            Tuple of:
                - x_dot: CasADi SX column vector of length len(state_names())
                - telemetry: Dictionary of key intermediate CasADi expressions
        """
        # Unpack state vector
        (
            vx, vy, vz, phi, theta, r_yaw, p_roll, q_pitch,
            omega_fl, omega_fr, omega_rl, omega_rr,
            zs_fl, zs_fr, zs_rl, zs_rr,
            dzs_fl, dzs_fr, dzs_rl, dzs_rr,
            delta_compl, theta_dt,
            soc, t_batt,
        ) = [x[i] for i in range(len(self.state_names()))]

        # Unpack control vector
        delta_cmd, throttle, brake_cmd, regen_cmd = [u[i] for i in range(len(self.control_names()))]

        # Effective road wheel angle after steering compliance
        delta_eff = delta_cmd + delta_compl

        # Suspension forces transmitted to chassis
        F_susp_fl, F_susp_fr, F_susp_rl, F_susp_rr = self.suspension.suspension_forces(
            zs_fl, zs_fr, zs_rl, zs_rr, dzs_fl, dzs_fr, dzs_rl, dzs_rr
        )

        # Aerodynamic forces
        ride_h_f = 0.035 - 0.5 * (zs_fl + zs_fr)
        ride_h_r = 0.040 - 0.5 * (zs_rl + zs_rr)
        F_drag, F_down_f, F_down_r, F_side = self.aero.forces_and_moments(
            vx, vy, vz, ride_h_f, ride_h_r, theta, phi
        )

        # Normal loads on tires (static + aero + suspension dynamic load)
        Fz_static_f = 0.5 * self.mass * self.g * (self.b / self.L)
        Fz_static_r = 0.5 * self.mass * self.g * (self.a / self.L)

        Fz_fl = ca.fmax(10.0, Fz_static_f + 0.5 * F_down_f + F_susp_fl)
        Fz_fr = ca.fmax(10.0, Fz_static_f + 0.5 * F_down_f + F_susp_fr)
        Fz_rl = ca.fmax(10.0, Fz_static_r + 0.5 * F_down_r + F_susp_rl)
        Fz_rr = ca.fmax(10.0, Fz_static_r + 0.5 * F_down_r + F_susp_rr)

        # Kinematic wheel velocities at contact patches
        # Front left & right velocities in wheel frame
        v_x_fl = vx - 0.5 * self.tf * r_yaw
        v_y_fl = vy + self.a * r_yaw
        v_x_fr = vx + 0.5 * self.tf * r_yaw
        v_y_fr = vy + self.a * r_yaw

        v_x_rl = vx - 0.5 * self.tr * r_yaw
        v_y_rl = vy - self.b * r_yaw
        v_x_rr = vx + 0.5 * self.tr * r_yaw
        v_y_rr = vy - self.b * r_yaw

        # Transform front velocities by road wheel angle delta_eff
        cos_d = ca.cos(delta_eff)
        sin_d = ca.sin(delta_eff)
        v_long_fl = v_x_fl * cos_d + v_y_fl * sin_d
        v_lat_fl = -v_x_fl * sin_d + v_y_fl * cos_d
        v_long_fr = v_x_fr * cos_d + v_y_fr * sin_d
        v_lat_fr = -v_x_fr * sin_d + v_y_fr * cos_d

        v_long_rl = v_x_rl
        v_lat_rl = v_y_rl
        v_long_rr = v_x_rr
        v_lat_rr = v_y_rr

        # Slip angles and slip ratios
        eps_v = 0.5
        alpha_fl = -ca.atan(v_lat_fl / ca.fmax(ca.fabs(v_long_fl), eps_v))
        alpha_fr = -ca.atan(v_lat_fr / ca.fmax(ca.fabs(v_long_fr), eps_v))
        alpha_rl = -ca.atan(v_lat_rl / ca.fmax(ca.fabs(v_long_rl), eps_v))
        alpha_rr = -ca.atan(v_lat_rr / ca.fmax(ca.fabs(v_long_rr), eps_v))

        kappa_fl = (omega_fl * self.r_wheel - v_long_fl) / ca.fmax(ca.fabs(v_long_fl), eps_v)
        kappa_fr = (omega_fr * self.r_wheel - v_long_fr) / ca.fmax(ca.fabs(v_long_fr), eps_v)
        kappa_rl = (omega_rl * self.r_wheel - v_long_rl) / ca.fmax(ca.fabs(v_long_rl), eps_v)
        kappa_rr = (omega_rr * self.r_wheel - v_long_rr) / ca.fmax(ca.fabs(v_long_rr), eps_v)

        # Tire forces in tire coordinate frames
        Fx_fl_t, Fy_fl_t = self.tire_front.combined_forces(alpha_fl, kappa_fl, Fz_fl)
        Fx_fr_t, Fy_fr_t = self.tire_front.combined_forces(alpha_fr, kappa_fr, Fz_fr)
        Fx_rl, Fy_rl = self.tire_rear.combined_forces(alpha_rl, kappa_rl, Fz_rl)
        Fx_rr, Fy_rr = self.tire_rear.combined_forces(alpha_rr, kappa_rr, Fz_rr)

        # Rotate front forces back to vehicle body frame
        Fx_fl = Fx_fl_t * cos_d - Fy_fl_t * sin_d
        Fy_fl = Fx_fl_t * sin_d + Fy_fl_t * cos_d
        Fx_fr = Fx_fr_t * cos_d - Fy_fr_t * sin_d
        Fy_fr = Fx_fr_t * sin_d + Fy_fr_t * cos_d

        # Powertrain & braking torques
        T_pw_fl, T_pw_fr, T_pw_rl, T_pw_rr, P_elec = self.powertrain.wheel_torques(
            throttle, regen_cmd, omega_fl, omega_fr, omega_rl, omega_rr
        )

        # Hydraulic braking torques
        brakes_cfg = self.config.brakes or {}
        max_f_f = float(brakes_cfg.get("max_force_front", 5000.0))
        max_f_r = float(brakes_cfg.get("max_force_rear", 3000.0))
        T_brk_fl = brake_cmd * max_f_f * self.r_wheel * ca.sign(omega_fl)
        T_brk_fr = brake_cmd * max_f_f * self.r_wheel * ca.sign(omega_fr)
        T_brk_rl = brake_cmd * max_f_r * self.r_wheel * ca.sign(omega_rl)
        T_brk_rr = brake_cmd * max_f_r * self.r_wheel * ca.sign(omega_rr)

        # Total forces and moments on chassis in body frame
        Fx_total = (Fx_fl + Fx_fr + Fx_rl + Fx_rr) - F_drag
        Fy_total = (Fy_fl + Fy_fr + Fy_rl + Fy_rr) + F_side
        Fz_total = self.mass * self.g - (Fz_fl + Fz_fr + Fz_rl + Fz_rr)

        # Yaw moment Mz
        Mz_total = (
            self.a * (Fy_fl + Fy_fr) - self.b * (Fy_rl + Fy_rr)
            + 0.5 * self.tf * (Fx_fr - Fx_fl)
            + 0.5 * self.tr * (Fx_rr - Fx_rl)
        )

        # Roll moment Mx
        Mx_total = (
            0.5 * self.tf * (F_susp_fl - F_susp_fr)
            + 0.5 * self.tr * (F_susp_rl - F_susp_rr)
            - self.h_cg * Fy_total
        )

        # Pitch moment My
        My_total = (
            -self.a * (F_susp_fl + F_susp_fr)
            + self.b * (F_susp_rl + F_susp_rr)
            + self.h_cg * Fx_total
        )

        # 6-DOF Chassis acceleration equations
        ax = (Fx_total / self.mass) + vy * r_yaw - vz * q_pitch
        ay = (Fy_total / self.mass) - vx * r_yaw + vz * p_roll
        az = (Fz_total / self.mass) + vx * q_pitch - vy * p_roll

        p_dot = Mx_total / self.Ixx
        q_dot = My_total / self.Iyy
        r_dot = Mz_total / self.Izz

        # 4 Wheel spin dynamics: I_w * d(omega)/dt = T_drive - T_brake - Fx * R
        d_omega_fl = (T_pw_fl - T_brk_fl - Fx_fl_t * self.r_wheel) / self.I_wheel
        d_omega_fr = (T_pw_fr - T_brk_fr - Fx_fr_t * self.r_wheel) / self.I_wheel
        d_omega_rl = (T_pw_rl - T_brk_rl - Fx_rl * self.r_wheel) / self.I_wheel
        d_omega_rr = (T_pw_rr - T_brk_rr - Fx_rr * self.r_wheel) / self.I_wheel

        # Suspension travel derivatives: dzs/dt = dzs, ddzs/dt = F_net / m_unsprung
        # For lap sim, quarter-car deflection velocity matches chassis bounce/pitch/roll kinematics
        ddzs_fl = - (F_susp_fl / 15.0)  # ~15kg unsprung mass
        ddzs_fr = - (F_susp_fr / 15.0)
        ddzs_rl = - (F_susp_rl / 15.0)
        ddzs_rr = - (F_susp_rr / 15.0)

        # Compliances: Steering column torque & Drivetrain torsion
        aligning_torque = (Fy_fl_t + Fy_fr_t) * 0.025  # pneumatic trail
        d_delta_compl = (aligning_torque - self.k_steer * delta_compl) / self.c_steer
        d_theta_dt = (T_pw_rl + T_pw_rr - self.k_dt * theta_dt) / self.c_dt

        # Battery dynamics
        d_soc_dt, d_t_batt_dt, v_term = self.battery.step(P_elec, soc, t_batt, dt)

        # Assemble full state derivative vector dx/dt
        x_dot = ca.vertcat(
            ax, ay, az, p_roll, q_pitch, r_dot, p_dot, q_dot,
            d_omega_fl, d_omega_fr, d_omega_rl, d_omega_rr,
            dzs_fl, dzs_fr, dzs_rl, dzs_rr,
            ddzs_fl, ddzs_fr, ddzs_rl, ddzs_rr,
            d_delta_compl, d_theta_dt,
            d_soc_dt, d_t_batt_dt,
        )

        telemetry = {
            "ax": ax,
            "ay": ay,
            "Fx_total": Fx_total,
            "Fy_total": Fy_total,
            "Fz_fl": Fz_fl, "Fz_fr": Fz_fr, "Fz_rl": Fz_rl, "Fz_rr": Fz_rr,
            "alpha_fl": alpha_fl, "alpha_fr": alpha_fr, "alpha_rl": alpha_rl, "alpha_rr": alpha_rr,
            "kappa_fl": kappa_fl, "kappa_fr": kappa_fr, "kappa_rl": kappa_rl, "kappa_rr": kappa_rr,
            "P_elec": P_elec,
            "V_terminal": v_term,
        }

        return x_dot, telemetry
