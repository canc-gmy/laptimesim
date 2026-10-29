"""Unit tests for high-fidelity subsystem models: Pacejka HiFi, K&C, 4WD Torque Vectoring, 2-RC Battery."""

import pytest
import casadi as ca
import numpy as np

from laptimesim.subsystems.tire.pacejka_hifi import PacejkaHiFiTire
from laptimesim.subsystems.suspension.kandc import KinematicComplianceSuspension
from laptimesim.subsystems.powertrain.four_wheel_drive_tv import FourWheelDriveTorqueVectoring
from laptimesim.subsystems.battery.ecm_2rc import TwoRCEquivalentCircuitBattery


def test_pacejka_hifi_load_and_thermal_sensitivity():
    tire = PacejkaHiFiTire({
        "B": 10.5, "C": 1.65, "D0_lat": 1.80, "dFz_lat": 0.0002,
        "T_opt": 85.0, "T_width": 25.0, "C_gamma": 2.0,
    })

    alpha_sym = ca.SX.sym("alpha")
    gamma_sym = ca.SX.sym("gamma")

    # Test load sensitivity: friction coefficient should decrease at higher Fz
    Fy_light = tire.lateral_force(alpha_sym, Fz=ca.SX(500.0), gamma=gamma_sym, T_surf=ca.SX(85.0))
    Fy_heavy = tire.lateral_force(alpha_sym, Fz=ca.SX(2000.0), gamma=gamma_sym, T_surf=ca.SX(85.0))

    f_eval = ca.Function("f_load_sens", [alpha_sym, gamma_sym], [Fy_light, Fy_heavy])
    val_l, val_h = f_eval(0.08, -0.035)

    mu_light = float(val_l) / 500.0
    mu_heavy = float(val_h) / 2000.0
    # Higher load yields lower effective mu
    assert mu_light > mu_heavy

    # Test thermal degradation: cold tire (30 C) vs optimal (85 C)
    Fy_cold = tire.lateral_force(alpha_sym, Fz=ca.SX(1000.0), gamma=gamma_sym, T_surf=ca.SX(30.0))
    Fy_hot = tire.lateral_force(alpha_sym, Fz=ca.SX(1000.0), gamma=gamma_sym, T_surf=ca.SX(85.0))
    f_therm = ca.Function("f_therm", [alpha_sym, gamma_sym], [Fy_cold, Fy_hot])
    c_val, h_val = f_therm(0.08, -0.035)
    assert float(h_val) > float(c_val)


def test_kandc_suspension_kinematics():
    kandc = KinematicComplianceSuspension({
        "front": {"static_camber_deg": -2.5, "camber_gain_deg_m": -35.0},
        "rear": {"static_camber_deg": -1.5, "camber_gain_deg_m": -25.0},
    })

    dummy = ca.SX.sym("dummy")
    gammas_and_toes = kandc.wheel_kinematics(
        zs_fl=ca.SX(0.015), zs_fr=ca.SX(-0.015), zs_rl=ca.SX(0.0), zs_rr=ca.SX(0.0),
        roll=ca.SX(0.02)
    )
    f_kandc = ca.Function("f_kandc", [dummy], [gammas_and_toes[0]])
    gamma_fl = f_kandc(0.0)
    assert float(gamma_fl) != 0.0


def test_4wd_torque_vectoring_powertrain():
    powertrain = FourWheelDriveTorqueVectoring({
        "motor_peak_torque_corner": 65.0,
        "total_power_ceiling": 80000.0,
        "gear_ratio": 11.5,
        "tv_yaw_gain": 450.0,
    })

    dummy = ca.SX.sym("dummy")
    t_fl, t_fr, t_rl, t_rr, p_elec = powertrain.wheel_torques(
        throttle=ca.SX(0.8),
        regen_cmd=ca.SX(0.0),
        omega_fl=ca.SX(60.0), omega_fr=ca.SX(60.0), omega_rl=ca.SX(60.0), omega_rr=ca.SX(60.0),
        yaw_rate=ca.SX(0.2), yaw_rate_target=ca.SX(0.5)
    )
    f_tv = ca.Function("f_tv", [dummy], [t_fl, t_fr, p_elec])
    t_left, t_right, p_el = f_tv(0.0)

    assert float(p_el) > 0.0
    assert float(t_left) > float(t_right)


def test_2rc_battery_model():
    batt = TwoRCEquivalentCircuitBattery({
        "capacity_kwh": 6.8,
        "voltage_nominal": 400.0,
        "r0": 0.035, "r1": 0.015, "r2": 0.020,
    })

    dummy = ca.SX.sym("dummy")
    d_soc, d_temp, v_term = batt.step(
        P_elec=ca.SX(60000.0),  # 60 kW discharge
        soc=ca.SX(0.85),
        temp=ca.SX(35.0),
        dt=ca.SX(0.01),
        v_rc1=ca.SX(1.5),
        v_rc2=ca.SX(2.0),
        car_speed=ca.SX(25.0)
    )
    f_batt = ca.Function("f_batt", [dummy], [d_soc, d_temp, v_term])
    ds, dt, vt = f_batt(0.0)

    assert float(ds) < 0.0
    assert float(dt) > 0.0
    # Terminal voltage should be physically reasonable (around nominal pack voltage)
    assert 300.0 < float(vt) < 460.0
