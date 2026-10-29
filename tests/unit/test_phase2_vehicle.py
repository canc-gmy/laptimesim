"""Tests for Phase 2 Vehicle Dynamics and Subsystems."""

import pytest
import casadi as ca
import numpy as np
from pathlib import Path
from laptimesim.core.config import FullSimulationConfig
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.subsystems.tire.pacejka import PacejkaMF52
from laptimesim.subsystems.aero.constant import ConstantAeroModel
from laptimesim.subsystems.suspension.linear import LinearSuspension
from laptimesim.subsystems.powertrain.electric import ElectricPowertrain
from laptimesim.subsystems.battery.ecm import EquivalentCircuitBattery


def test_pacejka_tire_model():
    params = {"B": 10.0, "C": 1.6, "D": 1.5, "E": -0.5, "radius": 0.23}
    tire = PacejkaMF52(params)

    alpha = ca.SX.sym("alpha")
    kappa = ca.SX.sym("kappa")
    Fz = ca.SX.sym("Fz")

    Fy = tire.lateral_force(alpha, Fz)
    Fx = tire.longitudinal_force(kappa, Fz)
    Fx_c, Fy_c = tire.combined_forces(alpha, kappa, Fz)

    # Compile CasADi function to test numerical evaluation
    f_eval = ca.Function("f_eval", [alpha, kappa, Fz], [Fx, Fy, Fx_c, Fy_c])
    res_fx, res_fy, res_fxc, res_fyc = f_eval(0.05, 0.05, 1000.0)

    assert float(res_fx) > 0.0
    assert float(res_fy) > 0.0
    # Combined slip should reduce pure forces
    assert float(res_fxc) < float(res_fx)
    assert float(res_fyc) < float(res_fy)


def test_aero_constant():
    aero = ConstantAeroModel({"Cl": 3.0, "Cd": 1.2, "frontal_area": 1.0})
    drag, df_f, df_r, side = aero.forces_and_moments(
        vx=ca.SX(30.0), vy=ca.SX(0.0), vz=ca.SX(0.0),
        ride_height_f=ca.SX(0.03), ride_height_r=ca.SX(0.04),
        pitch=ca.SX(0.0), roll=ca.SX(0.0)
    )
    dummy_in = ca.SX.sym("dummy_in")
    f_aero = ca.Function("f_aero", [dummy_in], [drag, df_f, df_r])
    d, df, dr = f_aero(0.0)
    assert float(d) > 0.0
    assert float(df) > 0.0
    assert float(dr) > 0.0


def test_electric_powertrain_and_battery():
    powertrain = ElectricPowertrain({
        "motor_peak_torque": 230.0,
        "motor_peak_power": 80000.0,
        "gear_ratio": 3.5,
    })
    battery = EquivalentCircuitBattery({
        "capacity_kwh": 6.5,
        "voltage_nominal": 400.0,
        "internal_resistance": 0.05,
    })

    t_fl, t_fr, t_rl, t_rr, P_elec = powertrain.wheel_torques(
        throttle=ca.SX(1.0),
        regen_cmd=ca.SX(0.0),
        omega_fl=ca.SX(100.0),
        omega_fr=ca.SX(100.0),
        omega_rl=ca.SX(100.0),
        omega_rr=ca.SX(100.0),
    )
    dummy = ca.SX.sym("dummy")
    f_pw = ca.Function("f_pw", [dummy], [t_rl, P_elec])
    t_wheel, p_el = f_pw(0.0)
    assert float(t_wheel) > 0.0
    assert float(p_el) > 0.0

    d_soc, d_temp, v_term = battery.step(P_elec=p_el, soc=ca.SX(0.9), temp=ca.SX(30.0), dt=ca.SX(0.01))
    f_batt = ca.Function("f_batt", [dummy], [d_soc, d_temp, v_term])
    ds, dt, vt = f_batt(0.0)
    assert float(ds) < 0.0  # Discharging reduces SoC
    assert float(dt) > 0.0  # Joule heating increases temperature
    assert float(vt) < 450.0


def test_full_vehicle_model_derivatives():
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    config = FullSimulationConfig.from_toml(config_path)

    vehicle = VehicleModel(config)
    n_states = len(vehicle.state_names())
    n_controls = len(vehicle.control_names())

    assert n_states >= 22
    assert n_controls == 4

    x_sym = ca.SX.sym("x", n_states)
    u_sym = ca.SX.sym("u", n_controls)

    x_dot, telem = vehicle.forward_dynamics(x_sym, u_sym)
    assert x_dot.shape == (n_states, 1)

    f_dynamics = ca.Function("f_dyn", [x_sym, u_sym], [x_dot, telem["ax"], telem["ay"]])

    # Test numerical evaluation at forward driving state (vx = 20 m/s) with drive slip
    x0 = np.zeros(n_states)
    x0[0] = 20.0  # vx = 20 m/s
    # In drive slip, wheels spin faster than road speed (omega > v/r)
    x0[8:12] = 20.5 / 0.2286  # wheel speeds omega ~ 89.6 rad/s
    x0[20] = 0.95  # soc = 95%
    x0[21] = 25.0  # t_batt = 25 C

    u0 = np.array([0.0, 0.8, 0.0, 0.0])  # 80% throttle

    res_xdot, ax_val, ay_val = f_dynamics(x0, u0)
    res_xdot = np.array(res_xdot).flatten()

    assert len(res_xdot) == n_states
    assert float(ax_val) > 0.0
