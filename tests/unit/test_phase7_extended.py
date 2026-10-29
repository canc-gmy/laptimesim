"""Unit tests for Phase 7 Extended Models: Aero maps, Nonlinear suspension, and OpenDRIVE parsing."""

import pytest
import numpy as np
import casadi as ca
from pathlib import Path

from laptimesim.subsystems.aero.aero_map import AeroMapModel
from laptimesim.subsystems.suspension.nonlinear import NonlinearSuspension
from laptimesim.track.opendrive_importer import OpenDRIVEImporter


def test_aero_map_model():
    aero = AeroMapModel({
        "frontal_area": 1.1,
        "air_density": 1.2,
    })

    drag, df_f, df_r, side = aero.forces_and_moments(
        vx=ca.SX(30.0), vy=ca.SX(0.0), vz=ca.SX(0.0),
        ride_height_f=ca.SX(0.030), ride_height_r=ca.SX(0.040),
        pitch=ca.SX(0.0), roll=ca.SX(0.0)
    )

    f_eval = ca.Function("f_eval_map", [ca.SX.sym("d")], [drag, df_f, df_r])
    d_val, df_f_val, df_r_val = f_eval(0.0)

    assert float(d_val) > 0.0
    assert float(df_f_val) > 0.0
    assert float(df_r_val) > 0.0


def test_nonlinear_suspension_bump_stops():
    susp = NonlinearSuspension({
        "front": {
            "spring_rate": 35000.0,
            "bump_stop_rate": 100000.0,
            "bump_stop_clearance": 0.010,
            "damping_comp": 1500.0,
            "damping_reb": 2200.0,
            "arb_rate": 5000.0,
        },
        "rear": {
            "spring_rate": 40000.0,
            "bump_stop_rate": 120000.0,
            "bump_stop_clearance": 0.010,
            "damping_comp": 1700.0,
            "damping_reb": 2400.0,
            "arb_rate": 4000.0,
        },
    })

    # Under bump stop clearance (zs = 0.005 m)
    f_linear, _, _, _ = susp.suspension_forces(
        ca.SX(0.005), ca.SX(0.005), ca.SX(0.005), ca.SX(0.005),
        ca.SX(0.0), ca.SX(0.0), ca.SX(0.0), ca.SX(0.0)
    )

    # Beyond bump stop clearance (zs = 0.025 m)
    f_bump, _, _, _ = susp.suspension_forces(
        ca.SX(0.025), ca.SX(0.025), ca.SX(0.025), ca.SX(0.025),
        ca.SX(0.0), ca.SX(0.0), ca.SX(0.0), ca.SX(0.0)
    )

    f_eval = ca.Function("f_eval_susp", [ca.SX.sym("d")], [f_linear, f_bump])
    val_lin, val_bump = f_eval(0.0)

    # Bump stop should add progressive stiffness
    assert float(val_lin) == pytest.approx(35000.0 * 0.005, rel=1e-3)
    assert float(val_bump) > 35000.0 * 0.025


def test_opendrive_importer():
    xodr_path = Path(__file__).resolve().parent.parent.parent / "examples" / "tracks" / "sample_oval.xodr"
    importer = OpenDRIVEImporter()
    track = importer.load(xodr_path)

    assert track.num_points > 10
    assert track.total_length == pytest.approx(300.0, rel=1e-2)
    assert len(track.x) == track.num_points
    assert len(track.y) == track.num_points
    # Curvature should be zero on straights and positive on arcs
    assert np.any(track.kappa > 0.01)
    assert np.any(track.kappa == 0.0)
