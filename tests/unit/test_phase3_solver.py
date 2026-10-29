"""Tests for Phase 3 Optimal Control Problem formulation and IPOPT solver."""

import pytest
import numpy as np
from pathlib import Path
from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.initial_guess import InitialGuessGenerator
from laptimesim.solver.ocp import CurvilinearLaptimeSolver


def create_test_track(length: float = 200.0, num_pts: int = 20) -> CanonicalTrack:
    s = np.linspace(0.0, length, num_pts)
    # Straight with a gentle curved chicane in the middle
    kappa = 0.01 * np.sin(2.0 * np.pi * s / length)
    w_l = np.full_like(s, 4.0)
    w_r = np.full_like(s, 4.0)
    return CanonicalTrack(s=s, kappa=kappa, width_left=w_l, width_right=w_r)


def test_initial_guess_generator():
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    config = FullSimulationConfig.from_toml(config_path)
    vehicle = VehicleModel(config)
    track = create_test_track()

    gen = InitialGuessGenerator(vehicle, track)
    guess = gen.generate(num_nodes=30)

    assert "s" in guess
    assert "X" in guess
    assert "U" in guess
    assert guess["X"].shape[1] == 30
    assert guess["U"].shape[1] == 30
    # Speeds should be physically positive and reasonable
    assert np.all(guess["X"][2, :] >= 5.0)


def test_curvilinear_solver_nlp_setup_and_execution():
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    config = FullSimulationConfig.from_toml(config_path)

    # Use a small node count for fast unit test convergence
    config.simulation.num_nodes = 20
    config.simulation.solver_max_iter = 100

    vehicle = VehicleModel(config)
    track = create_test_track(length=150.0, num_pts=20)

    solver = CurvilinearLaptimeSolver(vehicle, track, config)

    # Test spatial dynamics CasADi compilation
    spatial_f = solver._build_spatial_dynamics()
    assert spatial_f.n_in() == 3
    assert spatial_f.name_out()[0] == "dx_ds"
    assert spatial_f.name_out()[1] == "dt_ds"

    # Run solver on small test track
    result = solver.solve()

    assert result.lap_time > 0.0
    assert len(result.s) == 20
    assert len(result.t) == 20
    assert "vx" in result.states
    assert "delta_steer" in result.controls
    assert "ax" in result.telemetry
    assert result.solver_time > 0.0
