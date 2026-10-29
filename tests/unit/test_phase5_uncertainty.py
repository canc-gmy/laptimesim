"""Unit tests for Phase 5 Uncertainty Quantification and Monte Carlo Analysis."""

import pytest
import numpy as np
from pathlib import Path
from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.uncertainty.distributions import ParameterDistribution, UncertaintySampler
from laptimesim.uncertainty.monte_carlo import MonteCarloEngine
from laptimesim.uncertainty.sensitivity import SensitivityPlotter


def test_latin_hypercube_sampler():
    dists = [
        ParameterDistribution("vehicle.mass", "normal", mean=250.0, std=5.0),
        ParameterDistribution("tire.D", "uniform", mean=1.5, min_val=1.4, max_val=1.6),
    ]
    sampler = UncertaintySampler(dists, num_samples=20, seed=123)
    samples = sampler.sample_all()

    assert len(samples) == 20
    assert "vehicle.mass" in samples[0]
    assert "tire.D" in samples[0]

    masses = [s["vehicle.mass"] for s in samples]
    assert 230.0 < np.mean(masses) < 270.0


def test_monte_carlo_engine_and_sensitivity(tmp_path: Path):
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    config = FullSimulationConfig.from_toml(config_path)

    config.simulation.num_nodes = 12
    config.simulation.solver_max_iter = 50
    config.uncertainty = {
        "enabled": True,
        "num_samples": 4,
        "confidence_level": 0.90,
        "parameters": {
            "vehicle.mass": {"distribution": "normal", "mean": 230.0, "std": 5.0},
        },
    }

    s = np.linspace(0.0, 100.0, 12)
    kappa = np.zeros(12)
    track = CanonicalTrack(s=s, kappa=kappa, width_left=np.full(12, 4.0), width_right=np.full(12, 4.0))

    mc_engine = MonteCarloEngine(config, track)
    res = mc_engine.run(num_samples=4)

    assert res.num_samples_successful > 0
    assert res.mean_lap_time > 0.0
    assert res.ci_lower <= res.mean_lap_time <= res.ci_upper
    assert "vehicle.mass" in res.parameter_sensitivities

    # Test plotting
    plotter = SensitivityPlotter(res)
    dist_fig = plotter.plot_lap_time_distribution(tmp_path / "dist.png")
    assert (tmp_path / "dist.png").is_file()

    tornado_fig = plotter.plot_tornado_sensitivity(tmp_path / "tornado.png")
    assert (tmp_path / "tornado.png").is_file()
