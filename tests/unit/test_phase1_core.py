"""Unit tests for registry, configuration parsing, and track import."""

import pytest
import numpy as np
from pathlib import Path
from laptimesim.core.registry import Registry, SubsystemRegistry, model_registry
from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.track.csv_importer import CSVTrackImporter


def test_registry_registration_and_creation():
    reg = Registry("test_category")

    @reg.register("dummy")
    class DummyModel:
        def __init__(self, value: int = 42):
            self.value = value

    assert "dummy" in reg.list_available()
    instance = reg.create("dummy", value=99)
    assert instance.value == 99

    with pytest.raises(ValueError):
        @reg.register("dummy")
        class Duplicate:
            pass

    with pytest.raises(KeyError):
        reg.get("non_existent")


def test_subsystem_registry_global():
    sub_reg = SubsystemRegistry()

    @sub_reg.register("tire", "pacejka")
    class Pacejka:
        pass

    assert "tire" in sub_reg.list_all()
    assert "pacejka" in sub_reg.category("tire").list_available()


def test_canonical_track_generation_and_resampling():
    s = np.array([0.0, 50.0, 100.0, 150.0])
    kappa = np.array([0.0, 0.01, 0.01, 0.0])
    w_l = np.array([3.5, 3.5, 3.5, 3.5])
    w_r = np.array([3.5, 3.5, 3.5, 3.5])

    track = CanonicalTrack(s=s, kappa=kappa, width_left=w_l, width_right=w_r)
    assert track.total_length == 150.0
    assert track.num_points == 4
    assert len(track.x) == 4
    assert len(track.y) == 4

    # Test resampling
    resampled = track.resample(10)
    assert resampled.num_points == 10
    assert pytest.approx(resampled.total_length) == 150.0


def test_csv_track_importer(tmp_path: Path):
    csv_file = tmp_path / "track_sample.csv"
    csv_file.write_text("s,kappa,width_left,width_right\n0.0,0.0,4.0,4.0\n10.0,0.02,4.0,4.0\n20.0,0.0,4.0,4.0\n")

    importer = CSVTrackImporter()
    track = importer.load(csv_file)
    assert track.num_points == 3
    assert track.total_length == 20.0
    assert track.kappa[1] == 0.02


def test_config_from_toml(tmp_path: Path):
    toml_file = tmp_path / "test_config.toml"
    toml_content = """
    [simulation]
    name = "Test Run"
    track_file = "test.csv"
    track_format = "csv"
    num_nodes = 50
    solver_max_iter = 1000
    solver_tolerance = 1e-5

    [vehicle]
    mass = 220.0
    driver_mass = 70.0
    wheelbase = 1.55
    track_front = 1.20
    track_rear = 1.18
    cg_height = 0.29
    cg_longitudinal = 0.48

    [vehicle.inertia]
    Ixx = 80.0
    Iyy = 110.0
    Izz = 90.0
    Ixz = 0.0

    [tire]
    model = "pacejka_mf52"

    [aero]
    model = "constant"

    [suspension]
    model = "linear"

    [powertrain]
    model = "electric"
    """
    toml_file.write_text(toml_content)
    cfg = FullSimulationConfig.from_toml(toml_file)
    assert cfg.simulation.name == "Test Run"
    assert cfg.vehicle.mass == 220.0
    assert cfg.vehicle.inertia.Ixx == 80.0
