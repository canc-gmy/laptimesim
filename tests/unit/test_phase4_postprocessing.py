"""Unit tests for Phase 4 Post-Processing, Export, and Plots."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.ocp import CurvilinearLaptimeSolver
from laptimesim.postprocessing.telemetry import TelemetryProcessor
from laptimesim.postprocessing.export import ResultExporter
from laptimesim.postprocessing.plots import SimulationPlotter


def test_telemetry_processor_and_exporter(tmp_path: Path):
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    config = FullSimulationConfig.from_toml(config_path)
    config.simulation.num_nodes = 15
    config.simulation.solver_max_iter = 50

    s = np.linspace(0.0, 100.0, 15)
    kappa = np.zeros(15)
    track = CanonicalTrack(s=s, kappa=kappa, width_left=np.full(15, 4.0), width_right=np.full(15, 4.0))
    vehicle = VehicleModel(config)

    solver = CurvilinearLaptimeSolver(vehicle, track, config)
    res = solver.solve()

    processor = TelemetryProcessor(res, track)
    df = processor.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert "speed_km_h" in df.columns
    assert "ax_g" in df.columns
    assert "vehicle_x_m" in df.columns
    assert len(df) == 15

    # Test CSV export
    exporter = ResultExporter(res, track)
    csv_file = exporter.export(tmp_path, base_name="test_telem", file_format="csv")
    assert csv_file.is_file()
    assert csv_file.stat().st_size > 0

    # Test Plot generation
    plotter = SimulationPlotter(df)
    gg_fig = plotter.plot_gg_diagram(tmp_path / "gg.png")
    assert (tmp_path / "gg.png").is_file()

    telem_fig = plotter.plot_telemetry_channels(tmp_path / "channels.png")
    assert (tmp_path / "channels.png").is_file()

    track_fig = plotter.plot_track_racing_line(tmp_path / "racing_line.png")
    assert (tmp_path / "racing_line.png").is_file()
