"""High-level Simulation Orchestrator linking inputs, track, vehicle, solver, and outputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd

from laptimesim.core.config import FullSimulationConfig
from laptimesim.core.logging import setup_logging
from laptimesim.core.registry import model_registry
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.ocp import CurvilinearLaptimeSolver, OCPResult
from laptimesim.postprocessing.telemetry import TelemetryProcessor
from laptimesim.postprocessing.export import ResultExporter
from laptimesim.postprocessing.plots import SimulationPlotter
from laptimesim.uncertainty.monte_carlo import MonteCarloEngine, UncertaintyResult
from laptimesim.uncertainty.sensitivity import SensitivityPlotter
import laptimesim.track  # Ensure importers are loaded


@dataclass
class SimulationOutput:
    """Full bundle of simulation execution artifacts."""
    result: OCPResult
    telemetry_df: pd.DataFrame
    uncertainty: Optional[UncertaintyResult] = None
    exported_data_file: Optional[Path] = None
    gg_diagram_file: Optional[Path] = None
    telemetry_plot_file: Optional[Path] = None
    racing_line_plot_file: Optional[Path] = None
    distribution_plot_file: Optional[Path] = None
    tornado_plot_file: Optional[Path] = None


class LaptimeSimulator:
    """End-to-end simulator orchestrator."""

    def __init__(self, config: FullSimulationConfig):
        self.config = config
        self.logger = setup_logging(config.logging.level, config.logging.log_file)
        self.vehicle = VehicleModel(config)

        # Load Track
        importer_format = config.simulation.track_format
        track_path = Path(config.simulation.track_file)
        if not track_path.is_absolute():
            # If not found directly relative to CWD, check relative to examples/ or workspace root
            if not track_path.is_file():
                candidate = Path.cwd() / "examples" / track_path
                if candidate.is_file():
                    track_path = candidate
                else:
                    track_path = Path.cwd() / track_path
            else:
                track_path = track_path.resolve()

        importer = model_registry.create("track_importer", importer_format)
        self.track = importer.load(track_path)
        self.logger.info(f"Loaded track '{track_path.name}' (Length: {self.track.total_length:.1f} m)")

    def run(self) -> SimulationOutput:
        """Execute optimal control lap simulation and post-processing."""
        self.logger.info(f"Starting optimal control lap simulation: '{self.config.simulation.name}'")
        solver = CurvilinearLaptimeSolver(self.vehicle, self.track, self.config)
        result = solver.solve()

        self.logger.info(
            f"Solve finished. Lap time: {result.lap_time:.3f} s (Status: {'SUCCESS' if result.success else 'FAILED'}) in {result.solver_time:.2f} s"
        )

        # Post-processing
        processor = TelemetryProcessor(result, self.track)
        df = processor.to_dataframe()

        out_cfg = self.config.output
        out_dir = Path(out_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        exporter = ResultExporter(result, self.track)
        data_file = exporter.export(out_dir, base_name="telemetry", file_format=out_cfg.export_format)

        plotter = SimulationPlotter(df)
        gg_file = None
        telem_plot_file = None
        racing_line_file = None

        if out_cfg.gg_diagram:
            gg_file = out_dir / "gg_diagram.png"
            plotter.plot_gg_diagram(gg_file)

        if out_cfg.telemetry:
            telem_plot_file = out_dir / "telemetry_channels.png"
            plotter.plot_telemetry_channels(telem_plot_file)

        racing_line_file = out_dir / "racing_line.png"
        plotter.plot_track_racing_line(racing_line_file)

        # Uncertainty Quantification
        uq_result = None
        dist_plot_file = None
        tornado_file = None

        unc_cfg = self.config.uncertainty or {}
        if unc_cfg.get("enabled", False):
            self.logger.info("Executing Monte Carlo parameter uncertainty analysis...")
            mc_engine = MonteCarloEngine(self.config, self.track)
            uq_result = mc_engine.run()
            self.logger.info(
                f"Monte Carlo complete: Mean={uq_result.mean_lap_time:.3f} s, "
                f"95% CI=[{uq_result.ci_lower:.3f}, {uq_result.ci_upper:.3f}] s"
            )

            sens_plotter = SensitivityPlotter(uq_result)
            dist_plot_file = out_dir / "lap_time_distribution.png"
            sens_plotter.plot_lap_time_distribution(dist_plot_file)

            tornado_file = out_dir / "parameter_sensitivity_tornado.png"
            sens_plotter.plot_tornado_sensitivity(tornado_file)

        return SimulationOutput(
            result=result,
            telemetry_df=df,
            uncertainty=uq_result,
            exported_data_file=data_file,
            gg_diagram_file=gg_file,
            telemetry_plot_file=telem_plot_file,
            racing_line_plot_file=racing_line_file,
            distribution_plot_file=dist_plot_file,
            tornado_plot_file=tornado_file,
        )
