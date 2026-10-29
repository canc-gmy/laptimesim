"""Data exporter for telemetry and simulation results into CSV and Parquet."""

from pathlib import Path
from typing import Literal
import pandas as pd
from laptimesim.solver.ocp import OCPResult
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.postprocessing.telemetry import TelemetryProcessor


class ResultExporter:
    """Exports simulation results to disk in CSV or Parquet format."""

    def __init__(self, result: OCPResult, track: CanonicalTrack):
        self.result = result
        self.processor = TelemetryProcessor(result, track)
        self.df = self.processor.to_dataframe()

    def export(self, output_dir: str | Path, base_name: str = "simulation_result", file_format: Literal["csv", "parquet"] = "csv") -> Path:
        """Save telemetry dataframe to output directory.

        Args:
            output_dir: Target folder to store files.
            base_name: Base name for exported files.
            file_format: 'csv' or 'parquet'.

        Returns:
            Path to exported file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if file_format == "parquet":
            file_dest = out_path / f"{base_name}.parquet"
            # PyArrow engine handles parquet writing
            self.df.to_parquet(file_dest, index=False)
        else:
            file_dest = out_path / f"{base_name}.csv"
            self.df.to_csv(file_dest, index=False)

        return file_dest
