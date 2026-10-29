"""CSV Track Importer for waypoint and curvature tracks."""

from pathlib import Path
import numpy as np
import pandas as pd
from laptimesim.core.registry import model_registry
from laptimesim.track.base import TrackImporterBase
from laptimesim.track.canonical import CanonicalTrack


@model_registry.register("track_importer", "csv")
class CSVTrackImporter(TrackImporterBase):
    """Imports track from CSV containing waypoints (x, y) or curvature profile (s, kappa)."""

    def load(self, file_path: str | Path) -> CanonicalTrack:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Track CSV file not found: {path}")

        df = pd.read_csv(path)
        columns = [c.strip().lower() for c in df.columns]
        df.columns = columns

        # Case 1: Curvilinear representation already given (s, kappa)
        if "s" in df.columns and "kappa" in df.columns:
            s = df["s"].to_numpy(dtype=float)
            kappa = df["kappa"].to_numpy(dtype=float)
            width_left = df["width_left"].to_numpy(dtype=float) if "width_left" in df.columns else np.full_like(s, 4.0)
            width_right = df["width_right"].to_numpy(dtype=float) if "width_right" in df.columns else np.full_like(s, 4.0)
            elevation = df["elevation"].to_numpy(dtype=float) if "elevation" in df.columns else np.zeros_like(s)
            banking = df["banking"].to_numpy(dtype=float) if "banking" in df.columns else np.zeros_like(s)
            x = df["x"].to_numpy(dtype=float) if "x" in df.columns else np.zeros(0)
            y = df["y"].to_numpy(dtype=float) if "y" in df.columns else np.zeros(0)

            return CanonicalTrack(
                s=s,
                kappa=kappa,
                width_left=width_left,
                width_right=width_right,
                elevation=elevation,
                banking=banking,
                x=x,
                y=y,
                is_closed=True,
            )

        # Case 2: Cartesian 2D waypoints (x, y)
        elif "x" in df.columns and "y" in df.columns:
            x = df["x"].to_numpy(dtype=float)
            y = df["y"].to_numpy(dtype=float)
            n_pts = len(x)

            # Compute arc-length s
            dx = np.diff(x)
            dy = np.diff(y)
            seg_lengths = np.hypot(dx, dy)
            s = np.zeros(n_pts)
            s[1:] = np.cumsum(seg_lengths)

            # Compute curvature kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
            xp = np.gradient(x, s)
            yp = np.gradient(y, s)
            xpp = np.gradient(xp, s)
            ypp = np.gradient(yp, s)

            denom = (xp**2 + yp**2)**1.5
            denom = np.where(denom < 1e-9, 1e-9, denom)
            kappa = (xp * ypp - yp * xpp) / denom

            width_left = df["width_left"].to_numpy(dtype=float) if "width_left" in df.columns else np.full_like(s, 4.0)
            width_right = df["width_right"].to_numpy(dtype=float) if "width_right" in df.columns else np.full_like(s, 4.0)

            return CanonicalTrack(
                s=s,
                kappa=kappa,
                width_left=width_left,
                width_right=width_right,
                x=x,
                y=y,
                is_closed=True,
            )

        else:
            raise ValueError(
                f"Unsupported CSV format in {path}. Expected columns ('s', 'kappa') or ('x', 'y'). "
                f"Found columns: {list(df.columns)}"
            )
