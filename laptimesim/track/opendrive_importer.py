"""OpenDRIVE (.xodr) XML track importer converting road geometries to CanonicalTrack."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple
import numpy as np

from laptimesim.core.registry import model_registry
from laptimesim.track.base import TrackImporterBase
from laptimesim.track.canonical import CanonicalTrack


@model_registry.register("track_importer", "opendrive")
class OpenDRIVEImporter(TrackImporterBase):
    """Imports OpenDRIVE (.xodr) files, parses planView geometry elements (line, arc, spiral, poly3),

    and samples the road reference line into a canonical curvilinear track.
    """

    def load(self, file_path: str | Path) -> CanonicalTrack:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"OpenDRIVE file not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        # Parse roads
        s_list: List[float] = []
        x_list: List[float] = []
        y_list: List[float] = []
        hdg_list: List[float] = []
        curv_list: List[float] = []

        roads = root.findall(".//road")
        if not roads:
            raise ValueError(f"No <road> elements found in OpenDRIVE XML: {path}")

        # Iterate over roads in sequence
        current_s = 0.0

        for road in roads:
            plan_view = road.find("planView")
            if plan_view is None:
                continue

            for geom in plan_view.findall("geometry"):
                s_geom = float(geom.get("s", 0.0))
                x_geom = float(geom.get("x", 0.0))
                y_geom = float(geom.get("y", 0.0))
                hdg_geom = float(geom.get("hdg", 0.0))
                length_geom = float(geom.get("length", 0.0))

                if length_geom <= 1e-4:
                    continue

                # Discretize geometry element
                num_substeps = max(5, int(length_geom / 2.0))
                ds_sub = length_geom / (num_substeps - 1)

                line = geom.find("line")
                arc = geom.find("arc")
                spiral = geom.find("spiral")

                for i in range(num_substeps):
                    ds_local = i * ds_sub
                    s_total = current_s + ds_local

                    if arc is not None:
                        curvature = float(arc.get("curvature", 0.0))
                        # Arc geometry integration
                        theta = hdg_geom + curvature * ds_local
                        if abs(curvature) > 1e-7:
                            x_pt = x_geom + (np.sin(theta) - np.sin(hdg_geom)) / curvature
                            y_pt = y_geom - (np.cos(theta) - np.cos(hdg_geom)) / curvature
                        else:
                            x_pt = x_geom + ds_local * np.cos(hdg_geom)
                            y_pt = y_geom + ds_local * np.sin(hdg_geom)
                    elif spiral is not None:
                        c_start = float(spiral.get("curvStart", 0.0))
                        c_end = float(spiral.get("curvEnd", 0.0))
                        curvature = c_start + (c_end - c_start) * (ds_local / length_geom)
                        theta = hdg_geom + (c_start * ds_local + 0.5 * (c_end - c_start) * (ds_local**2) / length_geom)
                        x_pt = x_geom + ds_local * np.cos(theta)
                        y_pt = y_geom + ds_local * np.sin(theta)
                    else:
                        # Straight line
                        curvature = 0.0
                        x_pt = x_geom + ds_local * np.cos(hdg_geom)
                        y_pt = y_geom + ds_local * np.sin(hdg_geom)

                    s_list.append(s_total)
                    x_list.append(x_pt)
                    y_list.append(y_pt)
                    curv_list.append(curvature)

                current_s += length_geom

        s_arr = np.array(s_list)
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)
        k_arr = np.array(curv_list)

        # Default standard track widths (4.0 m left and right)
        w_l = np.full_like(s_arr, 4.0)
        w_r = np.full_like(s_arr, 4.0)

        return CanonicalTrack(
            s=s_arr,
            kappa=k_arr,
            width_left=w_l,
            width_right=w_r,
            x=x_arr,
            y=y_arr,
            is_closed=True,
        )
