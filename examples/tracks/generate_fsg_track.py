"""Generator for standard Formula Student Germany (FSG) Autocross/Endurance benchmark track (~1 km).

Follows FSG rules:
- Total length: ~1000 meters
- Minimum corner radius: ~4-5 m (hairpins)
- Maximum straight length: ~75-80 m
- Slaloms and chicanes
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_fsg_endurance_track(output_path: str = "examples/tracks/fsg_endurance_1km.csv") -> Path:
    """Builds a realistic 1,000m FSG Endurance circuit based on official regulations."""
    segments = [
        # (type, length, curvature)
        ("straight", 75.0, 0.0),            # Main straight
        ("turn_left", 35.0, 0.08),          # T1 medium left
        ("straight", 40.0, 0.0),            # Short straight
        ("hairpin_right", 28.0, -0.18),     # T2 tight hairpin right (R~5.5m)
        ("straight", 50.0, 0.0),            # Straight
        ("chicane_left", 20.0, 0.12),       # Chicane entry
        ("chicane_right", 20.0, -0.12),     # Chicane exit
        ("straight", 60.0, 0.0),            # Acceleration zone
        ("turn_right", 45.0, -0.06),        # Sweeper right
        ("straight", 35.0, 0.0),            # Straight
        ("hairpin_left", 30.0, 0.17),       # Hairpin left
        ("straight", 45.0, 0.0),            # Straight
        ("slalom_1", 20.0, -0.10),          # Slalom turn 1
        ("slalom_2", 20.0, 0.10),           # Slalom turn 2
        ("slalom_3", 20.0, -0.10),          # Slalom turn 3
        ("straight", 65.0, 0.0),            # Back straight
        ("turn_left", 40.0, 0.07),          # Left entry into stadium
        ("straight", 30.0, 0.0),            # Straight
        ("turn_left", 55.0, 0.09),          # Final stadium double-apex left
        ("straight", 70.0, 0.0),            # Pit straight return
        ("turn_right", 40.0, -0.08),        # Final corner right onto start/finish
        ("straight", 60.0, 0.0),            # Start/finish straight
    ]

    total_len = sum(s[1] for s in segments)
    num_pts = 200
    s_grid = np.linspace(0.0, total_len, num_pts)
    kappa_grid = np.zeros(num_pts)

    # Accumulate curvature profile along s
    cur_s = 0.0
    for seg_type, seg_len, seg_k in segments:
        mask = (s_grid >= cur_s) & (s_grid <= cur_s + seg_len)
        kappa_grid[mask] = seg_k
        cur_s += seg_len

    # Smooth transitions
    kappa_smooth = np.convolve(kappa_grid, [0.15, 0.70, 0.15], mode="same")

    # Formula Student track width is typically 3.0 m to 4.5 m (use 3.5 m left/right)
    w_left = np.full(num_pts, 3.5)
    w_right = np.full(num_pts, 3.5)

    df = pd.DataFrame({
        "s": s_grid,
        "kappa": kappa_smooth,
        "width_left": w_left,
        "width_right": w_right,
    })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated FSG Endurance Benchmark track: {out_file.resolve()} (Total length: {total_len:.1f} m)")
    return out_file


if __name__ == "__main__":
    generate_fsg_endurance_track()
