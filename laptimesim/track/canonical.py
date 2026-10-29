"""Canonical track representation in curvilinear (Frenet) coordinates."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CanonicalTrack:
    """Represents a track discretized along arc-length s in Frenet coordinates.

    Attributes:
        s: Arc length along the centerline from start [m], array of shape (N,).
        kappa: Horizontal curvature (1/R) along centerline [rad/m], positive for left turn.
        width_left: Allowed track boundary to the left of centerline [m].
        width_right: Allowed track boundary to the right of centerline [m].
        elevation: Elevation z along centerline [m].
        banking: Banking angle [rad].
        x: Global Cartesian X coordinate of centerline [m] (for plotting/telemetry).
        y: Global Cartesian Y coordinate of centerline [m] (for plotting/telemetry).
        is_closed: Whether the track is a closed circuit (loop).
    """
    s: np.ndarray
    kappa: np.ndarray
    width_left: np.ndarray
    width_right: np.ndarray
    elevation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    banking: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    is_closed: bool = True

    def __post_init__(self):
        n = len(self.s)
        if len(self.kappa) != n or len(self.width_left) != n or len(self.width_right) != n:
            raise ValueError("All track arrays (s, kappa, width_left, width_right) must have the same length.")

        if len(self.elevation) == 0:
            self.elevation = np.zeros(n)
        if len(self.banking) == 0:
            self.banking = np.zeros(n)
        if len(self.x) == 0 or len(self.y) == 0:
            # Integrate Cartesian coordinates from curvature if not provided
            self._reconstruct_cartesian()

    @property
    def total_length(self) -> float:
        """Total length of the track circuit in meters."""
        return float(self.s[-1] - self.s[0]) if len(self.s) > 1 else 0.0

    @property
    def num_points(self) -> int:
        """Number of points along the track."""
        return len(self.s)

    def _reconstruct_cartesian(self) -> None:
        """Reconstruct 2D x, y coordinates by integrating curvature."""
        n = len(self.s)
        x = np.zeros(n)
        y = np.zeros(n)
        theta = 0.0  # Heading angle

        for i in range(1, n):
            ds = self.s[i] - self.s[i - 1]
            k_avg = 0.5 * (self.kappa[i] + self.kappa[i - 1])
            theta += k_avg * ds
            x[i] = x[i - 1] + ds * np.cos(theta)
            y[i] = y[i - 1] + ds * np.sin(theta)

        self.x = x
        self.y = y

    def resample(self, num_nodes: int) -> "CanonicalTrack":
        """Resample track uniformly along arc-length s."""
        new_s = np.linspace(self.s[0], self.s[-1], num_nodes)
        new_kappa = np.interp(new_s, self.s, self.kappa)
        new_w_left = np.interp(new_s, self.s, self.width_left)
        new_w_right = np.interp(new_s, self.s, self.width_right)
        new_elev = np.interp(new_s, self.s, self.elevation)
        new_bank = np.interp(new_s, self.s, self.banking)
        new_x = np.interp(new_s, self.s, self.x)
        new_y = np.interp(new_s, self.s, self.y)

        return CanonicalTrack(
            s=new_s,
            kappa=new_kappa,
            width_left=new_w_left,
            width_right=new_w_right,
            elevation=new_elev,
            banking=new_bank,
            x=new_x,
            y=new_y,
            is_closed=self.is_closed,
        )
