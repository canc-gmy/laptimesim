"""Abstract base class for aerodynamic models producing CasADi symbolic expressions."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import casadi as ca


class AeroModelBase(ABC):
    """Base interface for all aerodynamics models."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def forces_and_moments(
        self,
        vx: ca.SX,
        vy: ca.SX,
        vz: ca.SX,
        ride_height_f: ca.SX,
        ride_height_r: ca.SX,
        pitch: ca.SX,
        roll: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
        """Compute aerodynamic forces and moments.

        Args:
            vx, vy, vz: Vehicle body velocities [m/s].
            ride_height_f, ride_height_r: Front and rear ride heights [m].
            pitch, roll: Chassis pitch and roll angles [rad].

        Returns:
            Tuple of:
                - F_drag: Aerodynamic drag force along body x [N] (opposing motion).
                - F_downforce_front: Downforce acting on front axle [N] (positive down).
                - F_downforce_rear: Downforce acting on rear axle [N] (positive down).
                - F_side: Aerodynamic side force along body y [N].
        """
        ...
