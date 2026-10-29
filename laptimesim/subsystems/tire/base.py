"""Abstract base class for tire models producing CasADi symbolic expressions."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import casadi as ca


class TireModelBase(ABC):
    """Base interface for all tire models.

    All methods return symbolic CasADi expressions (SX or MX) to guarantee
    end-to-end differentiability in optimal control formulations.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def lateral_force(self, alpha: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0) -> ca.SX:
        """Calculate pure lateral force Fy [N].

        Args:
            alpha: Tire slip angle [rad].
            Fz: Normal vertical load [N] (positive in compression).
            gamma: Camber angle [rad].

        Returns:
            Lateral force Fy [N].
        """
        ...

    @abstractmethod
    def longitudinal_force(self, kappa: ca.SX, Fz: ca.SX) -> ca.SX:
        """Calculate pure longitudinal force Fx [N].

        Args:
            kappa: Longitudinal slip ratio [-] (e.g. (omega*r - v)/v).
            Fz: Normal vertical load [N] (positive in compression).

        Returns:
            Longitudinal force Fx [N].
        """
        ...

    @abstractmethod
    def combined_forces(
        self, alpha: ca.SX, kappa: ca.SX, Fz: ca.SX, gamma: ca.SX = 0.0
    ) -> Tuple[ca.SX, ca.SX]:
        """Calculate combined slip forces (Fx, Fy) with friction ellipse interaction.

        Args:
            alpha: Tire slip angle [rad].
            kappa: Longitudinal slip ratio [-].
            Fz: Normal vertical load [N].
            gamma: Camber angle [rad].

        Returns:
            Tuple of (Fx, Fy) [N].
        """
        ...
