"""Abstract base class for battery models."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import casadi as ca


class BatteryModelBase(ABC):
    """Base interface for battery pack models."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def step(
        self, P_elec: ca.SX, soc: ca.SX, temp: ca.SX, dt: ca.SX
    ) -> Tuple[ca.SX, ca.SX, ca.SX]:
        """Compute battery state derivatives or updates.

        Args:
            P_elec: Electrical power demand [W] (positive discharge, negative charging/regen).
            soc: State of Charge [0.0 - 1.0].
            temp: Battery cell temperature [deg C or K].
            dt: Time step [s].

        Returns:
            Tuple of (d_soc_dt, d_temp_dt, terminal_voltage [V]).
        """
        ...
