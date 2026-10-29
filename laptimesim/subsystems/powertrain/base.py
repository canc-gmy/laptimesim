"""Abstract base class for powertrain models."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import casadi as ca


class PowertrainModelBase(ABC):
    """Base interface for powertrain (motor/engine + transmission + differential)."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def wheel_torques(
        self,
        throttle: ca.SX,
        regen_cmd: ca.SX,
        omega_fl: ca.SX,
        omega_fr: ca.SX,
        omega_rl: ca.SX,
        omega_rr: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        """Calculate drive torques at all 4 wheels and net electrical power demand.

        Args:
            throttle: Accelerator pedal command [0.0 - 1.0].
            regen_cmd: Regenerative braking command [0.0 - 1.0].
            omega_*: Rotational speed of each wheel [rad/s].

        Returns:
            Tuple of:
                - T_wheel_fl, T_wheel_fr, T_wheel_rl, T_wheel_rr: Drive/regen torque per wheel [Nm].
                - P_elec: Total battery electrical power demand [W].
        """
        ...
