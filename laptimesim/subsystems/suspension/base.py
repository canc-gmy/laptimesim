"""Abstract base class for suspension models producing CasADi symbolic expressions."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import casadi as ca


class SuspensionModelBase(ABC):
    """Base interface for suspension kinematics and dynamics."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    @abstractmethod
    def suspension_forces(
        self,
        z_s_fl: ca.SX,
        z_s_fr: ca.SX,
        z_s_rl: ca.SX,
        z_s_rr: ca.SX,
        dz_s_fl: ca.SX,
        dz_s_fr: ca.SX,
        dz_s_rl: ca.SX,
        dz_s_rr: ca.SX,
    ) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
        """Compute vertical suspension forces transmitted to the chassis at each corner [N].

        Args:
            z_s_*: Suspension deflection (compression positive) [m].
            dz_s_*: Suspension velocity (compression positive) [m/s].

        Returns:
            Tuple of (F_susp_FL, F_susp_FR, F_susp_RL, F_susp_RR) [N].
        """
        ...
