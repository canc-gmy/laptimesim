"""laptimesim - High-fidelity modular lap time simulator."""

from laptimesim.core.config import FullSimulationConfig
from laptimesim.core.simulator import LaptimeSimulator, SimulationOutput
from laptimesim.core.registry import model_registry
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.ocp import CurvilinearLaptimeSolver, OCPResult
from laptimesim.uncertainty.monte_carlo import MonteCarloEngine, UncertaintyResult

__version__ = "0.1.0"

__all__ = [
    "FullSimulationConfig",
    "LaptimeSimulator",
    "SimulationOutput",
    "model_registry",
    "CanonicalTrack",
    "VehicleModel",
    "CurvilinearLaptimeSolver",
    "OCPResult",
    "MonteCarloEngine",
    "UncertaintyResult",
]
