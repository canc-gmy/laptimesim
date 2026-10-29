"""Configuration schemas and loaders for laptimesim using Pydantic."""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

# In Python 3.11+, tomllib is built-in
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


class InertiaConfig(BaseModel):
    Ixx: float = Field(..., gt=0, description="Roll moment of inertia [kg*m^2]")
    Iyy: float = Field(..., gt=0, description="Pitch moment of inertia [kg*m^2]")
    Izz: float = Field(..., gt=0, description="Yaw moment of inertia [kg*m^2]")
    Ixz: float = Field(0.0, description="Cross product of inertia [kg*m^2]")


class VehicleConfig(BaseModel):
    mass: float = Field(..., gt=0, description="Chassis mass without driver [kg]")
    driver_mass: float = Field(75.0, ge=0, description="Driver mass [kg]")
    wheelbase: float = Field(..., gt=0, description="Wheelbase [m]")
    track_front: float = Field(..., gt=0, description="Front track width [m]")
    track_rear: float = Field(..., gt=0, description="Rear track width [m]")
    cg_height: float = Field(..., gt=0, description="Center of gravity height [m]")
    cg_longitudinal: float = Field(
        0.5, ge=0.0, le=1.0, description="CG distance from front axle fraction"
    )
    inertia: InertiaConfig


class SubsystemConfig(BaseModel):
    model: str = Field(..., description="Registered model name to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")


class SimulationSettingsConfig(BaseModel):
    name: str = Field("Lap Simulation", description="Simulation run name")
    track_file: str = Field(..., description="Path to track file")
    track_format: Literal["csv", "curvature", "opendrive"] = Field(
        "csv", description="Format of the track file"
    )
    num_nodes: int = Field(150, gt=10, description="Number of collocation nodes per lap")
    solver_max_iter: int = Field(3000, gt=0, description="Max iterations for IPOPT")
    solver_tolerance: float = Field(1e-6, gt=0, description="Convergence tolerance")


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    solver_iterations: bool = False
    log_file: Optional[str] = None


class OutputConfig(BaseModel):
    telemetry: bool = True
    gg_diagram: bool = True
    tire_utilization: bool = True
    energy_profile: bool = True
    sensitivity_report: bool = True
    export_format: Literal["csv", "parquet"] = "csv"
    animation: bool = False
    output_dir: str = "./results"


class FullSimulationConfig(BaseModel):
    """Complete root configuration for a simulation run."""
    simulation: SimulationSettingsConfig
    vehicle: VehicleConfig
    tire: Dict[str, Any]
    aero: Dict[str, Any]
    suspension: Dict[str, Any]
    powertrain: Dict[str, Any]
    battery: Optional[Dict[str, Any]] = None
    brakes: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_toml(cls, file_path: str | Path) -> "FullSimulationConfig":
        """Load and validate configuration from a TOML file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str_or_path: str | Path) -> "FullSimulationConfig":
        """Load and validate configuration from JSON string or JSON file."""
        path = Path(json_str_or_path)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(json_str_or_path))
        return cls.model_validate(data)
