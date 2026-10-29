"""Pydantic schemas for FastAPI REST endpoints."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Payload to trigger an optimal control lap simulation via API."""
    config: Dict[str, Any] = Field(..., description="Full simulation configuration dict (matching schema)")
    run_uncertainty: bool = Field(False, description="Whether to also run Monte Carlo uncertainty analysis")
    num_uncertainty_samples: Optional[int] = Field(None, description="Override number of Monte Carlo runs")


class SimulationResponse(BaseModel):
    """Immediate response after submitting or polling a simulation."""
    job_id: str
    status: str  # "completed", "failed", "running"
    lap_time: Optional[float] = None
    solver_message: Optional[str] = None
    solver_time: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    telemetry_summary: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    """Response listing registered vehicle subsystems and importers."""
    categories: Dict[str, List[str]]


class HealthResponse(BaseModel):
    """Service health response."""
    status: str
    version: str
