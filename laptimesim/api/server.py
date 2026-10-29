"""FastAPI application for laptimesim REST API and WebSockets."""

import uuid
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from laptimesim import __version__
from laptimesim.core.config import FullSimulationConfig
from laptimesim.core.registry import model_registry
from laptimesim.core.simulator import LaptimeSimulator
from laptimesim.api.schemas import (
    SimulationRequest,
    SimulationResponse,
    ModelListResponse,
    HealthResponse,
)

app = FastAPI(
    title="laptimesim API",
    description="REST API for High-Fidelity Vehicle Dynamics & Lap Time Simulation",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demonstration / single-node deployment
job_storage: Dict[str, SimulationResponse] = {}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/api/v1/models", response_model=ModelListResponse, tags=["Subsystems"])
def list_available_models():
    """List all registered vehicle dynamics subsystems and track importers."""
    return ModelListResponse(categories=model_registry.list_all())


@app.get("/api/v1/schema", tags=["Configuration"])
def get_config_schema():
    """Retrieve full JSON Schema for simulation configuration."""
    return FullSimulationConfig.model_json_schema()


@app.post("/api/v1/simulate", response_model=SimulationResponse, tags=["Simulation"])
def run_simulation(request: SimulationRequest):
    """Run an optimal control lap simulation given a JSON configuration payload."""
    job_id = str(uuid.uuid4())

    try:
        cfg = FullSimulationConfig.model_validate(request.config)
        if request.run_uncertainty:
            if not cfg.uncertainty:
                cfg.uncertainty = {"enabled": True}
            else:
                cfg.uncertainty["enabled"] = True
            if request.num_uncertainty_samples:
                cfg.uncertainty["num_samples"] = request.num_uncertainty_samples

        sim = LaptimeSimulator(cfg)
        output = sim.run()

        ci_data = None
        if output.uncertainty:
            ci_data = {
                "mean": output.uncertainty.mean_lap_time,
                "std": output.uncertainty.std_lap_time,
                "ci_lower": output.uncertainty.ci_lower,
                "ci_upper": output.uncertainty.ci_upper,
                "confidence_level": output.uncertainty.confidence_level,
            }

        telem_summary = {
            "max_speed_km_h": float(output.telemetry_df["speed_km_h"].max()),
            "avg_speed_km_h": float(output.telemetry_df["speed_km_h"].mean()),
            "max_ay_g": float(output.telemetry_df.get("ay_g", 0.0).max()),
            "min_ay_g": float(output.telemetry_df.get("ay_g", 0.0).min()),
            "num_data_points": len(output.telemetry_df),
        }

        resp = SimulationResponse(
            job_id=job_id,
            status="completed" if output.result.success else "failed",
            lap_time=output.result.lap_time,
            solver_message=output.result.message,
            solver_time=output.result.solver_time,
            confidence_interval=ci_data,
            telemetry_summary=telem_summary,
        )
        job_storage[job_id] = resp
        return resp

    except Exception as e:
        failed_resp = SimulationResponse(
            job_id=job_id,
            status="failed",
            solver_message=str(e),
        )
        job_storage[job_id] = failed_resp
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/simulate/{job_id}", response_model=SimulationResponse, tags=["Simulation"])
def get_simulation_status(job_id: str):
    """Retrieve results for a previous simulation by job ID."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail=f"Simulation job '{job_id}' not found.")
    return job_storage[job_id]
