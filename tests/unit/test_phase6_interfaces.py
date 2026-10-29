"""Unit tests for Phase 6 Interfaces: Python Library imports, CLI commands, and FastAPI endpoints."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

import laptimesim
from laptimesim.api.server import app
from laptimesim.core.config import FullSimulationConfig

client = TestClient(app)


def test_python_library_top_level_exports():
    """Verify that all core classes are cleanly accessible from root namespace."""
    assert hasattr(laptimesim, "FullSimulationConfig")
    assert hasattr(laptimesim, "LaptimeSimulator")
    assert hasattr(laptimesim, "CanonicalTrack")
    assert hasattr(laptimesim, "VehicleModel")
    assert hasattr(laptimesim, "CurvilinearLaptimeSolver")
    assert hasattr(laptimesim, "MonteCarloEngine")


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == laptimesim.__version__


def test_api_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "tire" in data["categories"]
    assert "powertrain" in data["categories"]


def test_api_schema_endpoint():
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    schema = response.json()
    assert "properties" in schema
    assert "vehicle" in schema["properties"]


def test_api_simulate_endpoint():
    config_path = Path(__file__).resolve().parent.parent.parent / "examples" / "simple_run.toml"
    cfg = FullSimulationConfig.from_toml(config_path)

    cfg_dict = cfg.model_dump()
    cfg_dict["simulation"]["num_nodes"] = 12
    cfg_dict["simulation"]["solver_max_iter"] = 50

    payload = {
        "config": cfg_dict,
        "run_uncertainty": False,
    }

    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "job_id" in res_data
    assert res_data["lap_time"] is not None
    assert res_data["lap_time"] > 0.0

    # Poll status
    job_id = res_data["job_id"]
    poll_resp = client.get(f"/api/v1/simulate/{job_id}")
    assert poll_resp.status_code == 200
    assert poll_resp.json()["job_id"] == job_id
