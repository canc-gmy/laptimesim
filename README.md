# laptimesim

High-fidelity, modular lap time simulator built in Python for Formula Student (electric) and vehicle dynamics research.

## Features
- **18+ DOF Multi-Body Dynamics**: High-fidelity chassis, suspension, tire, and powertrain dynamics.
- **Optimal Control Problem (OCP)**: Minimum lap time calculation via CasADi and IPOPT direct collocation in curvilinear coordinates.
- **Modular Subsystems**: Abstract Base Classes & Registry for tires, aerodynamics, suspension, and powertrains.
- **Uncertainty Quantification**: Monte Carlo parameter sensitivity and lap time confidence intervals.
- **Flexible Interfaces**: CLI, Python Library, and FastAPI REST server.

## Installation
```bash
poetry install
```

## Quick Start
```bash
# Run via CLI
poetry run laptimesim run examples/simple_run.toml
```
