"""Monte Carlo sensitivity and uncertainty engine for lap time simulation."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from laptimesim.core.config import FullSimulationConfig
from laptimesim.track.canonical import CanonicalTrack
from laptimesim.vehicle.vehicle_model import VehicleModel
from laptimesim.solver.ocp import CurvilinearLaptimeSolver
from laptimesim.uncertainty.distributions import ParameterDistribution, UncertaintySampler


def _solve_single_mc_sample(args: Tuple[Dict[str, Any], Dict[str, float], CanonicalTrack]) -> Tuple[Dict[str, float], float]:
    """Top-level worker function for parallel process execution."""
    base_dict, s_dict, track = args
    modified_dict = deepcopy(base_dict)

    # Set parameters
    for path, val in s_dict.items():
        tokens = path.split(".")
        target = modified_dict
        for t in tokens[:-1]:
            if t not in target:
                target[t] = {}
            target = target[t]
        target[tokens[-1]] = val

    cfg = FullSimulationConfig.model_validate(modified_dict)
    # Turn off solver iterations printing in worker processes
    cfg.logging.solver_iterations = False
    vehicle = VehicleModel(cfg)
    solver = CurvilinearLaptimeSolver(vehicle, track, cfg)
    res = solver.solve()

    return s_dict, float(res.lap_time)


@dataclass
class UncertaintyResult:
    """Statistical summary of Monte Carlo lap time simulation."""
    mean_lap_time: float
    std_lap_time: float
    min_lap_time: float
    max_lap_time: float
    ci_lower: float  # e.g. 2.5th percentile
    ci_upper: float  # e.g. 97.5th percentile
    confidence_level: float
    lap_times: List[float]
    parameter_sensitivities: Dict[str, float]  # Pearson correlation with lap time
    num_samples_successful: int
    num_samples_total: int
    elapsed_time: float


class MonteCarloEngine:
    """Executes Monte Carlo parameter variations to estimate confidence intervals and sensitivity."""

    def __init__(self, base_config: FullSimulationConfig, track: CanonicalTrack):
        self.base_config = base_config
        self.track = track

    def _extract_distributions(self) -> List[ParameterDistribution]:
        """Parse parameter distributions from the uncertainty configuration block."""
        unc_cfg = self.base_config.uncertainty or {}
        params_cfg = unc_cfg.get("parameters", {})
        distributions: List[ParameterDistribution] = []

        for path, spec in params_cfg.items():
            dist_type = spec.get("distribution", "normal")
            mean_val = float(spec.get("mean", 0.0))
            std_val = float(spec.get("std", 0.0))
            min_val = float(spec.get("min", mean_val - 3 * std_val))
            max_val = float(spec.get("max", mean_val + 3 * std_val))

            distributions.append(
                ParameterDistribution(
                    param_path=path,
                    distribution=dist_type,
                    mean=mean_val,
                    std=std_val,
                    min_val=min_val,
                    max_val=max_val,
                )
            )
        return distributions

    @staticmethod
    def _apply_parameter(config_dict: Dict[str, Any], path: str, value: float) -> None:
        """Set nested dictionary parameter by dot notation path (e.g. 'vehicle.mass')."""
        tokens = path.split(".")
        target = config_dict
        for t in tokens[:-1]:
            if t not in target:
                target[t] = {}
            target = target[t]
        target[tokens[-1]] = value

    def run(self, num_samples: Optional[int] = None, max_workers: Optional[int] = None) -> UncertaintyResult:
        """Run Monte Carlo simulation over sampled parameter variations in parallel."""
        start_time = time.time()
        unc_cfg = self.base_config.uncertainty or {}
        n_samples = num_samples or int(unc_cfg.get("num_samples", 20))
        conf_level = float(unc_cfg.get("confidence_level", 0.95))
        workers = max_workers or min(os.cpu_count() or 4, 8)

        dists = self._extract_distributions()
        if not dists:
            dists = [
                ParameterDistribution("vehicle.mass", "normal", self.base_config.vehicle.mass, 3.0),
                ParameterDistribution("tire.D", "normal", 1.5, 0.05),
            ]

        sampler = UncertaintySampler(dists, num_samples=n_samples)
        sample_list = sampler.sample_all()

        lap_times: List[float] = []
        param_records: Dict[str, List[float]] = {d.param_path: [] for d in dists}

        base_dict = self.base_config.model_dump()
        tasks = [(base_dict, s_dict, self.track) for s_dict in sample_list]

        # Execute simulations across multiple processes in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_solve_single_mc_sample, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    s_dict, lap_t = future.result()
                    if lap_t > 0.0 and not np.isnan(lap_t):
                        lap_times.append(lap_t)
                        for path, val in s_dict.items():
                            param_records[path].append(val)
                except Exception:
                    continue

        times_arr = np.array(lap_times)
        if len(times_arr) == 0:
            raise RuntimeError("All parallel Monte Carlo simulation runs failed.")

        # Compute confidence interval bounds
        alpha = (1.0 - conf_level) / 2.0
        ci_lower = float(np.percentile(times_arr, alpha * 100))
        ci_upper = float(np.percentile(times_arr, (1.0 - alpha) * 100))

        # Compute sensitivity (Pearson correlation coefficient r between each parameter and lap time)
        sensitivities = {}
        for path, values in param_records.items():
            val_arr = np.array(values)
            if np.std(val_arr) > 1e-9 and np.std(times_arr) > 1e-9:
                corr = float(np.corrcoef(val_arr, times_arr)[0, 1])
                sensitivities[path] = corr if not np.isnan(corr) else 0.0
            else:
                sensitivities[path] = 0.0

        elapsed = time.time() - start_time

        return UncertaintyResult(
            mean_lap_time=float(np.mean(times_arr)),
            std_lap_time=float(np.std(times_arr)),
            min_lap_time=float(np.min(times_arr)),
            max_lap_time=float(np.max(times_arr)),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=conf_level,
            lap_times=lap_times,
            parameter_sensitivities=sensitivities,
            num_samples_successful=len(lap_times),
            num_samples_total=n_samples,
            elapsed_time=elapsed,
        )
