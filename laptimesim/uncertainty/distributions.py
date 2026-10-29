"""Probability distributions and Latin Hypercube Sampler for vehicle parameters."""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class ParameterDistribution:
    """Represents uncertainty distribution for a parameter."""
    param_path: str  # e.g. "vehicle.mass" or "aero.Cl" or "tire.front.D"
    distribution: str  # "normal" or "uniform"
    mean: float
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0

    def sample(self, u: float) -> float:
        """Inverse transform sampling using uniform random variable u in (0, 1)."""
        if self.distribution == "uniform":
            return self.min_val + u * (self.max_val - self.min_val)
        elif self.distribution == "normal":
            from scipy.stats import norm
            return float(norm.ppf(u, loc=self.mean, scale=max(self.std, 1e-6)))
        return self.mean


class UncertaintySampler:
    """Generates Latin Hypercube samples for parametric uncertainty analysis."""

    def __init__(self, distributions: List[ParameterDistribution], num_samples: int = 50, seed: int = 42):
        self.distributions = distributions
        self.num_samples = num_samples
        self.seed = seed

    def sample_all(self) -> List[Dict[str, float]]:
        """Generate Latin Hypercube Samples across all defined distributions."""
        rng = np.random.default_rng(self.seed)
        d = len(self.distributions)
        N = self.num_samples

        if d == 0:
            return []

        # Latin Hypercube Sampling: divide [0, 1] into N intervals
        u = np.zeros((N, d))
        for j in range(d):
            perm = rng.permutation(N)
            jitter = rng.uniform(0.0, 1.0, N)
            u[:, j] = (perm + jitter) / N

        samples = []
        for i in range(N):
            sample_dict = {}
            for j, dist in enumerate(self.distributions):
                sample_dict[dist.param_path] = dist.sample(u[i, j])
            samples.append(sample_dict)

        return samples
