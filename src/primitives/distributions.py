from src.primitives.base import DistributionPrimitive

import numpy as np
from numpy.random import Generator
from typing import Dict, Any

# -------------------------
# Concrete Distributions
# -------------------------

class Uniform(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.uniform(
            low=params["low"],
            high=params["high"],
            size=size,
        )


class Normal(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.normal(
            loc=params["mean"],
            scale=params["std"],
            size=size,
        )


class Exponential(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.exponential(
            scale=params["scale"],
            size=size,
        )


class Pareto(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.pareto(a=params["alpha"], size=size)
    

class Beta(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.beta(a=params["alpha"], b=params["beta"], size=size)


class LogNormal(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.lognormal(mean=params["mean"], sigma=params["sigma"], size=size)


class Poisson(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:
        return rng.poisson(lam=params["lambda"], size=size)


class Binomial(DistributionPrimitive):
    def generate(
        self,
        rng: Generator,
        size: int,
        params: Dict[str, float],
    ) -> np.ndarray:

        n: int = int(params["n"])
        p: float = float(params["p"])

        if n <= 0:
            raise ValueError("n must be positive")

        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")

        return rng.binomial(n=n, p=p, size=size)
