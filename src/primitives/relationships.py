from src.primitives.base import RelationshipPrimitive

import numpy as np
from typing import List, Dict, Any

# -------------------------
# Concrete Relationsiphs
# -------------------------

class Linear(RelationshipPrimitive):
    def generate(
        self,
        inputs,
        params
    ) -> np.ndarray:

        coefficients = params["coefficients"]
        intercept = params.get("intercept", 0.0)

        y = np.zeros_like(next(iter(inputs.values())))

        for name, array in inputs.items():
            coef = coefficients[name]
            y += coef * array

        y += intercept
        
        return y


class Polynomial(RelationshipPrimitive):
    def generate(
        self,
        inputs: Dict[str, np.ndarray],
        params: Dict[str, Any]
    ) -> np.ndarray:
        x = next(iter(inputs.values()))
        coeffs = params["coefficients"]

        y = np.zeros_like(x, dtype=float)

        for power, coef in enumerate(coeffs):
            y += coef * np.power(x, power)

        return y


class Logistic(RelationshipPrimitive):
    def generate(
        self,
        inputs: Dict[str, np.ndarray],
        params: Dict[str, float]
    ) -> np.ndarray:
        x = list(inputs.values())[0]
        slope = params["slope"]
        intercept = params.get("intercept", 0.0)

        z = slope * x + intercept

        return 1 / (1 + np.exp(-z))
    

class Interaction(RelationshipPrimitive):
    def generate(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        arrays = list(inputs.values())
        result = arrays[0]

        for arr in arrays[1:]:
            result = result * arr

        return result

class PiecewiseLinear(RelationshipPrimitive):
    def generate(
        self,
        inputs: Dict[str, np.ndarray],
        params: Dict[str, Any],
    ) -> np.ndarray:

        # Expect exactly one input variable for now
        if len(inputs) != 1:
            raise ValueError("PiecewiseLinear currently supports exactly one input variable")

        x = next(iter(inputs.values()))

        breakpoints: List[float] = params["breakpoints"]
        slopes: List[float] = params["slopes"]
        intercepts: List[float] = params["intercepts"]

        if len(slopes) != len(intercepts):
            raise ValueError("slopes and intercepts must have same length")

        if len(slopes) != len(breakpoints) + 1:
            raise ValueError("Number of segments must equal len(breakpoints) + 1")

        y = np.zeros_like(x)

        segments = [-np.inf] + breakpoints + [np.inf]

        for i in range(len(slopes)):
            mask = (x >= segments[i]) & (x < segments[i + 1])
            y[mask] = slopes[i] * x[mask] + intercepts[i]

        return y