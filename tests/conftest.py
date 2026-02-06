import pytest
import sys
from pathlib import Path

from src.domain.contract import Contract
from src.utils.utils import load_schema


@pytest.fixture
def sample_contract():
    """Create a standard Contract instance for testing."""
    return Contract(
        variables={"x": {"type": "float"}, "y": {"type": "float"}},
        marginals=[{
            "variable": "x",
            "distribution": "uniform",
            "params": {"min": 0.0, "max": 1.0}
        }],
        relationships=[{
            "type": "linear",
            "independent": "x",
            "dependent": "y",
            "slope": 2.0,
            "noise": {
                "distribution": "normal",
                "params": {"mean": 0.0, "std": 1.0}
            }
        }],
        validation={
            "distribution_test": {"test": "ks", "alpha": 0.05},
            "relationship_tolerance": {"slope": 0.2}
        }
    )


@pytest.fixture
def sample_spec():
    """Create a standard spec dictionary for testing."""
    return {
        "variables": {"x": {"type": "float"}, "y": {"type": "float"}},
        "assumptions": {
            "marginals": [{
                "variable": "x",
                "distribution": "uniform",
                "params": {"min": 0.0, "max": 1.0}
            }],
            "relationships": [{
                "type": "linear",
                "independent": "x",
                "dependent": "y",
                "slope": 2.0,
                "noise": {
                    "distribution": "normal",
                    "params": {"mean": 0.0, "std": 1.0}
                }
            }]
        },
        "validation": {
            "distribution_test": {"test": "ks", "alpha": 0.05},
            "relationship_tolerance": {"slope": 0.2}
        }
    }


@pytest.fixture
def sample_yaml():
    """Return valid YAML text conforming to the linear regression schema."""
    return """
variables:
  x:
    type: float
  y:
    type: float
assumptions:
  marginals:
    - variable: x
      distribution: uniform
      params:
        min: 0.0
        max: 1.0
  relationships:
    - type: linear
      independent: x
      dependent: y
      slope: 2.0
      noise:
        distribution: normal
        params:
          mean: 0.0
          std: 1.0
validation:
  distribution_test:
    test: ks
    alpha: 0.05
  relationship_tolerance:
    slope: 0.2
"""


@pytest.fixture
def linear_regression_schema():
    """Load the linear regression JSON schema."""
    return load_schema("linear_regression.schema.json")


@pytest.fixture
def sample_generated_code():
    """Return sample generated Python code."""
    return """def relationship_generation(num_samples, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, num_samples)
    y = 2.0 * x + rng.normal(0, 1, num_samples)
    return x, y
"""
