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
            "params": {"low": 0.0, "high": 1.0}
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
                "params": {"low": 0.0, "high": 1.0}
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
        low: 0.0
        high: 1.0
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


