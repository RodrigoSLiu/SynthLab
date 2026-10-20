import pytest
from src.domain.contract import Contract


class TestContractInit:
    """Tests for Contract constructor."""

    def test_contract_stores_variables(self):
        contract = _make_contract()
        assert contract.variables == {"x": {"type": "float"}, "y": {"type": "float"}}

    def test_contract_stores_marginals(self):
        contract = _make_contract()
        assert len(contract.marginals) == 1
        marginal = contract.marginals[0]
        assert marginal["variable"] == "x"
        assert marginal["distribution"] == "uniform"
        assert marginal["params"] == {"low": 0.0, "high": 1.0}

    def test_contract_stores_relationships(self):
        contract = _make_contract()
        assert len(contract.relationships) == 1
        rel = contract.relationships[0]
        assert rel["type"] == "linear"
        assert rel["independent"] == "x"
        assert rel["dependent"] == "y"
        assert rel["slope"] == 2.0

    def test_contract_stores_noise_in_relationship(self):
        contract = _make_contract()
        noise = contract.relationships[0]["noise"]
        assert noise["distribution"] == "normal"
        assert noise["params"]["mean"] == 0.0
        assert noise["params"]["std"] == 1.0

    def test_contract_stores_validation(self):
        contract = _make_contract()
        dist_val = contract.validation["distribution_test"]
        assert dist_val["test"] == "ks"
        assert dist_val["alpha"] == 0.05

    def test_contract_stores_relationship_tolerance(self):
        contract = _make_contract()
        assert contract.validation["relationship_tolerance"]["slope"] == 0.2


class TestContractEdgeCases:
    """Edge case tests for Contract."""

    def test_contract_with_empty_marginals(self):
        contract = Contract(
            variables={"x": {"type": "float"}},
            marginals=[],
            relationships=[],
            validation={}
        )
        assert contract.marginals == []

    def test_contract_with_empty_relationships(self):
        contract = Contract(
            variables={"x": {"type": "float"}},
            marginals=[{"variable": "x", "distribution": "uniform", "params": {}}],
            relationships=[],
            validation={}
        )
        assert contract.relationships == []

    def test_contract_preserves_extra_variable_metadata(self):
        variables = {
            "x": {"type": "float", "description": "input variable"},
            "y": {"type": "float", "unit": "meters"}
        }
        contract = Contract(
            variables=variables,
            marginals=[],
            relationships=[],
            validation={}
        )
        assert contract.variables["x"]["description"] == "input variable"
        assert contract.variables["y"]["unit"] == "meters"


def _make_contract():
    """Helper to create a standard test contract."""
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


