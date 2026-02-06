import pytest
import json

from src.llm.prompts import create_spec_prompt, create_code_prompt
from src.domain.contract import Contract


class TestCreateSpecPrompt:
    """Tests for create_spec_prompt function."""

    def test_returns_string(self):
        schema = {"type": "object"}
        intent = "Generate linear regression data"

        result = create_spec_prompt(schema, intent)

        assert isinstance(result, str)

    def test_includes_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        intent = "Generate data"

        result = create_spec_prompt(schema, intent)

        assert '"type": "object"' in result
        assert '"properties"' in result

    def test_includes_intent(self):
        schema = {"type": "object"}
        intent = "Generate x uniformly distributed"

        result = create_spec_prompt(schema, intent)

        assert intent in result

    def test_includes_yaml_instruction(self):
        schema = {"type": "object"}
        intent = "test"

        result = create_spec_prompt(schema, intent)

        assert "YAML" in result

    def test_includes_no_comments_instruction(self):
        schema = {"type": "object"}
        intent = "test"

        result = create_spec_prompt(schema, intent)

        assert "comment" in result.lower()

    def test_includes_float_type_instruction(self):
        schema = {"type": "object"}
        intent = "test"

        result = create_spec_prompt(schema, intent)

        assert "float" in result.lower()

    def test_handles_complex_schema(self):
        schema = {
            "type": "object",
            "required": ["variables", "assumptions"],
            "properties": {
                "variables": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "object"}
                    }
                }
            }
        }
        intent = "Generate data with x and y"

        result = create_spec_prompt(schema, intent)

        assert "variables" in result
        assert "assumptions" in result

    def test_schema_is_json_formatted(self):
        schema = {"key": "value", "nested": {"inner": 123}}
        intent = "test"

        result = create_spec_prompt(schema, intent)

        schema_text = json.dumps(schema, indent=2)
        assert schema_text in result

    def test_empty_intent(self):
        schema = {"type": "object"}
        intent = ""

        result = create_spec_prompt(schema, intent)

        assert isinstance(result, str)


class TestCreateCodePrompt:
    """Tests for create_code_prompt function."""

    def test_returns_string(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert isinstance(result, str)

    def test_includes_function_name(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "relationship_generation" in result

    def test_includes_num_samples_parameter(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "num_samples" in result

    def test_includes_seed_parameter(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "seed" in result

    def test_includes_numpy_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "numpy" in result.lower()

    def test_includes_rng_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "default_rng" in result

    def test_includes_variables(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert str(contract.variables) in result

    def test_includes_marginals(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert str(contract.marginals) in result

    def test_includes_relationships(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert str(contract.relationships) in result

    def test_includes_no_io_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "I/O" in result or "file" in result.lower()

    def test_includes_vectorized_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "vectorized" in result.lower()

    def test_includes_deterministic_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "deterministic" in result.lower()

    def test_includes_no_markdown_instruction(self):
        contract = _make_contract()

        result = create_code_prompt(contract)

        assert "markdown" in result.lower()

    def test_with_multiple_variables(self):
        contract = Contract(
            variables={
                "x": {"type": "float"},
                "y": {"type": "float"},
                "z": {"type": "float"}
            },
            marginals=[{"variable": "x", "distribution": "uniform", "params": {}}],
            relationships=[],
            validation={}
        )

        result = create_code_prompt(contract)

        assert "'x'" in result
        assert "'y'" in result
        assert "'z'" in result


def _make_contract():
    """Helper to create a standard test contract."""
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
