import pytest
import yaml

from src.validation.validator import Validator, ValidationError
from src.domain.contract import Contract
from src.utils.utils import load_schema


@pytest.fixture
def schema():
    """Load the linear regression schema for testing."""
    return load_schema("linear_regression.schema.json")


@pytest.fixture
def validator(schema):
    """Create a Validator instance with the linear regression schema."""
    return Validator(schema)


@pytest.fixture
def valid_yaml():
    """Return valid YAML text conforming to the schema."""
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


class TestValidatorInit:
    """Tests for Validator initialization."""

    def test_stores_schema(self, schema):
        validator = Validator(schema)

        assert validator.schema == schema

    def test_accepts_empty_schema(self):
        validator = Validator({})

        assert validator.schema == {}


class TestValidateYaml:
    """Tests for Validator.validate_yaml method."""

    def test_returns_parsed_spec_for_valid_yaml(self, validator, valid_yaml):
        result = validator.validate_yaml(valid_yaml)

        assert isinstance(result, dict)
        assert "variables" in result
        assert "assumptions" in result
        assert "validation" in result

    def test_raises_validation_error_for_invalid_yaml_syntax(self, validator):
        invalid_yaml = """
variables:
  x:
    type: float
  invalid yaml here: [unclosed bracket
"""
        with pytest.raises(ValidationError, match="Invalid YAML"):
            validator.validate_yaml(invalid_yaml)

    def test_raises_validation_error_for_missing_required_field(self, validator):
        missing_variables = """
assumptions:
  marginals: []
  relationships: []
validation:
  distribution_test:
    test: ks
    alpha: 0.05
  relationship_tolerance:
    slope: 0.1
"""
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validator.validate_yaml(missing_variables)

    def test_raises_validation_error_for_wrong_type(self, validator):
        wrong_type = """
variables:
  x:
    type: integer
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
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validator.validate_yaml(wrong_type)

    def test_raises_validation_error_for_extra_properties(self, validator):
        extra_props = """
variables:
  x:
    type: float
  y:
    type: float
  z:
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
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validator.validate_yaml(extra_props)

    def test_raises_validation_error_for_invalid_alpha_range(self, validator):
        invalid_alpha = """
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
    alpha: 1.5
  relationship_tolerance:
    slope: 0.2
"""
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validator.validate_yaml(invalid_alpha)

    def test_raises_validation_error_for_negative_std(self, validator):
        negative_std = """
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
          std: -1.0
validation:
  distribution_test:
    test: ks
    alpha: 0.05
  relationship_tolerance:
    slope: 0.2
"""
        with pytest.raises(ValidationError, match="Schema validation failed"):
            validator.validate_yaml(negative_std)


class TestValidateSemantics:
    """Tests for Validator.validate_semantics method."""

    def test_passes_for_valid_spec(self, validator, valid_yaml):
        spec = yaml.safe_load(valid_yaml)

        validator.validate_semantics(spec)

    def test_raises_error_for_undefined_marginal_variable(self, validator):
        spec = {
            "variables": {"x": {"type": "float"}, "y": {"type": "float"}},
            "assumptions": {
                "marginals": [{"variable": "z", "distribution": "uniform", "params": {}}],
                "relationships": []
            },
            "validation": {}
        }

        with pytest.raises(ValidationError, match="Marginal variable 'z' not defined"):
            validator.validate_semantics(spec)

    def test_raises_error_for_undefined_independent_variable(self, validator):
        spec = {
            "variables": {"x": {"type": "float"}, "y": {"type": "float"}},
            "assumptions": {
                "marginals": [],
                "relationships": [{
                    "type": "linear",
                    "independent": "z",
                    "dependent": "y",
                    "slope": 1.0
                }]
            },
            "validation": {}
        }

        with pytest.raises(ValidationError, match="Independent variable 'z' not defined"):
            validator.validate_semantics(spec)

    def test_raises_error_for_undefined_dependent_variable(self, validator):
        spec = {
            "variables": {"x": {"type": "float"}, "y": {"type": "float"}},
            "assumptions": {
                "marginals": [],
                "relationships": [{
                    "type": "linear",
                    "independent": "x",
                    "dependent": "z",
                    "slope": 1.0
                }]
            },
            "validation": {}
        }

        with pytest.raises(ValidationError, match="Dependent variable 'z' not defined"):
            validator.validate_semantics(spec)

    def test_passes_with_empty_marginals(self, validator):
        spec = {
            "variables": {"x": {"type": "float"}},
            "assumptions": {
                "marginals": [],
                "relationships": []
            },
            "validation": {}
        }

        validator.validate_semantics(spec)

    def test_passes_with_empty_relationships(self, validator):
        spec = {
            "variables": {"x": {"type": "float"}},
            "assumptions": {
                "marginals": [{"variable": "x", "distribution": "uniform", "params": {}}],
                "relationships": []
            },
            "validation": {}
        }

        validator.validate_semantics(spec)


class TestValidateContract:
    """Tests for Validator.validate_contract method."""

    def test_returns_contract_for_valid_yaml(self, validator, valid_yaml):
        result = validator.validate_contract(valid_yaml)

        assert isinstance(result, Contract)

    def test_contract_has_correct_variables(self, validator, valid_yaml):
        contract = validator.validate_contract(valid_yaml)

        assert "x" in contract.variables
        assert "y" in contract.variables

    def test_contract_has_correct_marginals(self, validator, valid_yaml):
        contract = validator.validate_contract(valid_yaml)

        assert len(contract.marginals) == 1
        assert contract.marginals[0]["variable"] == "x"
        assert contract.marginals[0]["distribution"] == "uniform"

    def test_contract_has_correct_relationships(self, validator, valid_yaml):
        contract = validator.validate_contract(valid_yaml)

        assert len(contract.relationships) == 1
        assert contract.relationships[0]["type"] == "linear"
        assert contract.relationships[0]["slope"] == 2.0

    def test_contract_has_correct_validation(self, validator, valid_yaml):
        contract = validator.validate_contract(valid_yaml)

        assert contract.validation["distribution_test"]["test"] == "ks"
        assert contract.validation["distribution_test"]["alpha"] == 0.05

    def test_raises_error_for_invalid_yaml(self, validator):
        invalid_yaml = "not: valid: yaml: here:"

        with pytest.raises(ValidationError):
            validator.validate_contract(invalid_yaml)

    def test_raises_error_for_semantic_violation(self, schema):
        validator = Validator({})
        invalid_spec_yaml = """
variables:
  a:
    type: float
assumptions:
  marginals:
    - variable: undefined_var
      distribution: uniform
      params: {}
  relationships: []
validation: {}
"""
        with pytest.raises(ValidationError, match="not defined"):
            validator.validate_contract(invalid_spec_yaml)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_is_exception(self):
        assert issubclass(ValidationError, Exception)

    def test_stores_message(self):
        error = ValidationError("test message")

        assert str(error) == "test message"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("custom error")

        assert "custom error" in str(exc_info.value)
