from typing import Any, Dict
import yaml
import jsonschema

from src.domain.contract import Contract


class ValidationError(Exception):
    """Raised when contract validation fails."""

    def __init__(self, error_type: str, details: Dict[str, Any]):
        self.error_type = error_type
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return f"{self.error_type}: {self.details}"

    def create_error_json(error_type: str, **kwargs) -> Dict[str, Any]:
        return {
            "error_type": error_type,
            "details": kwargs,
        }

class Validator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate_yaml(self, yaml_text: str) -> Dict[str, Any]:
        """
        Parse and structurally validate YAML against JSON Schema.
        Returns parsed spec dict if valid.
        """
            # 1. Parse YAML
        try:
            spec = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ValidationError(
                "YAML_PARSE_ERROR",
                ValidationError.create_error_json(
                    "YAML_PARSE_ERROR",
                    message=str(e),
                ),
            )

        # 2. Validate schema
        try:
            jsonschema.validate(instance=spec, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ValidationError(
                "SCHEMA_VALIDATION_ERROR",
                ValidationError.create_error_json(
                    "SCHEMA_VALIDATION_ERROR",
                    message=e.message,
                    path=list(e.path),
                    schema_path=list(e.schema_path),
                ),
            )

        return spec
    
    def validate_semantics(self, spec: Dict[str, Any]) -> None:
        """
        Lightweight semantic checks that schema cannot express.
        """
        variables = set(spec["variables"].keys())

        # Marginal variable exists
        for marginal in spec["assumptions"]["marginals"]:
            if marginal["variable"] not in variables:
                raise ValidationError(
                    f"Marginal variable '{marginal['variable']}' not defined"
                )

        # Relationship variables exist
        for rel in spec["assumptions"]["relationships"]:
            if rel["independent"] not in variables:
                raise ValidationError(
                    f"Independent variable '{rel['independent']}' not defined"
                )
            if rel["dependent"] not in variables:
                raise ValidationError(
                    f"Dependent variable '{rel['dependent']}' not defined"
                )

    def validate_contract(self, yaml_text: str) -> Contract:
        """
        Full validation pipeline: YAML → schema → semantics → Contract.
        """
        spec = self.validate_yaml(yaml_text)
        self.validate_semantics(spec)

        return Contract.from_spec(spec)