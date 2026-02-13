from src.validation.errors import SpecValidatorError
from src.domain.contract import Contract

from typing import Any, Dict
import logging
import yaml
import jsonschema
import pandas as pd

logger = logging.getLogger(__name__)


class SpecValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate_yaml(self, yaml_text: str) -> Dict[str, Any]:
        """
        Parse and structurally validate YAML against JSON Schema.
        Returns parsed spec dict if valid.
        """
        # Parse YAML
        logger.info("Validating Spec...")

        try:
            spec = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise SpecValidatorError(
                "YAML_PARSE_ERROR",
                SpecValidatorError.create_error_json(
                    "YAML_PARSE_ERROR",
                    message=str(e),
                    received=yaml_text,
                ),
            )

        # Validate schema
        try:
            jsonschema.validate(instance=spec, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise SpecValidatorError(
                "SCHEMA_VALIDATION_ERROR",
                SpecValidatorError.create_error_json(
                    "SCHEMA_VALIDATION_ERROR",
                    message=str(e),
                    path=list(e.path),
                    schema_path=list(e.schema_path),
                ),
            )
        
        logger.info("Spec is valid")

        return spec
    
    def validate_semantics(self, spec: Dict[str, Any]) -> None:
        """
        Lightweight semantic checks that schema cannot express.
        """
        variables = set(spec["variables"].keys())

        # Marginal variable exists
        for marginal in spec["assumptions"]["marginals"]:
            if marginal["variable"] not in variables:
                raise SpecValidatorError(
                    "SCHEMA_VALIDATION_ERROR",
                    SpecValidatorError.create_error_json(
                        "SCHEMA_VALIDATION_ERROR",
                        message=f"Marginal variable '{marginal['variable']}' not defined",
                    ),
                )

        # Relationship variables exist
        for rel in spec["assumptions"]["relationships"]:
            if rel["independent"] not in variables:
                raise SpecValidatorError(
                        "SCHEMA_VALIDATION_ERROR",
                        SpecValidatorError.create_error_json(
                            "SCHEMA_VALIDATION_ERROR",
                            message=f"Independent variable '{rel['independent']}' not defined",
                        ),
                    )
            if rel["dependent"] not in variables:
                raise SpecValidatorError(
                    "SCHEMA_VALIDATION_ERROR",
                    SpecValidatorError.create_error_json(
                        "SCHEMA_VALIDATION_ERROR",
                        message= f"Dependent variable '{rel['dependent']}' not defined"
                    ),
                )

    def validate_contract(self, yaml_text: str) -> Contract:
        """
        Full validation pipeline: YAML → schema → semantics → Contract.
        """
        spec = self.validate_yaml(yaml_text)
        self.validate_semantics(spec)

        return Contract.from_spec(spec)
