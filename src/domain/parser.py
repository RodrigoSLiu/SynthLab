from typing import Any, Dict
import logging
import yaml
import jsonschema

from src.domain.contract import Contract
from src.errors import ParserError

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def parse_yaml(self, yaml_text: str) -> Dict[str, Any]:
        """
        Parse YAML and check it against the JSON Schema.
        Returns parsed spec dict if valid.
        """
        logger.info("Parsing spec...")

        try:
            spec = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ParserError(
                "YAML_PARSE_ERROR",
                {
                    "message": str(e),
                    "received": yaml_text,
                },
            )

        # Check against schema
        try:
            jsonschema.validate(instance=spec, schema=self.schema)
        except jsonschema.ValidationError as e:
            raise ParserError(
                "SCHEMA_VALIDATION_ERROR",
                {
                    "message": str(e),
                    "path": list(e.path),
                    "schema_path": list(e.schema_path),
                },
            )

        logger.info("Spec is valid")

        return spec

    def parse_semantics(self, spec: Dict[str, Any]) -> None:
        """
        Lightweight semantic checks that schema cannot express.
        """
        variables = set(spec["variables"].keys())

        # Marginal variable exists
        for marginal in spec["assumptions"]["marginals"]:
            if marginal["variable"] not in variables:
                raise ParserError(
                    "SCHEMA_VALIDATION_ERROR",
                    {
                        "message": f"Marginal variable '{marginal['variable']}' not defined",
                    },
                )

        # Relationship variables exist
        for rel in spec["assumptions"]["relationships"]:
            if rel["independent"] not in variables:
                raise ParserError(
                    "SCHEMA_VALIDATION_ERROR",
                    {
                        "message": f"Independent variable '{rel['independent']}' not defined",
                    },
                )
            if rel["dependent"] not in variables:
                raise ParserError(
                    "SCHEMA_VALIDATION_ERROR",
                    {
                        "message": f"Dependent variable '{rel['dependent']}' not defined",
                    },
                )

    def parse_contract(self, yaml_text: str) -> Contract:
        """
        Full parsing pipeline: YAML → schema → semantics → Contract.
        """
        spec = self.parse_yaml(yaml_text)
        self.parse_semantics(spec)

        return Contract.from_spec(spec)
