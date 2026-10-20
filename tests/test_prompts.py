import pytest
import json

from src.llm.prompts import create_spec_prompt


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


