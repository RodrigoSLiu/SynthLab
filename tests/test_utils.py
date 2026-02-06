import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.utils.utils import (
    write_generated_code,
    write_generated_yaml,
    load_schema,
    load_spec,
    extract_yaml
)


class TestWriteGeneratedCode:
    """Tests for write_generated_code function."""

    def test_writes_code_to_file(self, tmp_path):
        code = "def foo():\n    return 42"
        output_file = tmp_path / "output.py"

        write_generated_code(code, output_file)

        assert output_file.exists()
        assert output_file.read_text() == code

    def test_creates_parent_directories(self, tmp_path):
        code = "print('hello')"
        output_file = tmp_path / "nested" / "dir" / "output.py"

        write_generated_code(code, output_file)

        assert output_file.exists()
        assert output_file.read_text() == code

    def test_raises_error_for_empty_code(self, tmp_path):
        output_file = tmp_path / "output.py"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_code("", output_file)

    def test_raises_error_for_whitespace_only_code(self, tmp_path):
        output_file = tmp_path / "output.py"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_code("   \n\t  ", output_file)

    def test_raises_error_for_non_string_code(self, tmp_path):
        output_file = tmp_path / "output.py"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_code(None, output_file)

    def test_raises_error_for_non_py_extension(self, tmp_path):
        output_file = tmp_path / "output.txt"

        with pytest.raises(ValueError, match="must be a .py file"):
            write_generated_code("code", output_file)

    def test_accepts_string_path(self, tmp_path):
        code = "x = 1"
        output_file = str(tmp_path / "output.py")

        write_generated_code(code, output_file)

        assert Path(output_file).read_text() == code

    def test_overwrites_existing_file(self, tmp_path):
        output_file = tmp_path / "output.py"
        output_file.write_text("old content")

        write_generated_code("new content", output_file)

        assert output_file.read_text() == "new content"


class TestWriteGeneratedYaml:
    """Tests for write_generated_yaml function."""

    def test_writes_yaml_to_file(self, tmp_path):
        spec = {"key": "value", "number": 42}
        output_file = tmp_path / "output.yaml"

        write_generated_yaml(spec, output_file)

        assert output_file.exists()
        loaded = yaml.safe_load(output_file.read_text())
        assert loaded == spec

    def test_creates_parent_directories(self, tmp_path):
        spec = {"test": True}
        output_file = tmp_path / "nested" / "dir" / "output.yaml"

        write_generated_yaml(spec, output_file)

        assert output_file.exists()

    def test_raises_error_for_empty_dict(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_yaml({}, output_file)

    def test_raises_error_for_non_dict(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_yaml("not a dict", output_file)

    def test_raises_error_for_none(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            write_generated_yaml(None, output_file)

    def test_raises_error_for_non_yaml_extension(self, tmp_path):
        output_file = tmp_path / "output.json"

        with pytest.raises(ValueError, match="must be a .yaml file"):
            write_generated_yaml({"key": "value"}, output_file)

    def test_accepts_string_path(self, tmp_path):
        spec = {"data": [1, 2, 3]}
        output_file = str(tmp_path / "output.yaml")

        write_generated_yaml(spec, output_file)

        loaded = yaml.safe_load(Path(output_file).read_text())
        assert loaded == spec

    def test_preserves_key_order(self, tmp_path):
        spec = {"z": 1, "a": 2, "m": 3}
        output_file = tmp_path / "output.yaml"

        write_generated_yaml(spec, output_file)

        content = output_file.read_text()
        z_pos = content.find("z:")
        a_pos = content.find("a:")
        m_pos = content.find("m:")
        assert z_pos < a_pos < m_pos


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_loads_existing_schema(self):
        schema = load_schema("linear_regression.schema.json")

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "variables" in schema["properties"]

    def test_raises_error_for_nonexistent_schema(self):
        with pytest.raises(FileNotFoundError):
            load_schema("nonexistent.schema.json")

    def test_returns_valid_json_schema(self):
        schema = load_schema("linear_regression.schema.json")

        assert "$schema" in schema or "type" in schema
        assert schema.get("type") == "object"


class TestLoadSpec:
    """Tests for load_spec function."""

    def test_loads_existing_spec(self):
        spec = load_spec("linear_regression.yaml")

        assert isinstance(spec, dict)
        assert "variables" in spec

    def test_raises_error_for_nonexistent_spec(self):
        with pytest.raises(FileNotFoundError):
            load_spec("nonexistent.yaml")


class TestExtractYaml:
    """Tests for extract_yaml function."""

    def test_extracts_yaml_from_fenced_block(self):
        llm_output = """Here is the YAML:
```yaml
key: value
number: 42
```
Done!"""
        result = extract_yaml(llm_output)

        assert result == "key: value\nnumber: 42"

    def test_extracts_yaml_with_complex_content(self):
        llm_output = """```yaml
variables:
  x:
    type: float
  y:
    type: float
```"""
        result = extract_yaml(llm_output)
        parsed = yaml.safe_load(result)

        assert parsed["variables"]["x"]["type"] == "float"

    def test_returns_entire_output_when_no_fence(self):
        llm_output = """key: value
number: 42"""
        result = extract_yaml(llm_output)

        assert result == llm_output

    def test_strips_whitespace(self):
        llm_output = """```yaml

key: value

```"""
        result = extract_yaml(llm_output)

        assert result == "key: value"

    def test_handles_empty_fenced_block(self):
        llm_output = """```yaml
```"""
        result = extract_yaml(llm_output)

        assert result == ""

    def test_handles_multiple_fenced_blocks(self):
        llm_output = """First block:
```yaml
first: one
```
Second block:
```yaml
second: two
```"""
        result = extract_yaml(llm_output)

        assert "first: one" in result

    def test_handles_yaml_with_special_characters(self):
        llm_output = """```yaml
message: "Hello: World!"
list:
  - item1
  - item2
```"""
        result = extract_yaml(llm_output)
        parsed = yaml.safe_load(result)

        assert parsed["message"] == "Hello: World!"
        assert len(parsed["list"]) == 2

    def test_handles_whitespace_only_input(self):
        llm_output = "   \n\t   "
        result = extract_yaml(llm_output)

        assert result == ""

    def test_case_sensitive_yaml_fence(self):
        llm_output = """```YAML
key: value
```"""
        result = extract_yaml(llm_output)

        assert result == llm_output.strip()
