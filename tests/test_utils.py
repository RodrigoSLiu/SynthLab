import pytest
import yaml
from pathlib import Path

from src.utils.utils import Utils


class TestWriteGeneratedYaml:
    """Tests for write_generated_yaml function."""

    def test_writes_yaml_to_file(self, tmp_path):
        spec = {"key": "value", "number": 42}
        output_file = tmp_path / "output.yaml"

        Utils.write_generated_yaml(spec, output_file)

        assert output_file.exists()
        loaded = yaml.safe_load(output_file.read_text())
        assert loaded == spec

    def test_creates_parent_directories(self, tmp_path):
        spec = {"test": True}
        output_file = tmp_path / "nested" / "dir" / "output.yaml"

        Utils.write_generated_yaml(spec, output_file)

        assert output_file.exists()

    def test_raises_error_for_empty_dict(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            Utils.write_generated_yaml({}, output_file)

    def test_raises_error_for_non_dict(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            Utils.write_generated_yaml("not a dict", output_file)

    def test_raises_error_for_none(self, tmp_path):
        output_file = tmp_path / "output.yaml"

        with pytest.raises(ValueError, match="empty or invalid"):
            Utils.write_generated_yaml(None, output_file)

    def test_raises_error_for_non_yaml_extension(self, tmp_path):
        output_file = tmp_path / "output.json"

        with pytest.raises(ValueError, match="must be a .yaml file"):
            Utils.write_generated_yaml({"key": "value"}, output_file)

    def test_accepts_string_path(self, tmp_path):
        spec = {"data": [1, 2, 3]}
        output_file = str(tmp_path / "output.yaml")

        Utils.write_generated_yaml(spec, output_file)

        loaded = yaml.safe_load(Path(output_file).read_text())
        assert loaded == spec

    def test_preserves_key_order(self, tmp_path):
        spec = {"z": 1, "a": 2, "m": 3}
        output_file = tmp_path / "output.yaml"

        Utils.write_generated_yaml(spec, output_file)

        content = output_file.read_text()
        z_pos = content.find("z:")
        a_pos = content.find("a:")
        m_pos = content.find("m:")
        assert z_pos < a_pos < m_pos


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_loads_existing_schema(self):
        schema = Utils.load_schema("specification.schema.json")

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "variables" in schema["properties"]

    def test_raises_error_for_nonexistent_schema(self):
        with pytest.raises(FileNotFoundError):
            Utils.load_schema("nonexistent.schema.json")

    def test_returns_valid_json_schema(self):
        schema = Utils.load_schema("specification.schema.json")

        assert "$schema" in schema or "type" in schema
        assert schema.get("type") == "object"


class TestLoadSpec:
    """Tests for load_spec function."""

    def test_loads_existing_spec(self):
        spec = Utils.load_spec("test.yaml")

        assert isinstance(spec, dict)
        assert "variables" in spec

    def test_raises_error_for_nonexistent_spec(self):
        with pytest.raises(FileNotFoundError):
            Utils.load_spec("nonexistent.yaml")


