import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os

from src.pipeline import run_pipeline, main


@pytest.fixture
def mock_schema():
    """Return a mock schema."""
    return {
        "type": "object",
        "required": ["variables", "assumptions", "validation"],
        "properties": {
            "variables": {"type": "object"},
            "assumptions": {"type": "object"},
            "validation": {"type": "object"}
        }
    }


@pytest.fixture
def mock_spec():
    """Return a mock validated spec."""
    return {
        "variables": {"x": {"type": "float"}, "y": {"type": "float"}},
        "assumptions": {
            "marginals": [{
                "variable": "x",
                "distribution": "uniform",
                "params": {"min": 0.0, "max": 1.0}
            }],
            "relationships": [{
                "type": "linear",
                "independent": "x",
                "dependent": "y",
                "slope": 2.0,
                "noise": {"distribution": "normal", "params": {"mean": 0.0, "std": 1.0}}
            }]
        },
        "validation": {
            "distribution_test": {"test": "ks", "alpha": 0.05},
            "relationship_tolerance": {"slope": 0.2}
        }
    }


@pytest.fixture
def mock_yaml_response():
    """Return mock YAML from LLM."""
    return """```yaml
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
```"""


@pytest.fixture
def mock_code_response():
    """Return mock code from LLM."""
    return """def relationship_generation(num_samples, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, num_samples)
    y = 2.0 * x + rng.normal(0, 1, num_samples)
    return x, y"""


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @pytest.mark.asyncio
    async def test_loads_schema(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema) as mock_load:
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_load.assert_called_once_with("linear_regression.schema.json")

    @pytest.mark.asyncio
    async def test_creates_validator_with_schema(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                MockValidator.assert_called_once_with(mock_schema)

    @pytest.mark.asyncio
    async def test_creates_llm_client_with_model(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'my-model'}):
                                await run_pipeline()

                                MockClient.assert_called_once_with('my-model')

    @pytest.mark.asyncio
    async def test_generates_spec_from_intent(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_client.generate_spec_from_intent.assert_called_once()
                                call_args = mock_client.generate_spec_from_intent.call_args
                                assert call_args[0][0] == mock_schema
                                assert "uniform" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_validates_yaml(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_validator.validate_yaml.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_validated_yaml(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml') as mock_write_yaml:
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_write_yaml.assert_called_once_with(mock_spec, "specs/test.yaml")

    @pytest.mark.asyncio
    async def test_generates_code(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_client.generate_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_writes_generated_code(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code') as mock_write_code:
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                mock_write_code.assert_called_once_with(
                                    mock_code_response,
                                    "experiments/code.py"
                                )

    @pytest.mark.asyncio
    async def test_prints_yaml_in_debug_mode(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response, capsys):
        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model', 'DEBUG': 'true'}):
                                await run_pipeline()

                                captured = capsys.readouterr()
                                assert "###" in captured.out


class TestMain:
    """Tests for main function."""

    @pytest.mark.asyncio
    async def test_loads_dotenv(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_dotenv') as mock_dotenv:
            with patch('src.pipeline.load_schema', return_value=mock_schema):
                with patch('src.pipeline.Validator') as MockValidator:
                    mock_validator = MagicMock()
                    mock_validator.validate_yaml.return_value = mock_spec
                    MockValidator.return_value = mock_validator

                    with patch('src.pipeline.OllamaClient') as MockClient:
                        mock_client = MagicMock()
                        mock_client.generate_spec_from_intent = AsyncMock(return_value=mock_yaml_response)
                        mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                        MockClient.return_value = mock_client

                        with patch('src.pipeline.write_generated_yaml'):
                            with patch('src.pipeline.write_generated_code'):
                                with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                    await main()

                                    mock_dotenv.assert_called_once()

    @pytest.mark.asyncio
    async def test_calls_run_pipeline(self, mock_schema, mock_spec, mock_yaml_response, mock_code_response):
        with patch('src.pipeline.load_dotenv'):
            with patch('src.pipeline.run_pipeline', new_callable=AsyncMock) as mock_pipeline:
                await main()

                mock_pipeline.assert_called_once()


class TestExtractYamlIntegration:
    """Tests for YAML extraction within the pipeline."""

    @pytest.mark.asyncio
    async def test_extracts_yaml_from_fenced_response(self, mock_schema, mock_spec, mock_code_response):
        fenced_yaml = """Here is the spec:
```yaml
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
```
That's the specification."""

        with patch('src.pipeline.load_schema', return_value=mock_schema):
            with patch('src.pipeline.Validator') as MockValidator:
                mock_validator = MagicMock()
                mock_validator.validate_yaml.return_value = mock_spec
                MockValidator.return_value = mock_validator

                with patch('src.pipeline.OllamaClient') as MockClient:
                    mock_client = MagicMock()
                    mock_client.generate_spec_from_intent = AsyncMock(return_value=fenced_yaml)
                    mock_client.generate_code = AsyncMock(return_value=mock_code_response)
                    MockClient.return_value = mock_client

                    with patch('src.pipeline.write_generated_yaml'):
                        with patch('src.pipeline.write_generated_code'):
                            with patch.dict(os.environ, {'LLM_MODEL': 'test-model'}):
                                await run_pipeline()

                                call_args = mock_validator.validate_yaml.call_args[0][0]
                                assert "variables:" in call_args
                                assert "Here is the spec" not in call_args
