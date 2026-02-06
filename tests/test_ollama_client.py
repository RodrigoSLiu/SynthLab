import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os

from src.llm.ollama_client import OllamaClient, LLMError


@pytest.fixture
def client():
    """Create an OllamaClient instance for testing."""
    return OllamaClient(model="test-model")


@pytest.fixture
def sample_schema():
    """Return a sample schema for testing."""
    return {
        "type": "object",
        "properties": {
            "variables": {"type": "object"}
        }
    }


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_stores_model_name(self):
        client = OllamaClient(model="codellama:7b")

        assert client.model == "codellama:7b"

    def test_initializes_context_to_none(self):
        client = OllamaClient(model="test")

        assert client.context is None

    def test_accepts_any_model_name(self):
        client = OllamaClient(model="custom-model:latest")

        assert client.model == "custom-model:latest"


class TestGenerateSpecFromIntent:
    """Tests for OllamaClient.generate_spec_from_intent method."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, client, sample_schema):
        expected_response = "variables:\n  x:\n    type: float"

        with patch.object(client, '_get_llm_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = expected_response

            result = await client.generate_spec_from_intent(
                sample_schema,
                "Generate x uniformly distributed"
            )

            assert result == expected_response

    @pytest.mark.asyncio
    async def test_raises_error_for_empty_intent(self, client, sample_schema):
        with pytest.raises(LLMError, match="User intent is empty"):
            await client.generate_spec_from_intent(sample_schema, "")

    @pytest.mark.asyncio
    async def test_raises_error_for_whitespace_intent(self, client, sample_schema):
        with pytest.raises(LLMError, match="User intent is empty"):
            await client.generate_spec_from_intent(sample_schema, "   \n\t   ")

    @pytest.mark.asyncio
    async def test_calls_llm_with_prompt(self, client, sample_schema):
        with patch.object(client, '_get_llm_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "response"

            await client.generate_spec_from_intent(
                sample_schema,
                "Generate linear regression data"
            )

            mock_llm.assert_called_once()
            prompt_arg = mock_llm.call_args[0][0]
            assert "Generate linear regression data" in prompt_arg

    @pytest.mark.asyncio
    async def test_includes_schema_in_prompt(self, client, sample_schema):
        with patch.object(client, '_get_llm_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "response"

            await client.generate_spec_from_intent(sample_schema, "test intent")

            prompt_arg = mock_llm.call_args[0][0]
            assert "variables" in prompt_arg


class TestGenerateCode:
    """Tests for OllamaClient.generate_code method."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, client):
        expected_code = "def relationship_generation(num_samples, seed):\n    pass"

        with patch.object(client, '_get_llm_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = expected_code

            result = await client.generate_code("Generate code for linear regression")

            assert result == expected_code

    @pytest.mark.asyncio
    async def test_raises_error_for_empty_prompt(self, client):
        with pytest.raises(LLMError, match="Prompt is empty"):
            await client.generate_code("")

    @pytest.mark.asyncio
    async def test_raises_error_for_whitespace_prompt(self, client):
        with pytest.raises(LLMError, match="Prompt is empty"):
            await client.generate_code("   \n\t   ")

    @pytest.mark.asyncio
    async def test_calls_llm_with_prompt(self, client):
        code_prompt = "Generate numpy code for data generation"

        with patch.object(client, '_get_llm_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "code"

            await client.generate_code(code_prompt)

            mock_llm.assert_called_once_with(code_prompt)


class TestGetLLMResponse:
    """Tests for OllamaClient._get_llm_response method."""

    @pytest.mark.asyncio
    async def test_calls_ollama_chat(self, client):
        mock_response = {
            'message': {'content': 'Generated text'}
        }

        with patch('src.llm.ollama_client.chat', return_value=mock_response) as mock_chat:
            result = await client._get_llm_response("Test prompt")

            mock_chat.assert_called_once()
            assert result == 'Generated text'

    @pytest.mark.asyncio
    async def test_uses_correct_model(self, client):
        mock_response = {'message': {'content': 'response'}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response) as mock_chat:
            await client._get_llm_response("prompt")

            call_kwargs = mock_chat.call_args
            assert call_kwargs[1]['model'] == 'test-model'

    @pytest.mark.asyncio
    async def test_sends_user_message(self, client):
        mock_response = {'message': {'content': 'response'}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response) as mock_chat:
            await client._get_llm_response("My test prompt")

            call_kwargs = mock_chat.call_args
            messages = call_kwargs[1]['messages']
            assert len(messages) == 1
            assert messages[0]['role'] == 'user'
            assert 'My test prompt' in messages[0]['content']

    @pytest.mark.asyncio
    async def test_prints_debug_output_when_debug_enabled(self, client, capsys):
        mock_response = {'message': {'content': 'LLM response text'}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response):
            with patch.dict(os.environ, {'DEBUG': 'true'}):
                await client._get_llm_response("Test prompt for debug")

                captured = capsys.readouterr()
                assert "User:" in captured.out
                assert "Ollama:" in captured.out

    @pytest.mark.asyncio
    async def test_no_debug_output_when_debug_disabled(self, client, capsys):
        mock_response = {'message': {'content': 'response'}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('DEBUG', None)
                await client._get_llm_response("Test prompt")

                captured = capsys.readouterr()
                assert "User:" not in captured.out


class TestLLMError:
    """Tests for LLMError exception."""

    def test_is_exception(self):
        assert issubclass(LLMError, Exception)

    def test_stores_message(self):
        error = LLMError("Something went wrong")

        assert str(error) == "Something went wrong"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LLMError) as exc_info:
            raise LLMError("test error")

        assert "test error" in str(exc_info.value)


class TestOllamaClientIntegration:
    """Integration-style tests for OllamaClient (still mocked)."""

    @pytest.mark.asyncio
    async def test_full_spec_generation_flow(self, sample_schema):
        client = OllamaClient(model="codellama:7b")
        mock_response = {'message': {'content': 'variables:\n  x:\n    type: float'}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response):
            result = await client.generate_spec_from_intent(
                sample_schema,
                "Generate x uniformly distributed between 0 and 1"
            )

            assert "variables" in result
            assert "float" in result

    @pytest.mark.asyncio
    async def test_full_code_generation_flow(self):
        client = OllamaClient(model="codellama:7b")
        mock_code = """def relationship_generation(num_samples, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, num_samples)
    return x"""
        mock_response = {'message': {'content': mock_code}}

        with patch('src.llm.ollama_client.chat', return_value=mock_response):
            result = await client.generate_code("Generate code for uniform distribution")

            assert "def relationship_generation" in result
            assert "numpy" in result
