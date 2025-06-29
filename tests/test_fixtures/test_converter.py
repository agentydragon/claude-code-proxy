"""Unit tests for the Anthropic <-> OpenAI converter."""

from unittest.mock import patch

import pytest

from claude_code_proxy.config import ProxyConfig
from claude_code_proxy.converter import anthropic_to_openai_request


@pytest.fixture
def mock_config():
    """Mock config with test-specific model mappings."""
    config = ProxyConfig(
        anthropic_to_openai_model={
            "claude-3-opus-20240229": "gpt-4o",
            "claude-3-5-sonnet-20241022": "gpt-4o",
            "claude-3-5-haiku-20241022": "gpt-4o-mini",
            "claude-opus-4-20250514": "o3",
            "claude-sonnet-4-20250514": "o3-mini",
        }
    )
    return config


class TestAnthropicToOpenAIRequest:
    def test_simple_text_message(self, mock_config):
        model_mappings = {
            "claude-3-opus-20240229": "gpt-4o",
            "claude-3-5-sonnet-20241022": "gpt-4o",
            "claude-3-5-haiku-20241022": "gpt-4o-mini",
            "claude-opus-4-20250514": "o3",
            "claude-sonnet-4-20250514": "o3-mini",
        }
        with (
            patch("claude_code_proxy.converter.CONFIG", mock_config),
            patch("claude_code_proxy.converter.ANTHROPIC_TO_OPENAI_MODEL", model_mappings),
        ):
            anthropic_req = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "max_tokens": 100,
            }
            openai_req = anthropic_to_openai_request(anthropic_req)
            assert openai_req["model"] == "gpt-4o"
            assert openai_req["input"][0]["role"] == "user"
            assert openai_req["input"][0]["content"] == "Hello, world!"
            assert openai_req["max_output_tokens"] == 100


# mypy: ignore_errors
