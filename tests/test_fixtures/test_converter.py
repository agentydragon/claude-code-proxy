"""Unit tests for the Anthropic <-> OpenAI converter."""


import pytest

from claude_code_proxy.converter import anthropic_to_openai_request


@pytest.fixture
def converter():
    return anthropic_to_openai_request


class TestAnthropicToOpenAIRequest:
    def test_simple_text_message(self):
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
