"""Unit tests for the Anthropic <-> OpenAI converter."""

import pytest

from claude_code_proxy.converter import AnthropicOpenAIConverter


@pytest.fixture
def converter():
    return AnthropicOpenAIConverter()


class TestAnthropicToOpenAIRequest:
    """Test converting Anthropic requests to OpenAI format."""

    def test_simple_text_message(self, converter):
        """Test converting a simple text message."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["model"] == "claude-3-opus-20240229"
        assert openai_req["messages"] == [{"role": "user", "content": "Hello, world!"}]
        assert openai_req["max_tokens"] == 100

    def test_system_message_string(self, converter):
        """Test converting string system message."""
        anthropic_req = {
            "model": "claude-3-haiku-20240307",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert openai_req["messages"][1] == {"role": "user", "content": "Hi"}

    def test_system_message_blocks(self, converter):
        """Test converting structured system message."""
        anthropic_req = {
            "model": "claude-3-sonnet-20240229",
            "system": [{"type": "text", "text": "You are helpful."}, {"type": "text", "text": "Be concise."}],
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["messages"][0] == {"role": "system", "content": "You are helpful.\nBe concise."}

    def test_image_content(self, converter):
        """Test converting image content."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "base64data"},
                        },
                    ],
                }
            ],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["messages"][0]["role"] == "user"
        assert openai_req["messages"][0]["content"] == [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,base64data"}},
        ]

    def test_tool_use(self, converter):
        """Test converting tool use."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll check the weather for you."},
                        {"type": "tool_use", "id": "tool_123", "name": "get_weather", "input": {"location": "NYC"}},
                    ],
                },
            ],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert len(openai_req["messages"]) == 2
        assert openai_req["messages"][1]["role"] == "assistant"
        assert openai_req["messages"][1]["content"] == "I'll check the weather for you."
        assert openai_req["messages"][1]["tool_calls"] == [
            {
                "id": "tool_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]

    def test_tool_results(self, converter):
        """Test converting tool results."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool_123", "content": "It's sunny and 72°F"}],
                },
            ],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        # Tool results should be converted to tool messages
        assert len(openai_req["messages"]) == 2
        assert openai_req["messages"][1] == {
            "role": "tool",
            "tool_call_id": "tool_123",
            "content": "It's sunny and 72°F",
        }

    def test_thinking_blocks_filtered(self, converter):
        """Test that thinking blocks are filtered out."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Here's my answer"},
                    ],
                }
            ],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        # Thinking block should be filtered out
        assert openai_req["messages"][0] == {"role": "assistant", "content": "Here's my answer"}

    def test_tools_conversion(self, converter):
        """Test converting tools."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

    def test_tool_choice_conversion(self, converter):
        """Test converting tool_choice."""
        # Test auto
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
            "max_tokens": 100,
        }
        openai_req = converter.anthropic_to_openai_request(anthropic_req)
        assert openai_req["tool_choice"] == "auto"

        # Test any
        anthropic_req["tool_choice"] = {"type": "any"}
        openai_req = converter.anthropic_to_openai_request(anthropic_req)
        assert openai_req["tool_choice"] == "required"

        # Test specific tool
        anthropic_req["tool_choice"] = {"type": "tool", "name": "get_weather"}
        openai_req = converter.anthropic_to_openai_request(anthropic_req)
        assert openai_req["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_metadata_conversion(self, converter):
        """Test converting metadata."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hi"}],
            "metadata": {"user_id": "user123"},
            "max_tokens": 100,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["user"] == "user123"

    def test_all_parameters(self, converter):
        """Test converting all parameters."""
        anthropic_req = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["\\n\\n", "END"],
            "stream": True,
        }

        openai_req = converter.anthropic_to_openai_request(anthropic_req)

        assert openai_req["max_tokens"] == 150
        assert openai_req["temperature"] == 0.7
        assert openai_req["top_p"] == 0.9
        assert openai_req["stop"] == ["\\n\\n", "END"]
        assert openai_req["stream"] is True


class TestOpenAIToAnthropicResponse:
    """Test converting OpenAI responses to Anthropic format."""

    def test_simple_text_response(self, converter):
        """Test converting a simple text response."""
        openai_resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello! How can I help you?"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

        anthropic_resp = converter.openai_to_anthropic_response(openai_resp)

        assert anthropic_resp["type"] == "message"
        assert anthropic_resp["role"] == "assistant"
        assert anthropic_resp["content"] == [{"type": "text", "text": "Hello! How can I help you?"}]
        assert anthropic_resp["model"] == "gpt-4"
        assert anthropic_resp["stop_reason"] == "end_turn"
        assert anthropic_resp["usage"] == {"input_tokens": 10, "output_tokens": 8}

    def test_tool_calls_response(self, converter):
        """Test converting response with tool calls."""
        openai_resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }

        anthropic_resp = converter.openai_to_anthropic_response(openai_resp)

        assert len(anthropic_resp["content"]) == 2
        assert anthropic_resp["content"][0] == {"type": "text", "text": "I'll check the weather for you."}
        assert anthropic_resp["content"][1] == {
            "type": "tool_use",
            "id": "call_123",
            "name": "get_weather",
            "input": {"location": "NYC"},
        }
        assert anthropic_resp["stop_reason"] == "tool_use"

    def test_stop_reason_mapping(self, converter):
        """Test stop reason mapping."""
        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("function_call", "tool_use"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),  # No direct mapping, defaults to end_turn
            ("unknown_reason", "end_turn"),  # Unknown reasons default to end_turn
        ]

        for openai_reason, expected_anthropic in test_cases:
            openai_resp = {
                "id": "chatcmpl-123",
                "model": "gpt-4",
                "choices": [{"message": {"role": "assistant", "content": "Test"}, "finish_reason": openai_reason}],
            }

            anthropic_resp = converter.openai_to_anthropic_response(openai_resp)
            assert anthropic_resp["stop_reason"] == expected_anthropic

    def test_empty_content(self, converter):
        """Test response with empty content."""
        openai_resp = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        anthropic_resp = converter.openai_to_anthropic_response(openai_resp)

        # Should only have tool use, no text content
        assert len(anthropic_resp["content"]) == 1
        assert anthropic_resp["content"][0]["type"] == "tool_use"


# mypy: ignore_errors
