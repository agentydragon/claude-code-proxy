"""Tests for conversation tracking and reasoning utilities."""

import pytest
from claude_code_proxy.tracking import tracker
from claude_code_proxy.converter import parse_json_arguments, _convert_tool_call_to_function_call

# Aliases for legacy test names
_messages_equal_ignoring_thinking = tracker.messages_equal_ignoring_thinking
_count_thinking_blocks = tracker.count_thinking_blocks


class TestMessageUtilities:
    def test_messages_equal_ignoring_thinking_simple(self):
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "user", "content": "Hello"}
        assert _messages_equal_ignoring_thinking(msg1, msg2)

        msg3 = {"role": "user", "content": "Hi"}
        assert not _messages_equal_ignoring_thinking(msg1, msg3)

    def test_messages_equal_ignoring_thinking_blocks(self):
        msg1 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "A"},
                {"type": "text", "text": "Answer"},
            ],
        }
        msg2 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "B"},
                {"type": "text", "text": "Answer"},
            ],
        }
        assert _messages_equal_ignoring_thinking(msg1, msg2)

        msg3 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "A"},
                {"type": "text", "text": "Wrong"},
            ],
        }
        assert not _messages_equal_ignoring_thinking(msg1, msg3)

    def test_count_thinking_blocks(self):
        messages = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "R1"},
                    {"type": "text", "text": "A"},
                    {"type": "thinking", "text": "R2"},
                ],
            },
        ]
        assert _count_thinking_blocks(messages) == 2


class TestJSONParsing:
    def test_parse_valid_json(self):
        result = parse_json_arguments('{"key": "value"}', "tool")
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        result = parse_json_arguments('{"key": invalid}', "tool")
        assert "error" in result


class TestConverterHelpers:
    def test_convert_tool_call_to_function_call(self):
        tool_use = {"id": "tool1", "name": "calc", "input": {"a": 1}}
        out = _convert_tool_call_to_function_call(tool_use)
        assert out["type"] == "function_call"


class TestConversationLogic:
    def test_append_concept(self):
        # simple prefix match
        cached = [{"role": "u", "content": "1"}, {"role": "a", "content": "ok"}]
        new = cached + [{"role": "u", "content": "2"}]
        tracker.update("c1", cached, False, False)
        cid, is_app, idx = tracker.detect_append(new, "x")
        assert is_app and idx == len(cached)


if __name__ == "__main__":
    pytest.main([__file__])
