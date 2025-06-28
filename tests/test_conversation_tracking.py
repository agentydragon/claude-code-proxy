"""Tests for conversation tracking and reasoning utilities."""

import pytest

from claude_code_proxy.converter import _convert_tool_call_to_function_call, parse_json_arguments
from claude_code_proxy.tracking import tracker

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
        tracker._cache.clear()
        tracker._index.clear()
        initial_messages = [{"role": "u", "content": "1"}, {"role": "a", "content": "ok"}]
        appended_messages = initial_messages + [{"role": "u", "content": "2"}]
        tracker.update(
            "c1",
            initial_messages,
            had_reasoning_filtered=False,
            original_had_reasoning=False,
        )
        conversation_id, is_append, append_index = tracker.detect_append(appended_messages, "x")
        assert conversation_id == "c1"
        assert is_append and append_index == len(initial_messages)

    def test_detect_no_append_on_mismatch(self):
        # prefix differs -> no append
        tracker._cache.clear()
        tracker._index.clear()
        original_messages = [{"role": "u", "content": "A"}, {"role": "a", "content": "B"}]
        tracker.update(
            "c1",
            original_messages,
            had_reasoning_filtered=False,
            original_had_reasoning=False,
        )
        mismatched_messages = [{"role": "u", "content": "A"}, {"role": "a", "content": "C"}]
        conversation_id, is_append, append_index = tracker.detect_append(mismatched_messages, "x")
        assert conversation_id == "x"
        assert not is_append and append_index == -1

    def test_multiple_sequential_appends(self):
        # multiple updates preserve append detection
        tracker._cache.clear()
        tracker._index.clear()
        messages1 = [{"role": "u", "content": "1"}]
        tracker.update(
            "c1",
            messages1,
            had_reasoning_filtered=False,
            original_had_reasoning=False,
        )
        messages2 = messages1 + [{"role": "a", "content": "response"}]
        conv_id1, is_append1, append_index1 = tracker.detect_append(messages2, "x")
        assert conv_id1 == "c1"
        assert is_append1 and append_index1 == len(messages1)
        tracker.update(
            conv_id1,
            messages2,
            had_reasoning_filtered=False,
            original_had_reasoning=False,
        )
        messages3 = messages2 + [{"role": "u", "content": "2"}]
        conv_id2, is_append2, append_index2 = tracker.detect_append(messages3, "x")
        assert conv_id2 == "c1"
        assert is_append2 and append_index2 == len(messages2)

    def test_thinking_blocks_ignored_in_append(self):
        # thinking blocks don't break append detection
        tracker._cache.clear()
        tracker._index.clear()
        original_messages = [
            {"role": "assistant", "content": [{"type": "thinking", "text": "X"}, {"type": "text", "text": "Y"}]}
        ]
        tracker.update(
            "c1",
            original_messages,
            had_reasoning_filtered=False,
            original_had_reasoning=False,
        )
        new_messages = [
            {"role": "assistant", "content": [{"type": "thinking", "text": "Z"}, {"type": "text", "text": "Y"}]},
            {"role": "user", "content": "next"},
        ]
        conversation_id, is_append, append_index = tracker.detect_append(new_messages, "x")
        assert conversation_id == "c1"
        assert is_append and append_index == 1

    def test_no_append_on_identical_messages(self):
        # identical history should not count as append
        tracker._cache.clear()
        tracker._index.clear()
        msgs = [{"role": "u", "content": "hi"}, {"role": "a", "content": "ok"}]
        tracker.update("c1", msgs, had_reasoning_filtered=False, original_had_reasoning=False)
        cid, is_append, idx = tracker.detect_append(msgs, "x")
        assert cid == "x" and not is_append and idx == -1


if __name__ == "__main__":
    pytest.main([__file__])
# mypy: ignore_errors
