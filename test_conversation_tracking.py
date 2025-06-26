"""Simplified tests for conversation tracking and reasoning preservation logic."""

import pytest
from claude_code_proxy.server import _messages_equal_ignoring_thinking, _count_thinking_blocks
from claude_code_proxy.converter import parse_json_arguments, _convert_tool_call_to_function_call


class TestMessageUtilities:
    """Test message comparison and counting utilities."""
    
    def test_messages_equal_ignoring_thinking_simple(self):
        """Test comparing simple text messages."""
        msg1 = {"role": "user", "content": "Hello"}
        msg2 = {"role": "user", "content": "Hello"}
        assert _messages_equal_ignoring_thinking(msg1, msg2)
        
        msg3 = {"role": "user", "content": "Hi"}
        assert not _messages_equal_ignoring_thinking(msg1, msg3)
    
    def test_messages_equal_ignoring_thinking_blocks(self):
        """Test that thinking blocks are ignored in comparison."""
        msg1 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "Internal reasoning A"},
                {"type": "text", "text": "The answer is 42"}
            ]
        }
        msg2 = {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "Different reasoning B"},
                {"type": "text", "text": "The answer is 42"}
            ]
        }
        # Should be equal since thinking blocks are ignored
        assert _messages_equal_ignoring_thinking(msg1, msg2)
        
        # Different text should not be equal
        msg3 = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think..."},
                {"type": "thinking", "text": "Internal reasoning"},
                {"type": "text", "text": "The answer is 43"}  # Different
            ]
        }
        assert not _messages_equal_ignoring_thinking(msg1, msg3)
    
    def test_count_thinking_blocks(self):
        """Test counting thinking blocks across messages."""
        messages = [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Reasoning 1"},
                    {"type": "text", "text": "Answer"},
                    {"type": "thinking", "text": "Reasoning 2"}
                ]
            },
            {"role": "user", "content": "Follow up"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {"type": "thinking", "text": "Reasoning 3"}
                ]
            }
        ]
        assert _count_thinking_blocks(messages) == 3
        
        # Test with no thinking blocks
        messages_no_thinking = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        assert _count_thinking_blocks(messages_no_thinking) == 0


class TestJSONParsing:
    """Test JSON argument parsing with error recovery."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON arguments."""
        result = parse_json_arguments('{"key": "value"}', "test_tool")
        assert result == {"key": "value"}
        
        # Test with dict input
        result = parse_json_arguments({"key": "value"}, "test_tool")
        assert result == {"key": "value"}
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns error object."""
        result = parse_json_arguments('{"key": invalid}', "test_tool")
        
        assert "error" in result
        assert "Failed to parse JSON arguments for tool" in result["error"]
        assert result["raw_arguments"] == '{"key": invalid}'
        assert "test_tool" in result["tool_name"]
        assert "instruction" in result
        assert "unescaped quotes" in result["instruction"]
    
    def test_parse_empty_arguments(self):
        """Test parsing empty arguments."""
        assert parse_json_arguments("", "test_tool") == {}
        assert parse_json_arguments(None, "test_tool") == {}
        assert parse_json_arguments({}, "test_tool") == {}


class TestConverterHelpers:
    """Test converter helper functions."""
    
    def test_convert_tool_call_to_function_call(self):
        """Test converting Anthropic tool_use to OpenAI function_call."""
        tool_use = {
            "id": "tool_123",
            "name": "calculator",
            "input": {"operation": "add", "a": 1, "b": 2}
        }
        
        result = _convert_tool_call_to_function_call(tool_use)
        
        assert result["type"] == "function_call"
        assert result["name"] == "calculator"
        assert result["call_id"] == "tool_123"
        assert result["arguments"] == '{"operation": "add", "a": 1, "b": 2}'


class TestConversationLogic:
    """Test conversation tracking logic concepts."""
    
    def test_append_scenario_concept(self):
        """Test the concept of append detection."""
        # Simulate cached messages
        cached_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
        
        # New messages that continue the conversation
        new_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What about 3+3?"},
            {"role": "assistant", "content": "6"}
        ]
        
        # This would be an append scenario
        assert len(new_messages) > len(cached_messages)
        
        # First messages should match
        for i, cached_msg in enumerate(cached_messages):
            assert _messages_equal_ignoring_thinking(new_messages[i], cached_msg)
    
    def test_reasoning_preservation_concept(self):
        """Test the concept of reasoning preservation in appends."""
        # Original conversation with reasoning (would be filtered)
        original = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Greeting analysis"},
                    {"type": "text", "text": "Hi!"}
                ]
            }
        ]
        
        # When appending, new messages can have reasoning
        append_messages = [
            {"role": "user", "content": "How are you?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Status check"},
                    {"type": "text", "text": "I'm doing well!"}
                ]
            }
        ]
        
        # Count reasoning blocks
        original_reasoning = _count_thinking_blocks(original)
        append_reasoning = _count_thinking_blocks(append_messages)
        
        assert original_reasoning == 1
        assert append_reasoning == 1
        
        # In append scenario, the append_reasoning would be preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])