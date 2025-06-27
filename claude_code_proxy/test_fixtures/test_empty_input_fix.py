"""Test case for empty input array causing OpenAI API error."""

import pytest
from unittest.mock import patch, MagicMock
from claude_code_proxy.config import ProxyConfig, ReasoningEffort, ReasoningSummary
from claude_code_proxy.converter import anthropic_to_openai_request


# Create a mock config that won't interfere with the model mapping
mock_config = MagicMock()
mock_config.reasoning_effort = ReasoningEffort.MEDIUM
mock_config.reasoning_summary = ReasoningSummary.AUTO


@patch('claude_code_proxy.converter.CONFIG', mock_config)
@patch('claude_code_proxy.converter.ANTHROPIC_TO_OPENAI_MODEL', {
    "claude-3-5-sonnet-20241022": "gpt-4o",
    "claude-3-5-haiku-20241022": "gpt-4o-mini",
})
def test_empty_input_from_filtered_messages():
    """Test that we handle cases where all messages get filtered out."""
    # Scenario: An append request where all new messages contain only thinking blocks
    anthropic_req = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "text": "This is a thinking block that will be filtered"
                    }
                ]
            }
        ],
        "max_tokens": 100
    }
    
    # This should not raise an exception
    openai_req = anthropic_to_openai_request(anthropic_req)
    
    # The input array should not be empty - we need at least a placeholder
    assert "input" in openai_req
    assert len(openai_req["input"]) > 0 or "prompt" in openai_req or "previous_response_id" in openai_req


@patch('claude_code_proxy.converter.CONFIG', mock_config)
@patch('claude_code_proxy.converter.ANTHROPIC_TO_OPENAI_MODEL', {
    "claude-3-5-sonnet-20241022": "gpt-4o",
    "claude-3-5-haiku-20241022": "gpt-4o-mini",
})
def test_empty_messages_array():
    """Test handling of completely empty messages array."""
    anthropic_req = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [],
        "max_tokens": 100
    }
    
    openai_req = anthropic_to_openai_request(anthropic_req)
    
    # Should handle empty messages gracefully
    assert "input" in openai_req
    assert isinstance(openai_req["input"], list)
    assert len(openai_req["input"]) > 0  # Should have minimal placeholder


@patch('claude_code_proxy.converter.CONFIG', mock_config)
@patch('claude_code_proxy.converter.ANTHROPIC_TO_OPENAI_MODEL', {
    "claude-3-5-sonnet-20241022": "gpt-4o",
    "claude-3-5-haiku-20241022": "gpt-4o-mini",
})
def test_all_messages_unsupported():
    """Test when all messages have unsupported format."""
    anthropic_req = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": None  # This will be skipped
            },
            {
                "role": "assistant",
                "content": 123  # Non-string, non-list content
            }
        ],
        "max_tokens": 100
    }
    
    openai_req = anthropic_to_openai_request(anthropic_req)
    
    # Should handle this case without creating empty input
    assert "input" in openai_req
    assert len(openai_req["input"]) > 0  # Should have minimal placeholder