"""Test the enable_append_tracking configuration flag."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_code_proxy.config import ProxyConfig
from claude_code_proxy.server import app
from claude_code_proxy.telemetry import setup_telemetry


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "test-response-id",
        "created_at": 1234567890,
        "model": "o4-mini",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Test response"}],
            }
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    
    # Create an async mock for responses.create
    async def async_create(**kwargs):
        return mock_response
    
    mock_client.responses.create = AsyncMock(side_effect=async_create)
    return mock_client


@pytest.mark.asyncio
async def test_append_tracking_disabled(mock_openai_client):
    """Test that disabling append tracking sends full conversation."""
    
    # Create config with append tracking disabled
    config_disabled = ProxyConfig(
        openai_api_key="test-key",
        enable_append_tracking=False,
        anthropic_to_openai_model={"o4-mini": "o4-mini"}
    )
    
    # First request - establish conversation
    first_request = {
        "model": "o4-mini",
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about first message"},
                {"type": "text", "text": "First response"}
            ]}
        ]
    }
    
    # Second request - append to conversation
    second_request = {
        "model": "o4-mini",
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about first message"},
                {"type": "text", "text": "First response"}
            ]},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about second message"},
                {"type": "text", "text": "Second response"}
            ]}
        ]
    }
    
    # Initialize telemetry
    setup_telemetry()
    
    with patch("claude_code_proxy.server_otel.config", config_disabled):
        # Import after patching to ensure config is loaded correctly
        from claude_code_proxy.server_otel import handle_anthropic_otel
        
        # Make first request
        await handle_anthropic_otel(first_request, mock_openai_client)
        
        # Make second request (append)
        response = await handle_anthropic_otel(second_request, mock_openai_client)
        
        # Check that OpenAI client was called with full conversation
        # (not just the new messages)
        call_args = mock_openai_client.responses.create.call_args_list[-1]
        openai_request = call_args.kwargs
        
        # Should have all 4 messages (2 user, 2 assistant) in the input
        assert len(openai_request["input"]) >= 4, (
            f"Expected full conversation in input, but got {len(openai_request['input'])} items"
        )
        
        # Verify the content of messages
        input_contents = [
            item.get("content", "") 
            for item in openai_request["input"] 
            if "content" in item
        ]
        assert "First message" in str(input_contents)
        assert "Second message" in str(input_contents)


@pytest.mark.asyncio
async def test_append_tracking_enabled(mock_openai_client):
    """Test that enabling append tracking sends only new messages."""
    
    # Create config with append tracking enabled (default)
    config_enabled = ProxyConfig(
        openai_api_key="test-key",
        enable_append_tracking=True,
        anthropic_to_openai_model={"o4-mini": "o4-mini"}
    )
    
    # Same requests as above
    first_request = {
        "model": "o4-mini",
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about first message"},
                {"type": "text", "text": "First response"}
            ]}
        ]
    }
    
    second_request = {
        "model": "o4-mini",
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about first message"},
                {"type": "text", "text": "First response"}
            ]},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": [
                {"type": "thinking", "text": "Thinking about second message"},
                {"type": "text", "text": "Second response"}
            ]}
        ]
    }
    
    # Initialize telemetry
    setup_telemetry()
    
    with patch("claude_code_proxy.server_otel.config", config_enabled):
        from claude_code_proxy.server_otel import handle_anthropic_otel
        
        # Make first request
        await handle_anthropic_otel(first_request, mock_openai_client)
        
        # Make second request (append)
        response = await handle_anthropic_otel(second_request, mock_openai_client)
        
        # Check that OpenAI client was called with only new messages
        call_args = mock_openai_client.responses.create.call_args_list[-1]
        openai_request = call_args.kwargs
        
        # Should have only the new messages (last 2)
        assert len(openai_request["input"]) == 2, (
            f"Expected only new messages in input, but got {len(openai_request['input'])} items"
        )
        
        # Verify only second message is present
        input_contents = [
            item.get("content", "") 
            for item in openai_request["input"] 
            if "content" in item
        ]
        assert "Second message" in str(input_contents)
        assert "First message" not in str(input_contents)


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v", "-s"]))