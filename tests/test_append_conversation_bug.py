"""Test to reproduce the bug where conversation history is dropped during append requests."""

import json
from unittest.mock import patch

import pytest

from claude_code_proxy.config import ProxyConfig
from claude_code_proxy.converter import anthropic_to_openai_request


@pytest.fixture
def mock_config():
    """Mock config with test-specific model mappings."""
    config = ProxyConfig(
        anthropic_to_openai_model={
            "o4-mini": "o4-mini",  # Direct mapping for this test
        }
    )
    return config


def test_append_request_preserves_full_conversation_history(mock_config):
    """Test that append requests preserve the full conversation history."""
    
    # Create a complex conversation similar to the bug report
    anthropic_req = {
        "model": "o4-mini",
        "max_tokens": 4096,
        "messages": [
            # First user message with task
            {
                "role": "user",
                "content": "You are an agent executing a task.\n\nTask: Split the bundle.js into 1MB chunks"
            },
            # Second user message (duplicate)
            {
                "role": "user", 
                "content": "You are an agent executing a task.\n\nTask: Split the bundle.js into 1MB chunks"
            },
            # Assistant response with thinking and tool use
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "I need to list the directory first"},
                    {"type": "tool_use", "id": "call_YVaSkLavOmkQgDcNWSGEMauy", "name": "List", "input": {"dir_path": ""}}
                ]
            },
            # User with tool result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_YVaSkLavOmkQgDcNWSGEMauy",
                        "content": "Contents of :\n[DIR]  analysis/ (3 items)\n[FILE] bundle.js (3.8 MB)"
                    }
                ]
            },
            # Another assistant response with thinking and tool use
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Now I'll check the package directory"},
                    {"type": "thinking", "text": "Let me see what's in there"},
                    {"type": "tool_use", "id": "call_zuk5xqsPok7i8sIQo3dADkMf", "name": "List", "input": {"dir_path": "package"}}
                ]
            },
            # Another tool result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_zuk5xqsPok7i8sIQo3dADkMf",
                        "content": "Contents of package:\n[FILE] __init__.py (28 bytes)"
                    }
                ]
            },
            # More assistant messages with tool uses
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "I'll read the first chunk"},
                    {"type": "tool_use", "id": "call_ioKQkzSj37bgeZOjnjzlg2eH", "name": "Read", "input": {"file_path": "package/chunk_000.py", "start_byte": 0, "end_byte": 5000}}
                ]
            },
            # Tool result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_ioKQkzSj37bgeZOjnjzlg2eH", 
                        "content": "File: package/chunk_000.py (total size: 1.6 KB)"
                    }
                ]
            },
            # Final assistant message
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "Let me list the directory again"},
                    {"type": "tool_use", "id": "call_5DIaB43SuRY2IYfKlvbCvVxf", "name": "List", "input": {"dir_path": ""}}
                ]
            },
            # Final tool result
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_5DIaB43SuRY2IYfKlvbCvVxf",
                        "content": "Contents of :\n[DIR]  analysis/ (3 items)"
                    }
                ]
            }
        ],
        "tools": [
            {
                "name": "List",
                "description": "List the contents of a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dir_path": {"type": "string", "description": "Path to directory"}
                    },
                    "required": ["dir_path"]
                }
            },
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object", 
                    "properties": {
                        "file_path": {"type": "string"},
                        "start_byte": {"type": "integer", "default": 0},
                        "end_byte": {"type": "integer"}
                    },
                    "required": ["file_path"]
                }
            }
        ]
    }
    
    model_mappings = {"o4-mini": "o4-mini"}
    
    with (
        patch("claude_code_proxy.converter.CONFIG", mock_config),
        patch("claude_code_proxy.converter.ANTHROPIC_TO_OPENAI_MODEL", model_mappings),
    ):
        openai_req = anthropic_to_openai_request(anthropic_req)
        
        # Print for debugging
        print(f"\nTotal messages in Anthropic request: {len(anthropic_req['messages'])}")
        print(f"Total input items in OpenAI request: {len(openai_req['input'])}")
        
        # Print the actual OpenAI request body to compare with bug trace
        print("\nActual OpenAI request body (truncated):")
        openai_req_json = json.dumps(openai_req, indent=2)
        print(openai_req_json[:2000] + "..." if len(openai_req_json) > 2000 else openai_req_json)
        
        # Print each input item type
        print("\nOpenAI input items:")
        for i, item in enumerate(openai_req['input']):
            if 'role' in item:
                print(f"  [{i}] {item['role']} message: {str(item.get('content', ''))[:50]}...")
            elif item.get('type') == 'function_call':
                print(f"  [{i}] function_call: {item['name']}")
            elif item.get('type') == 'function_call_output':
                print(f"  [{i}] function_call_output: {item.get('call_id', 'unknown')}")
        
        # The bug: Only the last tool call and result are included
        # We should have many more items in the input array
        assert len(openai_req['input']) > 2, (
            f"Expected more than 2 input items, but got {len(openai_req['input'])}. "
            f"This confirms the bug - most conversation history was dropped!"
        )
        
        # Check that we have the initial user messages
        user_messages = [item for item in openai_req['input'] if item.get('role') == 'user']
        assert len(user_messages) >= 2, "Should preserve initial user messages"
        
        # Check that we have multiple tool calls
        function_calls = [item for item in openai_req['input'] if item.get('type') == 'function_call']
        assert len(function_calls) >= 3, f"Should have at least 3 function calls, got {len(function_calls)}"


if __name__ == "__main__":
    # Run the test
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))