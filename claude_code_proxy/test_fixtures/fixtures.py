"""Test fixtures for Anthropic <-> OpenAI conversion testing."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TestFixture:
    name: str
    description: str
    anthropic_request: Dict[str, Any]


# Models
HAIKU = "claude-3-5-haiku-20241022"
SONNET = "claude-3-5-sonnet-20241022"

# Common parameters
MAX_TOKENS = 1024

# Common messages
SIMPLE_USER_MSG = {"role": "user", "content": "why lizard"}
MATH_MESSAGES = [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}, {"role": "user", "content": "What about 3+3?"}]
WEATHER_MESSAGES = [{"role": "user", "content": "weather in New York?"}]

# Common tools
WEATHER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
}

# Tool use content blocks
TOOL_USE_ANTHROPIC = {
    "type": "tool_use",
    "id": "toolu_01234",
    "name": "get_weather",
    "input": {"location": "Paris, France", "unit": "celsius"}
}

TOOL_RESULT_CONTENT = "Temperature: 22°C, Conditions: Sunny, Humidity: 45%"

# Test fixtures focusing on different behaviors
FIXTURES: List[TestFixture] = [
    TestFixture(
        name="simple_text",
        description="Basic text exchange",
        anthropic_request={
            "model": HAIKU,
            "messages": [SIMPLE_USER_MSG],
            "max_tokens": MAX_TOKENS,
        }
    ),
    
    TestFixture(
        name="with_system",
        description="System prompt handling",
        anthropic_request={
            "model": SONNET,
            "system": "You are a lizard.",
            "messages": [SIMPLE_USER_MSG],
            "max_tokens": MAX_TOKENS,
        }
    ),
    
    TestFixture(
        name="multi_turn",
        description="Multi-turn conversation",
        anthropic_request={"model": HAIKU, "messages": MATH_MESSAGES, "max_tokens": MAX_TOKENS}
    ),
    
    TestFixture(
        name="structured_content",
        description="Structured content blocks",
        anthropic_request={
            "model": HAIKU,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is:"},
                    {"type": "text", "text": "Lizard???"}
                ]
            }],
            "max_tokens": MAX_TOKENS,
        }
    ),
    
    TestFixture(
        name="tool_definition",
        description="Tool definition and auto mode",
        anthropic_request={
            "model": SONNET,
            "messages": WEATHER_MESSAGES,
            "max_tokens": MAX_TOKENS ,
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": WEATHER_INPUT_SCHEMA,
            }],
            "tool_choice": {"type": "auto"}
        }
    ),
    
    TestFixture(
        name="tool_use_flow",
        description="Complete tool use conversation",
        anthropic_request={
            "model": HAIKU,
            "messages": [
                {"role": "user", "content": "weather Paris"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll check."}, TOOL_USE_ANTHROPIC]
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_01234", "content": TOOL_RESULT_CONTENT}]
                },
                {"role": "assistant", "content": "22°C, sunny, 45% humidity."}
            ],
            "max_tokens": MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="thinking_block",
        description="Thinking block filtering (Anthropic-specific)",
        anthropic_request={
            "model": SONNET,
            "messages": [
                {"role": "user", "content": "Explain recursion"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "The user wants an explanation of recursion. I should provide a clear, simple explanation.",
                            "signature": "EtMFCkYIBBgCKkCUFMbgAiGkpoL7NGc5Kj7f6u3I"
                        },
                        {"type": "text", "text": "hiss"}
                    ]
                }
            ],
            "max_tokens": MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="parameters",
        description="Various parameters (temperature, top_p, stop)",
        anthropic_request={
            "model": HAIKU,
            "messages": [{"role": "user", "content": "What is lizard"}],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["\\n\\n", "END"]
        }
    ),

    TestFixture(
        name="streaming",
        description="Streaming response",
        anthropic_request={
            "model": HAIKU,
            "messages": [SIMPLE_USER_MSG],
            "max_tokens": MAX_TOKENS,
            "stream": True
        }
    ),
    TestFixture(
        name="system_blocks",
        description="System message as content blocks",
        anthropic_request={
            "model": SONNET,
            "system": [
                {"type": "text", "text": "You are a lizard."},
                {"type": "text", "text": "Write clean code."}
            ],
            "messages": [{"role": "user", "content": "sort a list"}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0
        }
    )
]
