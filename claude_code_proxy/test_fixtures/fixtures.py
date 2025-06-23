"""Test fixtures for Anthropic <-> OpenAI conversion testing."""

from typing import Any, Dict, List


class TestFixture:
    """A test fixture with requests for both APIs."""
    
    def __init__(self, name: str, description: str, 
                 anthropic_request: Dict[str, Any],
                 openai_request: Dict[str, Any]):
        self.name = name
        self.description = description
        self.anthropic_request = anthropic_request
        self.openai_request = openai_request


# Models
ANTHROPIC_HAIKU = "claude-3-5-haiku-20241022"
ANTHROPIC_SONNET = "claude-3-5-sonnet-20241022"
OPENAI_O3 = "o3"
OPENAI_O4_MINI = "o4-mini"

# Common parameters
DEFAULT_MAX_TOKENS = 1024
O_SERIES_TEMP = 1.0  # Required for O-series models

# Common messages
SIMPLE_USER_MSG = {"role": "user", "content": "Hello, how are you?"}
MATH_USER_MSG = {"role": "user", "content": "What's 2+2?"}
MATH_ASSISTANT_MSG = {"role": "assistant", "content": "2+2 equals 4."}
MATH_FOLLOWUP_MSG = {"role": "user", "content": "What about 3+3?"}

# Common tools
WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit"
            }
        },
        "required": ["location"]
    }
}

WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": WEATHER_TOOL_ANTHROPIC["input_schema"]
    }
}

# Tool use content blocks
TOOL_USE_ANTHROPIC = {
    "type": "tool_use",
    "id": "toolu_01234",
    "name": "get_weather",
    "input": {"location": "Paris, France", "unit": "celsius"}
}

TOOL_CALL_OPENAI = {
    "id": "call_01234",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "Paris, France", "unit": "celsius"}'
    }
}

TOOL_RESULT_CONTENT = "Temperature: 22°C, Conditions: Sunny, Humidity: 45%"

# Test fixtures focusing on different behaviors
FIXTURES: List[TestFixture] = [
    TestFixture(
        name="simple_text",
        description="Basic text exchange",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [SIMPLE_USER_MSG],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [SIMPLE_USER_MSG],
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="with_system",
        description="System prompt handling",
        anthropic_request={
            "model": ANTHROPIC_SONNET,
            "system": "You are a helpful assistant who speaks like a pirate.",
            "messages": [{"role": "user", "content": "Tell me about Python"}],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O3,
            "input": [
                {"role": "system", "content": "You are a helpful assistant who speaks like a pirate."},
                {"role": "user", "content": "Tell me about Python"}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": O_SERIES_TEMP
        }
    ),
    
    TestFixture(
        name="multi_turn",
        description="Multi-turn conversation",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [MATH_USER_MSG, MATH_ASSISTANT_MSG, MATH_FOLLOWUP_MSG],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [MATH_USER_MSG, MATH_ASSISTANT_MSG, MATH_FOLLOWUP_MSG],
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="structured_content",
        description="Structured content blocks",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this code:"},
                    {"type": "text", "text": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"}
                ]
            }],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [{
                "role": "user",
                "content": "Analyze this code:\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
            }],
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="tool_definition",
        description="Tool definition and auto mode",
        anthropic_request={
            "model": ANTHROPIC_SONNET,
            "messages": [{"role": "user", "content": "What's the weather in New York?"}],
            "max_tokens": DEFAULT_MAX_TOKENS * 2,
            "tools": [WEATHER_TOOL_ANTHROPIC],
            "tool_choice": {"type": "auto"}
        },
        openai_request={
            "model": OPENAI_O3,
            "input": [{"role": "user", "content": "What's the weather in New York?"}],
            "max_tokens": DEFAULT_MAX_TOKENS * 2,
            "temperature": O_SERIES_TEMP,
            "tools": [WEATHER_TOOL_OPENAI],
            "tool_choice": "auto"
        }
    ),
    
    TestFixture(
        name="tool_use_flow",
        description="Complete tool use conversation",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll check the weather in Paris for you."},
                        TOOL_USE_ANTHROPIC
                    ]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_01234",
                        "content": TOOL_RESULT_CONTENT
                    }]
                },
                {"role": "assistant", "content": "Based on the weather data, Paris is currently enjoying beautiful weather at 22°C with sunny conditions and 45% humidity."}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "I'll check the weather in Paris for you.",
                    "tool_calls": [TOOL_CALL_OPENAI]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_01234",
                    "content": TOOL_RESULT_CONTENT
                },
                {"role": "assistant", "content": "Based on the weather data, Paris is currently enjoying beautiful weather at 22°C with sunny conditions and 45% humidity."}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    ),
    
    TestFixture(
        name="thinking_block",
        description="Thinking block filtering (Anthropic-specific)",
        anthropic_request={
            "model": ANTHROPIC_SONNET,
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
                        {"type": "text", "text": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems."}
                    ]
                }
            ],
            "max_tokens": DEFAULT_MAX_TOKENS
        },
        openai_request={
            "model": OPENAI_O3,
            "input": [
                {"role": "user", "content": "Explain recursion"},
                {"role": "assistant", "content": "Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller subproblems."}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": O_SERIES_TEMP
        }
    ),
    
    TestFixture(
        name="parameters",
        description="Various parameters (temperature, top_p, stop)",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [{"role": "user", "content": "Write a haiku about coding"}],
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["\\n\\n", "END"]
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [{"role": "user", "content": "Write a haiku about coding"}],
            "max_tokens": 200,
            "temperature": O_SERIES_TEMP,  # Override for O-series
            "top_p": 0.9,
            "stop": ["\\n\\n", "END"]
        }
    ),
    
    TestFixture(
        name="streaming",
        description="Streaming response",
        anthropic_request={
            "model": ANTHROPIC_HAIKU,
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": True
        },
        openai_request={
            "model": OPENAI_O4_MINI,
            "input": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": True
        }
    ),
    
    TestFixture(
        name="system_blocks",
        description="System message as content blocks",
        anthropic_request={
            "model": ANTHROPIC_SONNET,
            "system": [
                {"type": "text", "text": "You are an expert Python developer."},
                {"type": "text", "text": "Always write clean, well-documented code."}
            ],
            "messages": [{"role": "user", "content": "Write a function to sort a list"}],
            "max_tokens": DEFAULT_MAX_TOKENS * 2,
            "temperature": 0.0
        },
        openai_request={
            "model": OPENAI_O3,
            "input": [
                {"role": "system", "content": "You are an expert Python developer.\nAlways write clean, well-documented code."},
                {"role": "user", "content": "Write a function to sort a list"}
            ],
            "max_tokens": DEFAULT_MAX_TOKENS * 2,
            "temperature": O_SERIES_TEMP  # Override for O-series
        }
    )
]
