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
