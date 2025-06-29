"""Configuration management for claude-code-proxy using Pydantic and XDG."""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

import platformdirs
import tomli as tomllib
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningSummary(str, Enum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"
    NONE = "none"


class ModelMapping(BaseModel):  # type: ignore
    """Custom model mapping rule."""

    source_anthropic_model: str
    target_openai_model: str


class ProxyConfig(BaseModel):  # type: ignore
    """Main configuration for claude-code-proxy."""

    openai_api_key: str | None = None

    # Custom mappings
    anthropic_to_openai_model: dict[str, str] = Field(default_factory=dict)

    # If true, forward unknown Anthropic model names to OpenAI as-is instead of erroring
    forward_unknown_model_names: bool = Field(
        False,
        description="Forward unknown Anthropic model names to OpenAI as-is",
    )

    # Server settings
    host: str = Field("127.0.0.1", description="Host to bind to")
    port: int = Field(8082, description="Port to listen on")
    openai_timeout: float = Field(
        300.0, description="Timeout for OpenAI requests in seconds; wrong format is fatal if set"
    )
    log_level: str = Field("WARNING", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    # Reasoning model settings
    reasoning_effort: ReasoningEffort = Field(ReasoningEffort.MEDIUM, description="Reasoning effort for o1/o3 models")
    reasoning_summary: ReasoningSummary = Field(ReasoningSummary.AUTO, description="Reasoning summary type")

    # Text search-and-replace patterns applied to system, user, assistant, and thinking text
    search_replace: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of regex pattern to replacement string for text content filtering and transformation",
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @validator("openai_timeout")
    def validate_openai_timeout(cls, v):
        # Ensure provided timeout is positive and correctly formatted
        if v <= 0:
            raise ValueError("openai_timeout must be positive")
        return v


def load_config() -> ProxyConfig:
    """Load configuration from file, environment variables, and defaults."""
    config_path = Path(platformdirs.user_config_dir("claude-code-proxy")) / "config.toml"
    logger.info(f"Loading config from: {config_path}")
    config_data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

    # OpenAI API key in config takes precedence over environment variable
    if "OPENAI_API_KEY" in os.environ and "openai_api_key" not in config_data:
        config_data["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    # For other env vars, config file values take precedence.
    if "PROXY_HOST" in os.environ:
        config_data["host"] = os.environ["PROXY_HOST"]
    if "PROXY_PORT" in os.environ:
        config_data["port"] = int(os.environ["PROXY_PORT"])
    if "LOG_LEVEL" in os.environ:
        config_data["log_level"] = os.environ["LOG_LEVEL"].upper()
    if "OPENAI_TIMEOUT" in os.environ:
        config_data["openai_timeout"] = float(os.environ["OPENAI_TIMEOUT"])

    return ProxyConfig(**config_data)
