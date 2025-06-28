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

    # Server settings
    host: str = Field("127.0.0.1", description="Host to bind to")
    port: int = Field(8082, description="Port to listen on")
    log_level: str = Field("WARNING", description="Logging level: DEBUG, INFO, WARNING, ERROR")

    # Reasoning model settings
    reasoning_effort: ReasoningEffort = Field(ReasoningEffort.MEDIUM, description="Reasoning effort for o1/o3 models")
    reasoning_summary: ReasoningSummary = Field(ReasoningSummary.AUTO, description="Reasoning summary type")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


def find_config_path(config_file: str | None = None) -> Path:
    """Determine path to config.toml (cwd/config.toml, cwd/config.test.toml, or XDG config)."""
    if config_file:
        return Path(config_file)
    cwd = Path.cwd()
    txt = cwd / "config.toml"
    tst = cwd / "config.test.toml"
    xdg = Path(platformdirs.user_config_dir("claude-code-proxy")) / "config.toml"
    chosen = txt if txt.exists() else tst if tst.exists() else xdg
    logger.info(f"Candidate config paths: cwd config: {txt}, test config: {tst}, xdg config: {xdg}. Using: {chosen}")
    return chosen


def load_config(config_file: str | None = None) -> ProxyConfig:
    """Load configuration from file, environment variables, and defaults.

    Priority order:
    1. Config file values
    2. Environment variables
    3. Default values
    """
    config_data: dict[str, Any] = {}
    # Determine config file path and load if exists
    config_path = find_config_path(config_file)
    logger.info(f"Loading config from: {config_path}")
    if config_path.exists():
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

    # Override with environment variables (only if not set in config file)
    env_mapping = {
        "OPENAI_API_KEY": "openai_api_key",
        "PROXY_HOST": "host",
        "PROXY_PORT": "port",
        "LOG_LEVEL": "log_level",
    }
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ and config_key not in config_data:
            value: str | int = os.environ[env_var]
            if config_key == "port":
                try:
                    value = int(value)
                except ValueError:
                    continue
            config_data[config_key] = value

    return ProxyConfig(**config_data)
