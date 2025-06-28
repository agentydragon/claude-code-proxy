"""Configuration management for claude-code-proxy using Pydantic and XDG."""

import os
from enum import Enum
from pathlib import Path

import platformdirs
import tomli as tomllib
from pydantic import BaseModel, Field, validator


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningSummary(str, Enum):
    AUTO = "auto"
    CONCISE = "concise"
    DETAILED = "detailed"
    NONE = "none"


class ModelMapping(BaseModel):
    """Custom model mapping rule."""

    source_anthropic_model: str
    target_openai_model: str


class ProxyConfig(BaseModel):
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


def load_config(config_file: str | None = None) -> ProxyConfig:
    """Load configuration from file, environment variables, and defaults.

    Priority order:
    1. Config file values
    2. Environment variables
    3. Default values
    """
    config_data = {}

    # Load from config file if it exists (cwd/config.toml or XDG config)
    if config_file:
        config_path = Path(config_file)
    else:
        cwd = Path.cwd()
        txt = cwd / "config.toml"
        tst = cwd / "config.test.toml"
        xdg = Path(platformdirs.user_config_dir("claude-code-proxy")) / "config.toml"
        config_path = txt if txt.exists() else tst if tst.exists() else xdg
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
            value = os.environ[env_var]
            # Convert port to int
            if config_key == "port":
                try:
                    value = int(value)
                except ValueError:
                    continue
            config_data[config_key] = value

    # Create and return config object
    return ProxyConfig(**config_data)


# mypy: ignore_errors
