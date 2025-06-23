"""Configuration management for claude-code-proxy using Pydantic and XDG."""
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
# Platform directory utilities for config
import platformdirs
from dotenv import load_dotenv

load_dotenv()

# Use tomllib (Python 3.11+) or tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
import tomli_w


class ModelMapping(BaseModel):
    """Custom model mapping rule."""
    pattern: str = Field(..., description="Pattern to match in model name (case-insensitive)")
    target_model: str = Field(..., description="Model to map to")
    target_provider: Optional[str] = Field(None, description="Provider to use (openai, gemini, anthropic)")


class ProxyConfig(BaseModel):
    """Main configuration for claude-code-proxy."""
    
    # API Keys
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key (only needed if proxying TO Anthropic)")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    gemini_api_key: Optional[str] = Field(None, description="Google AI Studio (Gemini) API key")
    
    # Provider settings
    preferred_provider: str = Field("openai", description="Preferred provider: openai or google")
    big_model: str = Field("gpt-4o", description="Model to use for large/complex requests")
    small_model: str = Field("gpt-4o-mini", description="Model to use for small/simple requests")
    
    # Model lists (if not provided, use hardcoded defaults)
    openai_models: Optional[List[str]] = Field(None, description="List of available OpenAI models")
    gemini_models: Optional[List[str]] = Field(None, description="List of available Gemini models")
    
    # Custom mappings
    custom_mappings: List[ModelMapping] = Field(default_factory=list, description="Custom model mappings")
    
    # Server settings
    host: str = Field("0.0.0.0", description="Host to bind to")
    port: int = Field(8082, description="Port to listen on")
    log_level: str = Field("WARNING", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    
    @validator('preferred_provider')
    def validate_provider(cls, v):
        if v.lower() not in ['openai', 'google']:
            raise ValueError("preferred_provider must be 'openai' or 'google'")
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


def get_config_path(config_file: Optional[str] = None) -> Path:
    """Get the path to the config file using XDG standards."""
    if config_file:
        return Path(config_file)

    # Test harness override: if a config.test.toml exists in cwd, use it for tests
    cwd_test_cfg = Path("config.test.toml")
    if cwd_test_cfg.is_file():
        return cwd_test_cfg

    # If a test config file exists in the project root, use it for tests
    test_config = Path("config.test.toml")
    if test_config.exists():
        return test_config

    # Use XDG config directory
    xdg_config = Path(platformdirs.user_config_dir("claude-code-proxy"))
    config_path = xdg_config / "config.toml"
    
    # Check legacy location
    if not config_path.exists():
        legacy_path = Path.home() / ".config" / "claude-code-proxy" / "config.toml"
        if legacy_path.exists():
            return legacy_path
    
    return config_path


def load_config(config_file: Optional[str] = None) -> ProxyConfig:
    """Load configuration from file, environment variables, and defaults.
    
    Priority order:
    1. Config file values
    2. Environment variables
    3. Default values
    """
    config_data = {}
    
    # Load from config file if it exists
    config_path = get_config_path(config_file)
    if config_path.exists():
        try:
            with open(config_path, 'rb') as f:
                config_data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            # TOML syntax errors are fatal
            raise RuntimeError(f"TOML syntax error in {config_path}: {e}") from e
        except Exception as e:
            # Other errors (permissions, etc) are also fatal if file exists
            raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e
    
    # Override with environment variables (only if not set in config file)
    env_mapping = {
        'ANTHROPIC_API_KEY': 'anthropic_api_key',
        'OPENAI_API_KEY': 'openai_api_key',
        'GEMINI_API_KEY': 'gemini_api_key',
        'PREFERRED_PROVIDER': 'preferred_provider',
        'BIG_MODEL': 'big_model',
        'SMALL_MODEL': 'small_model',
        'PROXY_HOST': 'host',
        'PROXY_PORT': 'port',
        'LOG_LEVEL': 'log_level',
    }
    
    for env_var, config_key in env_mapping.items():
        if env_var in os.environ and config_key not in config_data:
            value = os.environ[env_var]
            # Convert port to int
            if config_key == 'port':
                try:
                    value = int(value)
                except ValueError:
                    continue
            config_data[config_key] = value
    
    # Create and return config object
    return ProxyConfig(**config_data)


def save_example_config(output_path: Optional[str] = None):
    """Save an example configuration file."""
    example = {
        "anthropic_api_key": "sk-ant-...",
        "openai_api_key": "sk-proj-...",
        "gemini_api_key": "AI...",
        "preferred_provider": "openai",
        "big_model": "gpt-4o",
        "small_model": "gpt-4o-mini",
        "openai_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3-mini"
        ],
        "gemini_models": [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "custom_mappings": [
            {
                "pattern": "opus",
                "target_model": "gpt-4o",
                "target_provider": "openai"
            },
            {
                "pattern": "claude-3",
                "target_model": "gpt-4-turbo",
                "target_provider": "openai"
            }
        ],
        "host": "0.0.0.0",
        "port": 8082,
        "log_level": "INFO"
    }
    
    output = Path(output_path) if output_path else Path("config.example.toml")
    with open(output, 'wb') as f:
        tomli_w.dump(example, f)
    
    print(f"Example config saved to {output}")


# Default model lists (fallback if not provided in config)
DEFAULT_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
]

DEFAULT_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro"
]