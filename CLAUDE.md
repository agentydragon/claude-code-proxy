# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python-based proxy server that translates between Anthropic's Messages API and OpenAI's Chat Completion API. It allows Anthropic clients (like Claude Code) to use OpenAI models through API translation.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -e .[dev]
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_converter.py

# Run with coverage
pytest --cov

# Run specific test
pytest tests/test_converter.py::test_basic_conversion
```

### Code Quality
```bash
# Run all pre-commit hooks (includes formatting, linting, type checking)
pre-commit run --all-files

# Individual tools
ruff check .          # Linting
black .               # Code formatting
mypy .                # Type checking
```

### Running the Server
```bash
# Default
claude-code-proxy

# With options
claude-code-proxy --host 127.0.0.1 --port 8082 --log-level debug

# Alternative (module execution)
python -m claude_code_proxy
```

## Architecture Overview

### Core Components

1. **API Translation Layer** (`converter.py`)
   - Converts Anthropic Messages API requests to OpenAI format
   - Handles response translation back to Anthropic format
   - Supports streaming and non-streaming responses
   - Special handling for reasoning models (o1/o3)

2. **Server** (`server.py`)
   - FastAPI-based REST API
   - Endpoints: `/v1/messages` (main), `/health`, `/` (flow visualization)
   - Static API key authentication
   - Request/response logging in JSONL format

3. **Configuration** (`config.py`)
   - Pydantic-based configuration management
   - Hierarchy: Config file → Environment variables → Defaults
   - XDG Base Directory compliant
   - Model mapping with wildcard support

4. **Tracking** (`tracking.py`)
   - Conversation tracking and caching
   - Append request detection
   - Optimizes repeated requests by caching responses

### Key Patterns

- **Async/Await**: All request handling is asynchronous
- **Type Safety**: Strict MyPy configuration, extensive type hints
- **Structured Logging**: JSONL format for request/response tracking
- **Model Mapping**: Configurable mapping between Anthropic and OpenAI model names with wildcard patterns

### Configuration

Configuration locations (in order of precedence):
1. `./config.toml` (local override)
2. `~/.config/claude-code-proxy/config.toml` (user config)
3. Environment variables (e.g., `ANTHROPIC_API_KEY`)
4. Default values in code

### Testing Strategy

- Test fixtures in `tests/fixtures/` for converter testing
- Async test support configured
- Coverage focused on core conversion logic
- Test configuration via `config.test.toml`

## Important Notes

- Python 3.10+ required
- Pre-commit hooks are configured - they will run automatically on commit
- The server uses a static API key "sk-ant-api03-fake-key" for simplicity
- Logs are stored in XDG_STATE_HOME (typically `~/.local/state/claude-code-proxy/`)
- The web UI at root path (`/`) provides flow visualization and API documentation
