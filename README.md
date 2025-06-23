# Anthropic API Proxy for Gemini & OpenAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini or OpenAI backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini or OpenAI models via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key 🔑
- Google AI Studio (Gemini) API key (if using Google provider) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure your proxy**: 
   Set up your configuration file following the Configuration section below. You can use either:
   - A `config.toml` file (recommended) - see the Configuration section
   - Environment variables as fallback

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Configuration 🔧

### Configuration File Location

The proxy looks for `config.toml` in the following locations (in order):

1. **XDG Config Directory** (recommended):
   - Linux: `~/.config/claude-code-proxy/config.toml`
   - macOS: `~/Library/Application Support/claude-code-proxy/config.toml`
   - Windows: `%APPDATA%\claude-code-proxy\config.toml`

2. **Legacy Location**: `~/.config/claude-code-proxy/config.toml`

### Configuration Setup

1. Copy the example configuration:
   ```bash
   mkdir -p ~/.config/claude-code-proxy
   cp config.example.toml ~/.config/claude-code-proxy/config.toml
   ```

2. Edit the configuration with your API keys:
   ```bash
   $EDITOR ~/.config/claude-code-proxy/config.toml
   ```

### Configuration Options

#### API Keys
```toml
anthropic_api_key = "sk-ant-..."  # Optional if not using Anthropic fallback
openai_api_key = "sk-proj-..."    # Required for OpenAI models
gemini_api_key = "AI..."          # Required for Gemini models
```

#### Model Mapping
```toml
preferred_provider = "openai"   # Default provider: "openai" or "google"
big_model = "gpt-4o"           # Model for complex requests (sonnet, opus)
small_model = "gpt-4o-mini"    # Model for simple requests (haiku)
```

#### Server Settings
```toml
host = "0.0.0.0"    # Listen address
port = 8082         # Listen port
log_level = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR
```

#### Custom Model Mappings

Map specific model patterns to different models:

```toml
[[custom_mappings]]
pattern = "opus"              # Match "opus" in model name
target_model = "gpt-4o"       # Map to this model
target_provider = "openai"    # Use this provider

[[custom_mappings]]
pattern = "claude-3.5-sonnet"
target_model = "gpt-4o"
target_provider = "openai"
```

### Environment Variables

Environment variables can be used as fallback when not set in config:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `PREFERRED_PROVIDER`
- `BIG_MODEL`
- `SMALL_MODEL`
- `PROXY_HOST`
- `PROXY_PORT`
- `LOG_LEVEL`

### Example Configurations

#### Using OpenAI for everything:
```toml
openai_api_key = "sk-proj-..."
preferred_provider = "openai"
big_model = "gpt-4o"
small_model = "gpt-4o-mini"
```

#### Using Gemini for everything:
```toml
gemini_api_key = "AI..."
preferred_provider = "google"
big_model = "gemini-1.5-pro"
small_model = "gemini-1.5-flash"
```

#### Mixed providers with custom mappings:
```toml
openai_api_key = "sk-proj-..."
gemini_api_key = "AI..."
preferred_provider = "openai"
big_model = "gpt-4o"
small_model = "gemini-1.5-flash"  # Use Gemini for small models

[[custom_mappings]]
pattern = "claude-3-opus"
target_model = "gemini-1.5-pro"
target_provider = "google"
```

### Troubleshooting

#### Config File Errors

If your config file has syntax errors, the proxy will refuse to start with a clear error message. Common issues:

- Missing quotes around strings
- Invalid TOML syntax
- Missing required fields

#### Testing Your Config

Run the proxy with debug logging to see model mappings:

```bash
LOG_LEVEL=DEBUG python server.py
```

#### Validating TOML Syntax

You can validate your TOML file online at https://www.toml-lint.com/ or use:

```bash
python -c "import tomli; tomli.load(open('config.toml', 'rb'))"
```

## Model Mapping 🗺️

The proxy automatically maps Anthropic model names to your configured providers:

1. **Haiku models** → `small_model`
2. **Sonnet/Opus models** → `big_model`
3. **Custom mappings** take precedence over default rules

| Claude Model | Default Mapping | When BIG_MODEL/SMALL_MODEL is a Gemini model |
|--------------|--------------|---------------------------|
| haiku | openai/gpt-4o-mini | gemini/[model-name] |
| sonnet | openai/gpt-4o | gemini/[model-name] |

### Supported Models

#### OpenAI Models
The following OpenAI models are supported with automatic `openai/` prefix handling:
- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- OpenAI models get the `openai/` prefix 
- Gemini models get the `gemini/` prefix
- The BIG_MODEL and SMALL_MODEL will get the appropriate prefix based on whether they're in the OpenAI or Gemini model lists

For example:
- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- When BIG_MODEL is set to a Gemini model, Claude Sonnet will map to `gemini/[model-name]`

### Customizing Model Mapping

See the Configuration section above for detailed instructions on customizing model mappings using either:
- The `config.toml` file (recommended)
- Environment variables as fallback

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to OpenAI format via LiteLLM 🔄
3. **Sending** the translated request to OpenAI 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
