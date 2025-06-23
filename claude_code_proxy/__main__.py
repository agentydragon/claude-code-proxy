"""Entry point for claude-code-proxy."""
import sys
import argparse
import uvicorn
from .server import app, config

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claude Code Proxy - Use Anthropic clients with OpenAI/Gemini models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Config file: ~/.config/claude-code-proxy/config.toml
  Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
  
Examples:
  claude-code-proxy                    # Use config file settings
  claude-code-proxy --port 8080        # Override port
  claude-code-proxy --host 127.0.0.1   # Listen on localhost only
"""
    )
    
    parser.add_argument(
        "--host",
        default=config.host,
        help=f"Host to bind to (default: {config.host})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.port,
        help=f"Port to bind to (default: {config.port})"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=config.log_level.lower(),
        help=f"Log level (default: {config.log_level.lower()})"
    )
    
    args = parser.parse_args()
    
    print(f"Starting Claude Code Proxy on http://{args.host}:{args.port}")
    
    # Run with specified settings
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()