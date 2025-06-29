from openai import AsyncOpenAI

from .config import load_config

config = load_config()
# Centralized AsyncOpenAI client with configurable timeout
client = AsyncOpenAI(api_key=config.openai_api_key, timeout=config.openai_timeout)

def get_openai_client() -> AsyncOpenAI:
    return client
# Centralized AsyncOpenAI client with configurable timeout
client = AsyncOpenAI(api_key=config.openai_api_key, timeout=config.openai_timeout)
