"""OpenAI streaming via official openai package."""

import logging

from fastapi import HTTPException
from openai import AsyncOpenAI

from .config import load_config

config = load_config()
logger = logging.getLogger(__name__)


async def stream_handler(openai_request: dict, request_id: str):
    """
    Stream assistant responses via OpenAI Python SDK's `responses.with_streaming_response`.
    See references/openai-api-docs/python-sdk-readme.md#with_streaming_response for details.
    """
    logger.debug(f"Starting stream handler for request {request_id}")

    try:
        client = AsyncOpenAI(api_key=config.openai_api_key)
        async with client.responses.with_streaming_response.create(**openai_request) as response:
            async for line in response.iter_lines():
                yield f"{line}\n"
    except Exception as e:
        logger.error("OpenAI streaming error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# mypy: ignore_errors
