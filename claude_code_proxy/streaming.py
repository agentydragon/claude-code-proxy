"""OpenAI streaming via official openai package."""

import logging
import time

from fastapi import HTTPException
from openai import AsyncOpenAI

from .config import load_config
from .logging_utils import log_jsonl, openai_responses_log

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
        accumulated_response = []

        async with client.responses.with_streaming_response.create(**openai_request) as response:
            async for line in response.iter_lines():
                accumulated_response.append(line)
                yield f"{line}\n"

        # Log the complete streaming response after it's done
        log_jsonl(
            openai_responses_log,
            {
                "request_id": request_id,
                "status_code": 200,
                "body": {"streaming_lines": accumulated_response},
                "timestamp": time.time(),
            },
        )
    except Exception as e:
        logger.error("OpenAI streaming error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# mypy: ignore_errors
