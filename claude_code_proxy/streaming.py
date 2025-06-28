"""OpenAI streaming via official openai package."""

import json
import logging

import openai
from fastapi import HTTPException

from .config import load_config

config = load_config()
logger = logging.getLogger(__name__)
openai.api_key = config.openai_api_key


async def stream_handler(openai_request: dict, request_id: str):
    """Stream assistant responses using openai.ChatCompletion.acreate."""
    logger.debug(f"Starting stream handler for request {request_id}")

    def _data(event_type: str, **kwargs: dict) -> list[str]:
        payload = {"type": event_type, **kwargs}
        return [f"event: {event_type}\n", f"data: {json.dumps(payload)}\n\n"]

    try:
        async for chunk in openai.ChatCompletion.acreate(stream=True, **openai_request):
            choice = chunk.choices[0]
            delta = choice.delta or {}
            if content := delta.get("content"):
                for line in _data("content_block_delta", index=0, delta={"type": "text_delta", "text": content}):
                    yield line
            if choice.finish_reason is not None:
                for line in _data(
                    "message_delta",
                    delta={"stop_reason": choice.finish_reason, "stop_sequence": None},
                    usage={"output_tokens": 0},
                ):
                    yield line
                for line in _data("message_stop"):
                    yield line
                return
    except Exception as e:
        logger.error("OpenAI streaming error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# mypy: ignore_errors
