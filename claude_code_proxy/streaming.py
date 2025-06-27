"""Streaming logic for OpenAI responses, including truncation helper."""
import asyncio
import copy
import json
import logging

import openai.responses
from fastapi import HTTPException

from .config import load_config

config = load_config()
logger = logging.getLogger(__name__)

def _trunc(x):
    """Truncate long request/response payloads for concise logging."""
    T = 10000
    x_copy = copy.deepcopy(x)
    if isinstance(x_copy, dict):
        def truncate_dict(d):
            for k, v in d.items():
                if isinstance(v, str):
                    d[k] = v[:20]
                elif isinstance(v, dict):
                    truncate_dict(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            truncate_dict(item)
        truncate_dict(x_copy)
        x_copy = json.dumps(x_copy)
    text = x_copy if not isinstance(x_copy, dict) else x_copy
    return text if len(text) <= T else text[:T] + "..."

async def stream_handler(openai_request: dict, request_id: str) -> None:
    """Stream events via the OpenAI Responses API and yield Anthropic-formatted SSE."""
    logger.debug(f"Starting stream handler for request {request_id}")
    # Use the OpenAI Responses API streaming endpoint
    try:
        async for resp_event in openai.responses.create(**openai_request):
            # resp_event is a ResponseStreamEvent; convert to SSE event(s)
            # (Insert existing mapping logic here)
            yield from _map_response_event(resp_event)
    except Exception as e:
        logger.error(f"Streaming error via OpenAI Responses API: {e}")
        raise HTTPException(status_code=500, detail=str(e))
