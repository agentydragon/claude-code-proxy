"""Streaming logic for OpenAI responses, including truncation helper."""

import asyncio
import json
import logging
import uuid

import httpx
from fastapi import HTTPException

from .config import load_config
from .logging_utils import truncate

config = load_config()
logger = logging.getLogger(__name__)

OPENAI_CLIENT = httpx.AsyncClient(headers={"Authorization": f"Bearer {config.openai_api_key}"})


async def stream_handler(openai_request: dict, request_id: str):
    logger.debug(f"Starting stream handler for request {request_id}")
    max_retries = 3
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            async with OPENAI_CLIENT.stream(
                "POST", "https://api.openai.com/v1/responses", json=openai_request, timeout=300.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        logger.warning(
                            f"Streaming server error {response.status_code} (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    logger.error(f"OpenAI streaming error: {error_text}")
                    raise HTTPException(status_code=response.status_code, detail=error_text.decode())

                logger.debug(f"Streaming response status: {response.status_code}")

                content_block_started = False
                message_started = False

                def _data(type, **kwargs):
                    yield f"event: {type}\n"
                    data = {"type": type, **kwargs}
                    yield f"data: {json.dumps(data)}\n\n"

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("event: "):
                        event = line.removeprefix("event: ").strip()
                        logger.debug(f"Received event: {event}")
                        if event == "response.done":
                            if content_block_started:
                                _data(type="content_block_stop", index=0)
                            _data(
                                type="message_delta",
                                delta={"stop_reason": "end_turn", "stop_sequence": None},
                                usage={"output_tokens": 0},
                            )
                            _data(type="message_stop")
                            break
                        continue

                    if not line.startswith("data: "):
                        logger.debug(f"Skipping non-data line: {line}")
                        continue

                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        yield "event: message_stop\n"
                        yield _data(type="message_stop")
                        break

                    chunk = json.loads(data)
                    logger.debug(f"Parsed chunk: {truncate(chunk)}")
                    chunk_type = chunk.get("type")

                    if chunk_type == "response.output_text.delta":
                        if text := chunk.get("delta", ""):
                            if not message_started:
                                _data(
                                    type="message_start",
                                    message={
                                        "id": f"msg_{uuid.uuid4().hex}",
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [],
                                        "model": openai_request.get("model", "unknown"),
                                        "stop_reason": None,
                                        "stop_sequence": None,
                                        "usage": {"input_tokens": 0, "output_tokens": 0},
                                    },
                                )
                                message_started = True

                            if not content_block_started:
                                _data(type="content_block_start", index=0, content_block={"type": "text", "text": ""})
                                content_block_started = True

                            _data(type="content_block_delta", index=0, delta={"type": "text_delta", "text": text})
                    elif chunk_type == "response.output.delta":
                        delta = chunk.get("delta", {})
                        if delta.get("type") == "output_text" and "text" in delta:
                            _data(
                                type="content_block_delta", index=0, delta={"type": "text_delta", "text": delta["text"]}
                            )
                        for i, tool_call in enumerate(delta.get("tool_calls", [])):
                            if not (fn := tool_call.get("function")):
                                logger.error(f"Skipping tool call without function: {tool_call}")
                                continue

                            if "name" in fn:
                                _data(
                                    type="content_block_start",
                                    index=i + 1,
                                    content_block={
                                        "type": "tool_use",
                                        "id": tool_call.get("id", f"tool_{i}"),
                                        "name": fn["name"],
                                        "input": {},
                                    },
                                )

                            if args := fn.get("arguments"):
                                _data(
                                    type="content_block_delta",
                                    index=i + 1,
                                    delta={"type": "input_json_delta", "partial_json": args},
                                )
                        return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Streaming failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Streaming failed after {max_retries} attempts: {str(e)}")
                raise
