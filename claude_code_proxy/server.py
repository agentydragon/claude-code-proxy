"""Claude Code Proxy Server - Direct Anthropic to OpenAI conversion."""

import copy
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpx import AsyncClient
from starlette.templating import _TemplateResponse as TemplateResponse
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from .config import load_config
from .converter import anthropic_to_openai_request, openai_to_anthropic_response
from .logging_utils import (
    anthropic_requests_log,
    anthropic_responses_log,
    conversation_tracking_log,
    log_dir,
    log_jsonl,
    openai_requests_log,
    openai_responses_log,
    session_id,
    truncate,
)
from .tracking import tracker

config = load_config()
OPENAI_CLIENT = AsyncClient(headers={"Authorization": f"Bearer {config.openai_api_key}"})

log_level = logging._nameToLevel[config.log_level.upper()]
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@retry(  # type: ignore[misc]
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def stream_handler(openai_request: dict[str, Any], request_id: str) -> AsyncGenerator[str, None]:
    logger.debug(f"Starting stream handler for request {request_id}")

    async with OPENAI_CLIENT.stream(
        "POST",
        "https://api.openai.com/v1/responses",
        json=openai_request,
        timeout=300.0,
    ) as response:
        if response.status_code != 200:
            err = (await response.aread()).decode(errors="ignore")
            logger.error("OpenAI streaming error: %s", err)
            raise HTTPException(status_code=response.status_code, detail=err)

        logger.debug(f"Streaming response status: {response.status_code}")

        content_started = False
        message_started = False

        def _data(event_type: str, **kwargs: Any) -> list[str]:
            body = {"type": event_type, **kwargs}
            return [f"event: {event_type}\n", f"data: {json.dumps(body)}\n\n"]

        async for raw in response.aiter_lines():
            line = raw.strip()
            if not line:
                continue

            if line.startswith("event: "):
                ev = line.removeprefix("event: ").strip()
                if ev == "response.done":
                    if content_started:
                        for o in _data("content_block_stop", index=0):
                            yield o
                    for o in _data(
                        "message_delta",
                        delta={"stop_reason": "end_turn", "stop_sequence": None},
                        usage={"output_tokens": 0},
                    ):
                        yield o
                    for o in _data("message_stop"):
                        yield o
                    return
                continue

            if not line.startswith("data: "):
                continue

            data = line.removeprefix("data: ")
            if data == "[DONE]":
                for o in _data("message_stop"):
                    yield o
                return

            chunk = json.loads(data)
            logger.debug(f"Parsed chunk: {truncate(chunk)}")
            break

        typ = chunk.get("type")
        if typ == "response.output_text.delta" and (txt := chunk.get("delta", "")):
            if not message_started:
                for o in _data(
                    "message_start",
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
                ):
                    yield o
                message_started = True

            if not content_started:
                for o in _data(
                    "content_block_start",
                    index=0,
                    content_block={"type": "text", "text": ""},
                ):
                    yield o
                content_started = True

            for o in _data("content_block_delta", index=0, delta={"type": "text_delta", "text": txt}):
                yield o
        elif typ == "response.output.delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "output_text" and "text" in delta:
                for o in _data("content_block_delta", index=0, delta={"type": "text_delta", "text": delta["text"]}):
                    yield o
            for i, call in enumerate(delta.get("tool_calls", [])):
                fn = call.get("function") or {}
                if nm := fn.get("name"):
                    for o in _data(
                        "content_block_start",
                        index=i + 1,
                        content_block={
                            "type": "tool_use",
                            "id": call.get("id", f"tool_{i}"),
                            "name": nm,
                            "input": {},
                        },
                    ):
                        yield o
                if args := fn.get("arguments"):
                    for o in _data(
                        "content_block_delta",
                        index=i + 1,
                        delta={"type": "input_json_delta", "partial_json": args},
                    ):
                        yield o


async def handle_anthropic(
    anthropic_req: dict[str, Any], request_headers: Mapping[str, str] | None = None
) -> StreamingResponse | JSONResponse:
    request_id = str(uuid.uuid4())
    logger.info(f"Received Anthropic request {request_id}: {truncate(anthropic_req)}")

    # Log full Anthropic request
    log_jsonl(
        anthropic_requests_log,
        {
            "request_id": request_id,
            "headers": dict(request_headers) if request_headers else {},
            "body": anthropic_req,
            "timestamp": time.time(),
        },
    )

    messages = anthropic_req.get("messages", [])
    has_reasoning = any(
        isinstance(block, dict) and block.get("type") == "thinking"
        for msg in messages
        for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    conversation_id = (
        dict(request_headers).get("x-conversation-id") or str(uuid.uuid4()) if request_headers else str(uuid.uuid4())
    )

    conversation_id, is_append, append_from_index = tracker.detect_append(messages, conversation_id)

    previous_had_reasoning_filtered = (
        tracker.cache.get(conversation_id, {}).get("had_reasoning_filtered", False) if is_append else False
    )

    # Log conversation tracking info
    action = (
        "append_api_with_reasoning"
        if has_reasoning and is_append
        else "filtering_reasoning_new_conversation" if has_reasoning else "passthrough"
    )

    log_jsonl(
        conversation_tracking_log,
        {
            "request_id": request_id,
            "conversation_id": conversation_id,
            "is_append_candidate": is_append,
            "has_reasoning": has_reasoning,
            "previous_had_reasoning_filtered": previous_had_reasoning_filtered,
            "message_count": len(messages),
            "append_from_index": append_from_index if is_append else -1,
            "action": action,
        },
    )

    if has_reasoning and is_append:
        reasoning_count = tracker.count_thinking_blocks(messages)
        if previous_had_reasoning_filtered:
            logger.info(
                f"[REASONING PRESERVED] Request {request_id} appending to filtered conversation "
                f"preserving {reasoning_count} reasoning blocks in new messages"
            )
        else:
            logger.info(
                f"[REASONING PRESERVED] Request {request_id} has reasoning and can use append API "
                f"preserving {reasoning_count} reasoning blocks"
            )
        # For append scenario, we can keep the reasoning blocks
        # Example flow:
        # Original: u[r] a u[r] a u[r] a -> Filtered: u a u a u a
        # Append:   u a u a u a | u[r] a -> OK! New messages can have reasoning
        # Only need to send the new messages (from append_from_index onwards)
        anthropic_req_for_append = copy.deepcopy(anthropic_req)
        anthropic_req_for_append["messages"] = messages[append_from_index:]
        openai_request = anthropic_to_openai_request(anthropic_req_for_append)
        # Note: We're sending only new messages but OpenAI doesn't have explicit
        # append API
    elif has_reasoning:
        reasoning_count = tracker.count_thinking_blocks(messages)
        logger.warning(
            f"[REASONING FILTERED] Request {request_id} has {reasoning_count} reasoning blocks"
            " that will be filtered out (new conversation)"
        )
        # Need to filter out reasoning blocks since this is a new conversation
        openai_request = anthropic_to_openai_request(anthropic_req)
    else:
        # No reasoning blocks, convert normally
        openai_request = anthropic_to_openai_request(anthropic_req)
    logger.debug(f"Converted to OpenAI request for {request_id}")

    # Update conversation cache with full message history
    # Update conversation cache and index
    tracker.update(
        conversation_id,
        messages,
        had_reasoning_filtered=(has_reasoning and not is_append),
        original_had_reasoning=has_reasoning,
    )

    # Log OpenAI request
    log_jsonl(
        openai_requests_log,
        {
            "request_id": request_id,
            "url": "https://api.openai.com/v1/responses",
            "headers": dict(OPENAI_CLIENT.headers),
            "body": openai_request,
            "timestamp": time.time(),
        },
    )

    if anthropic_req.get("stream"):
        try:
            return StreamingResponse(stream_handler(openai_request, request_id), media_type="text/event-stream")
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming error: {e}") from e

    # Non-streaming request with retry logic
    @retry(  # type: ignore[misc]
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def post_with_retry(req: dict[str, Any]) -> Any:
        resp = await OPENAI_CLIENT.post("https://api.openai.com/v1/responses", json=req, timeout=300.0)
        if resp.status_code >= 500:
            logger.warning(f"Server error {resp.status_code}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp

    response = await post_with_retry(openai_request)
    # Log OpenAI response
    log_jsonl(
        openai_responses_log,
        {
            "request_id": request_id,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text if response.status_code != 200 else response.json(),
            "timestamp": time.time(),
        },
    )

    if response.status_code != 200:
        logger.error(f"OpenAI error: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    openai_response = response.json()
    logger.debug(f"OpenAI response: {truncate(openai_response)}")

    anthropic_response = openai_to_anthropic_response(openai_response)
    logger.info(f"Response converted to Anthropic: {truncate(anthropic_response)}")

    # Log Anthropic response
    log_jsonl(
        anthropic_responses_log,
        {
            "request_id": request_id,
            "status_code": 200,
            "headers": {},  # FastAPI will add its own headers
            "body": anthropic_response,
        },
    )

    return JSONResponse(content=anthropic_response)


@app.post("/v1/messages", response_model=None)  # type: ignore[misc]
async def handle_messages(request: Request) -> StreamingResponse | JSONResponse:
    """Handle Anthropic Messages API requests."""
    try:
        return await handle_anthropic(await request.json(), request.headers)
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/messages/count_tokens", response_model=None)  # type: ignore[misc]
async def count_tokens(request: Request) -> JSONResponse:
    """Handle token counting requests."""
    return JSONResponse({"input_tokens": 100})


@app.get("/health", response_model=None)  # type: ignore[misc]
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/flows", response_model=None)  # type: ignore[misc]
async def flows_page(request: Request, session: str | None = None) -> TemplateResponse:
    """Render flow visualization page."""
    logs_root = log_dir.parent
    sess = session or session_id
    return templates.TemplateResponse(
        "flows.html",
        {"request": request, "session": sess, "logs_root": str(logs_root)},
    )


@app.get("/flows/data")  # type: ignore[misc]
async def flows_data(session: str | None = None) -> JSONResponse:
    """Return JSON of request-response flows for a session."""
    from pathlib import Path

    logs_root = log_dir.parent
    sess = session or session_id
    dpath = Path(logs_root) / sess
    feeds: dict[str, dict[str, Any]] = {}

    def load(name: str) -> list[dict[str, Any]]:
        fpath = dpath / name
        if not fpath.exists():
            return []
        with fpath.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    anth_req = load("anthropic_requests.jsonl")
    oai_req = load("openai_requests.jsonl")
    oai_resp = load("openai_responses.jsonl")
    anth_resp = load("anthropic_responses.jsonl")

    for entry in anth_req:
        feeds.setdefault(str(entry.get("request_id")), {})["anthropic_request"] = entry
    for entry in oai_req:
        feeds.setdefault(str(entry.get("request_id")), {})["openai_request"] = entry
    for entry in oai_resp:
        feeds.setdefault(str(entry.get("request_id")), {})["openai_response"] = entry
    for entry in anth_resp:
        feeds.setdefault(str(entry.get("request_id")), {})["anthropic_response"] = entry

    # sort by anthropic_request timestamp
    flows = sorted(feeds.values(), key=lambda x: x.get("anthropic_request", {}).get("timestamp", 0))
    return JSONResponse({"flows": flows})


@app.on_event("startup")  # type: ignore[misc]
async def startup_event() -> None:
    """Log startup information and validate configuration."""
    logger.info(f"Starting Claude Code Proxy - Session ID: {session_id}")
    logger.info(f"Logs directory: {log_dir}")
    logger.info("Log files:")
    logger.info(f"  - Anthropic requests: {anthropic_requests_log}")
    logger.info(f"  - Anthropic responses: {anthropic_responses_log}")
    logger.info(f"  - OpenAI requests: {openai_requests_log}")
    logger.info(f"  - OpenAI responses: {openai_responses_log}")
    logger.info(f"  - Conversation tracking: {conversation_tracking_log}")

    # Warn if no custom model mappings are configured (defaults will be used)
    if not config.anthropic_to_openai_model:
        logger.warning("No custom 'anthropic_to_openai_model' mappings found; using default mappings.")
    # Fail fast if no OpenAI API key is provided
    if not config.openai_api_key:
        logger.error("Configuration error: OPENAI_API_KEY not set; some endpoints may fail")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
