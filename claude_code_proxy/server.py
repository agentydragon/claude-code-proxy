"""Claude Code Proxy Server - Direct Anthropic to OpenAI conversion."""

import copy
import json
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import tiktoken
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from starlette.templating import _TemplateResponse as TemplateResponse

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
client = AsyncOpenAI(api_key=config.openai_api_key)

log_level = logging._nameToLevel[config.log_level.upper()]
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


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

    # Call OpenAI Responses API
    if anthropic_req.get("stream"):
        try:
            from .streaming import stream_handler

            return StreamingResponse(stream_handler(openai_request, request_id), media_type="text/event-stream")
        except Exception as e:
            logger.exception("Streaming error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Non-streaming call via OpenAI client
    try:
        oai_response = await client.responses.create(**openai_request)
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

    oai_dict = oai_response.model_dump()
    logger.debug(f"OpenAI response: {truncate(oai_dict)}")
    anthropic_response = openai_to_anthropic_response(oai_dict)
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
    """Handle token counting requests by computing token usage via tiktoken."""
    body = await request.json()
    oai_req = anthropic_to_openai_request(body)
    model = oai_req.get("model")
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    instr = oai_req.get("instructions")
    if isinstance(instr, str):
        total_tokens += len(enc.encode(instr))

    for item in oai_req.get("input", []):
        cont = item.get("content")
        if isinstance(cont, str) and cont:
            total_tokens += len(enc.encode(cont))

    return JSONResponse({"input_tokens": total_tokens})


@app.get("/", response_model=None)  # type: ignore[misc]
async def index(request: Request) -> TemplateResponse:
    """Root endpoint - flow visualization landing page."""
    logs_root = log_dir.parent
    return templates.TemplateResponse(
        "index.html", {"request": request, "session": session_id, "logs_root": str(logs_root)}
    )


@app.get("/health", response_model=None)  # type: ignore[misc]
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/data")  # type: ignore[misc]
async def flow_data(session: str | None = None) -> JSONResponse:
    """Return JSON of request-response flows for the current session."""
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
