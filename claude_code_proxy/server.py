"""Claude Code Proxy Server - Direct Anthropic to OpenAI conversion."""

import asyncio
import copy
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import platformdirs
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_config
from .converter import (anthropic_to_openai_request,
                        openai_to_anthropic_response)

# Load configuration
config = load_config()

# Configure logging
log_level = logging._nameToLevel[config.log_level.upper()]
logging.basicConfig(
  level=log_level,
  format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Setup XDG-compliant logging directory with session subdirectory
session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path(platformdirs.user_state_dir("claude-code-proxy")) / "logs" / session_id
log_dir.mkdir(parents=True, exist_ok=True)

# Session-based log files
anthropic_requests_log = log_dir / "anthropic_requests.jsonl"
anthropic_responses_log = log_dir / "anthropic_responses.jsonl"
openai_requests_log = log_dir / "openai_requests.jsonl"
openai_responses_log = log_dir / "openai_responses.jsonl"

def log_jsonl(filepath: Path, data: dict):
    """Write a JSON line to a file."""
    try:
        with filepath.open('a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        logger.error(f"Failed to write to {filepath}: {e}")


app = FastAPI()

OPENAI_CLIENT= httpx.AsyncClient(
    headers={"Authorization": f"Bearer {config.openai_api_key}"}
)

def _trunc(x):
    T = 10000
    x = copy.deepcopy(x)
    if isinstance(x, dict):
        # recurively crawl, trunc all strs to 30 chars
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
        truncate_dict(x)

        x = json.dumps(x)
    if len(x) > T:
        return x[:T] + "..."
    return x

async def stream_handler(openai_request, request_id):
    logger.debug(f"Starting stream handler for request {request_id}")
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            async with OPENAI_CLIENT.stream(
                "POST",
                "https://api.openai.com/v1/responses",
                json=openai_request,
                timeout=300.0
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    
                    # Retry on 5xx errors
                    if response.status_code >= 500 and attempt < max_retries - 1:
                        logger.warning(f"Streaming server error {response.status_code} (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
                    logger.error(f"OpenAI streaming error: {error_text}")
                    raise HTTPException(status_code=response.status_code, detail=error_text.decode())
                
                logger.debug(f"Streaming response status: {response.status_code}")
                
                # Track if we've started the content block
                content_block_started = False
                message_started = False
                current_event = None


                def _data(type, **kwargs):
                    """Helper function to format data for streaming."""
                    yield f"event: {type}\n"
                    data = {"type": type, **kwargs}
                    yield f"data: {json.dumps(data)}\n\n"

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    EVENT_PREFIX = "event: "
                    if line.startswith(EVENT_PREFIX):
                        event = line.removeprefix(EVENT_PREFIX).strip()
                        logger.debug(f"Received event: {event}")
                        if event in ("response.created", "response.in_progress", "response.output_item.added"):
                            logger.debug(f"Noop event: {event}")
                        elif event == "response.output_text.delta":
                            # This is a text delta event, we'll handle it when we get the data
                            logger.debug(f"Received text delta event: {event}")
                        elif event == "response.done":
                            # Convert to Anthropic's completion events
                            if content_block_started:
                                _data(type="content_block_stop", index=0)
                            _data(
                                type="message_delta",
                                delta={
                                    "stop_reason": "end_turn",
                                    "stop_sequence": None
                                },
                                # Would need to track this
                                usage={"output_tokens": 0}
                            )
                            _data(type="message_stop")
                            break
                        # Skip other events for now
                        continue


                    DATA_PREFIX = "data: "
                    if not line.startswith(DATA_PREFIX):
                        logger.debug(f"Skipping non-data line: {line}")
                        continue

                    data = line.removeprefix(DATA_PREFIX)

                    if data == "[DONE]":
                        # Convert to Anthropic's completion event
                        yield "event: message_stop\n"
                        yield _data({
                            "type": "message_stop"
                        })
                        break

                    chunk = json.loads(data)
                    logger.debug(f"Parsed chunk: {_trunc(chunk)}")

                    # Convert OpenAI Responses API chunk to Anthropic format
                    chunk_type = chunk.get("type")
            
            # Handle different streaming event types
            if chunk_type == "response.output_text.delta":
                # Text delta event from newer API format
                if (text := chunk.get("delta", "")):
                    # Start message if not started
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
                                "usage": {"input_tokens": 0, "output_tokens": 0}
                            }
                        )
                        message_started = True

                    # Start content block if not started
                    if not content_block_started:
                        _data(
                            type="content_block_start",
                            index=0,
                            content_block={
                                "type": "text",
                                "text": ""
                            }
                        )
                        content_block_started = True

                    _data(
                        type="content_block_delta",
                        index=0,
                        delta={"type": "text_delta", "text": text}
                    )
            elif chunk_type == "response.output.delta":
                # Legacy format
                delta = chunk.get("delta", {})
                if delta.get("type") == "output_text" and "text" in delta:
                    _data(
                        type="content_block_delta",
                        index=0,
                        delta={"type": "text_delta", "text": delta["text"]}
                    )

                # Handle tool calls for legacy format
                for i, tool_call in enumerate(delta.get("tool_calls", [])):
                    if not (fn := tool_call.get("function")):
                        logger.error(f"Skipping tool call without function: {tool_call}")
                        continue

                    # Tool call start
                    if "name" in fn:
                        _data(
                            type="content_block_start",
                            index=i + 1,  # After text content
                            content_block={
                                "type": "tool_use",
                                "id": tool_call.get("id", f"tool_{i}"),
                                "name": fn["name"],
                                "input": {}
                            }
                        )

                    # Tool call arguments delta
                    if (args := fn.get("arguments")):
                        _data(
                            type="content_block_delta",
                            index=i + 1,
                            delta={"type": "input_json_delta", "partial_json": args}
                        )
                
                # Successful streaming, exit retry loop
                return
                
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Streaming failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Streaming failed after {max_retries} attempts: {str(e)}")
                raise

async def handle_anthropic(anthropic_req, request_headers=None):
    request_id = str(uuid.uuid4())
    logger.info(f"Received Anthropic request {request_id}: {_trunc(anthropic_req)}")
    
    # Log full Anthropic request
    log_jsonl(anthropic_requests_log, {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "request_id": request_id,
        "headers": dict(request_headers) if request_headers else {},
        "body": anthropic_req
    })
    
    openai_request = anthropic_to_openai_request(anthropic_req)
    logger.debug(f"Converted to OpenAI request for {request_id}")
    
    # Log OpenAI request
    log_jsonl(openai_requests_log, {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "request_id": request_id,
        "url": "https://api.openai.com/v1/responses",
        "headers": dict(OPENAI_CLIENT.headers),
        "body": openai_request
    })

    if anthropic_req.get("stream"):
        try:
            return StreamingResponse(stream_handler(openai_request, request_id), media_type="text/event-stream")
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

    # Non-streaming request with retry logic
    max_retries = 3
    retry_delay = 1.0
    response = None
    
    for attempt in range(max_retries):
        try:
            response = await OPENAI_CLIENT.post(
                "https://api.openai.com/v1/responses",
                json=openai_request,
                timeout=300.0
            )
            
            # Check if it's a retryable error (5xx)
            if response.status_code >= 500 and attempt < max_retries - 1:
                logger.warning(f"Server error {response.status_code} (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            
            break  # Success or non-retryable error, exit loop
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                raise
    
    # Log OpenAI response
    log_jsonl(openai_responses_log, {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "request_id": request_id,
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": response.text if response.status_code != 200 else response.json()
    })
    
    if response.status_code != 200:
        logger.error(f"OpenAI error: {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.text)
    openai_response = response.json()
    logger.debug(f"OpenAI response: {_trunc(openai_response)}")

    anthropic_response = openai_to_anthropic_response(openai_response)
    logger.info(f"Response converted to Anthropic: {_trunc(anthropic_response)}")
    
    # Log Anthropic response
    log_jsonl(anthropic_responses_log, {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "request_id": request_id,
        "status_code": 200,
        "headers": {},  # FastAPI will add its own headers
        "body": anthropic_response
    })
    
    return JSONResponse(content=anthropic_response)


@app.post("/v1/messages")
async def handle_messages(request: Request):
    """Handle Anthropic Messages API requests."""
    try:
        return await handle_anthropic(await request.json(), request.headers)
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Handle token counting requests."""
    # For now, return a placeholder response
    # In a real implementation, you'd use tiktoken or similar
    return JSONResponse({"input_tokens": 100})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info(f"Starting Claude Code Proxy - Session ID: {session_id}")
    logger.info(f"Logs directory: {log_dir}")
    logger.info(f"Log files:")
    logger.info(f"  - Anthropic requests: {anthropic_requests_log}")
    logger.info(f"  - Anthropic responses: {anthropic_responses_log}")
    logger.info(f"  - OpenAI requests: {openai_requests_log}")
    logger.info(f"  - OpenAI responses: {openai_responses_log}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
