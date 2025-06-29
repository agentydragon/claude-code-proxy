"""OpenTelemetry-instrumented server handlers."""

import copy
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from opentelemetry import trace

from .config import load_config
from .converter import anthropic_to_openai_request, openai_to_anthropic_response
from .streaming_otel import stream_handler_otel
from .telemetry import get_tracer, record_request_event, set_error_status
from .tracking import tracker

config = load_config()
# TODO: centralize owner of OpenAI client
logger = logging.getLogger(__name__)


async def handle_anthropic_otel(
    anthropic_req: dict[str, Any],
    client: AsyncOpenAI,
    request_headers: Mapping[str, str] | None = None,
) -> StreamingResponse | JSONResponse:
    """Handle Anthropic Messages API requests with OpenTelemetry instrumentation."""
    request_id = str(uuid.uuid4())
    tracer = get_tracer()

    # Start span but don't use context manager yet
    span = tracer.start_span(
        "proxy_request",
        attributes={
            "proxy.request_id": request_id,
            "proxy.stream": anthropic_req.get("stream", False),
            "proxy.model": anthropic_req.get("model", "unknown"),
        },
    )

    try:
        logger.info(f"Received Anthropic request {request_id}")

        # Record Anthropic request as event
        record_request_event(
            span,
            "anthropic_request",
            {
                "request_id": request_id,
                "headers": dict(request_headers) if request_headers else {},
                "body": anthropic_req,
                "timestamp": time.time(),
            },
        )

        # Process messages for reasoning blocks
        messages = anthropic_req.get("messages", [])
        has_reasoning = any(
            isinstance(block, dict) and block.get("type") == "thinking"
            for msg in messages
            for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
        )

        # Track conversation
        conversation_id = (
            dict(request_headers).get("x-conversation-id") or str(uuid.uuid4())
            if request_headers
            else str(uuid.uuid4())
        )
        conversation_id, is_append, append_from_index = tracker.detect_append(messages, conversation_id)

        # Add conversation tracking attributes
        span.set_attributes(
            {
                "proxy.conversation_id": conversation_id,
                "proxy.is_append": is_append,
                "proxy.has_reasoning": has_reasoning,
                "proxy.message_count": len(messages),
            }
        )

        # Convert to OpenAI format
        if has_reasoning and is_append:
            reasoning_count = tracker.count_thinking_blocks(messages)
            span.set_attribute("proxy.reasoning_blocks_preserved", reasoning_count)
            anthropic_req = copy.deepcopy(anthropic_req)
            anthropic_req["messages"] = messages[append_from_index:]
        elif has_reasoning:
            reasoning_count = tracker.count_thinking_blocks(messages)
            span.set_attribute("proxy.reasoning_blocks_filtered", reasoning_count)

        openai_request = anthropic_to_openai_request(anthropic_req)

        # Record OpenAI request
        record_request_event(
            span,
            "openai_request",
            {
                "request_id": request_id,
                "body": openai_request,
                "timestamp": time.time(),
            },
        )

        # Update conversation cache
        tracker.update(
            conversation_id,
            messages,
            had_reasoning_filtered=(has_reasoning and not is_append),
            original_had_reasoning=has_reasoning,
        )

        # Handle streaming vs non-streaming
        if anthropic_req.get("stream"):
            # For streaming, we pass the span ownership to the handler
            # It will end the span when streaming is complete
            return StreamingResponse(
                stream_handler_otel(openai_request, request_id, span, client), media_type="text/event-stream"
            )

        # Non-streaming: we handle the span lifecycle here
        with trace.use_span(span, end_on_exit=True):
            # Non-streaming call
            with tracer.start_as_current_span("openai_api_call") as api_span:
                api_span.set_attributes(
                    {
                        "http.method": "POST",
                        "http.url": "https://api.openai.com/v1/responses",
                        "openai.model": openai_request.get("model", "unknown"),
                    }
                )

                # TODO: configuration in config.toml for timeout
                oai_response = await client.responses.create(**openai_request, timeout=config.openai_timeout)
                oai_dict = oai_response.model_dump()

            # Record OpenAI response
            record_request_event(
                span,
                "openai_response",
                {
                    # TODO: this should be folded into a helper / one method (DRY)
                    "request_id": request_id,
                    "status_code": 200,  # TODO: constant, doens't make sense to set
                    "body": oai_dict,
                    "timestamp": time.time(),
                },
            )

            # Convert back to Anthropic format
            anthropic_response = openai_to_anthropic_response(oai_dict)

            # Record Anthropic response
            record_request_event(
                span,
                "anthropic_response",
                {
                    "request_id": request_id,
                    "headers": {},  # TODO: constant, doesn't make sense to set
                    "status_code": 200,  # TODO: constant, doens't make sense to set
                    "body": anthropic_response,
                    "timestamp": time.time(),
                },
            )

            return JSONResponse(content=anthropic_response)

    except Exception as e:
        set_error_status(span, e)
        span.end()  # Make sure to end span on error
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e)) from e
