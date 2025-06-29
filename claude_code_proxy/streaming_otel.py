"""OpenTelemetry-instrumented streaming handler."""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import HTTPException
from openai import AsyncOpenAI
from opentelemetry import trace

from .config import load_config
from .telemetry import get_tracer, record_request_event, record_streaming_chunk

config = load_config()
logger = logging.getLogger(__name__)


async def stream_handler_otel(
    openai_request: dict[str, Any], request_id: str, parent_span: trace.Span, client: AsyncOpenAI
) -> AsyncGenerator[str, None]:
    """
    Stream assistant responses with OpenTelemetry instrumentation.

    This handler takes ownership of the parent span and will end it
    when streaming is complete or on error.
    """
    logger.debug(f"Starting OpenTelemetry stream handler for request {request_id}")
    tracer = get_tracer()

    try:
        # Use the parent span context
        with trace.use_span(parent_span, end_on_exit=False):
            accumulated_lines = []
            chunk_index = 0

            # Create a child span for the streaming operation
            with tracer.start_as_current_span(
                "openai_streaming_call",
                attributes={
                    "proxy.request_id": request_id,
                    "http.method": "POST",
                    "http.url": "https://api.openai.com/v1/responses",
                    "openai.model": openai_request.get("model", "unknown"),
                    "openai.streaming": True,
                },
            ) as streaming_span:
                # TODO: configuration in config.toml for timeout
                async with client.responses.with_streaming_response.create(
                    **openai_request, timeout=config.openai_timeout
                ) as response:
                    async for line in response.iter_lines():
                        accumulated_lines.append(line)

                        # Record chunk as event on parent span (still open)
                        record_streaming_chunk(parent_span, line, chunk_index)
                        chunk_index += 1

                        yield f"{line}\n"

                # Record completion of streaming
                streaming_span.set_attribute("proxy.total_chunks", chunk_index)

            # After streaming is complete, record the full response on the parent span
            record_request_event(
                parent_span,
                "openai_response",
                {
                    "request_id": request_id,
                    "status_code": 200,
                    "body": {
                        "streaming": True,
                        "total_chunks": chunk_index,
                        "lines": accumulated_lines[:100],  # Limit stored lines
                    },
                    "timestamp": time.time(),
                },
            )

            # Also record the Anthropic response event (streaming format)
            record_request_event(
                parent_span,
                "anthropic_response",
                {
                    "request_id": request_id,
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "body": {
                        "streaming": True,
                        "total_chunks": chunk_index,
                    },
                    "timestamp": time.time(),
                },
            )

    except Exception as e:
        logger.error("OpenAI streaming error: %s", str(e))
        parent_span.record_exception(e)
        parent_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Always end the parent span when done
        parent_span.end()
