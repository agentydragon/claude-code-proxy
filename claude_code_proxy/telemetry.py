"""OpenTelemetry instrumentation for Claude Code Proxy."""

import json
import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode

from .telemetry_store import VisualizationSpanProcessor, span_store

logger = logging.getLogger(__name__)

# Global tracer instance
tracer: trace.Tracer | None = None


def setup_telemetry(service_name: str = "claude-code-proxy", service_version: str = "0.2.0") -> None:
    """Initialize OpenTelemetry with OTLP exporter."""
    global tracer

    # Create resource attributes
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENV", "development"),
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporters based on environment
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if otlp_endpoint:
        # Use OTLP exporter if endpoint is configured
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint, insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"OTLP exporter configured with endpoint: {otlp_endpoint}")
    else:
        # Fallback to console exporter for development
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("Using console exporter (set OTEL_EXPORTER_OTLP_ENDPOINT for OTLP)")

    # Always add visualization processor for local UI
    viz_processor = VisualizationSpanProcessor(span_store)
    provider.add_span_processor(viz_processor)
    logger.info("Visualization span processor enabled")

    # Set the tracer provider
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Auto-instrument libraries
    FastAPIInstrumentor().instrument(tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if tracer is None:
        raise RuntimeError("Telemetry not initialized. Call setup_telemetry() first.")
    return tracer


def record_request_event(span: trace.Span, event_name: str, data: dict[str, Any]) -> None:
    """Record a request/response event with all data as attributes."""
    attributes = {}

    # Flatten the data structure for attributes
    for key, value in data.items():
        if isinstance(value, str | int | float | bool):
            attributes[f"proxy.{event_name}.{key}"] = value
        elif isinstance(value, dict) and key == "headers":
            # Special handling for headers
            for header_key, header_value in value.items():
                attributes[f"proxy.{event_name}.headers.{header_key}"] = str(header_value)
        elif key == "body":
            # Store body as JSON string
            attributes[f"proxy.{event_name}.body"] = json.dumps(value, default=str)

    span.add_event(event_name, attributes=attributes)


def record_streaming_chunk(span: trace.Span, chunk_data: str, chunk_index: int) -> None:
    """Record a streaming chunk event."""
    span.add_event(
        "streaming_chunk_received",
        attributes={
            "proxy.streaming.chunk_index": chunk_index,
            "proxy.streaming.chunk_size": len(chunk_data),
            "proxy.streaming.chunk_data": chunk_data[:1000],  # Limit size
        },
    )


def set_error_status(span: trace.Span, error: Exception) -> None:
    """Set error status on span with exception details."""
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.record_exception(error)
    span.set_attribute("error.type", type(error).__name__)
