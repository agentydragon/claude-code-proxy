"""In-memory storage for OpenTelemetry spans for local visualization."""

from collections import deque
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor


@dataclass
class StoredEvent:
    """Stored span event with timestamp and attributes."""

    name: str
    timestamp: float
    attributes: dict[str, Any]


@dataclass
class StoredSpan:
    """Stored span data for visualization."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None
    duration_ms: float | None
    kind: str
    status_code: str
    status_message: str | None
    attributes: dict[str, Any]
    events: list[StoredEvent]
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "kind": self.kind,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [{"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes} for e in self.events],
            "context": self.context,
        }


class InMemorySpanStore:
    """Thread-safe in-memory storage for spans."""

    def __init__(self, max_spans: int = 1000):
        self.max_spans = max_spans
        self._spans: deque[StoredSpan] = deque(maxlen=max_spans)
        self._traces: dict[str, list[StoredSpan]] = {}

    def add_span(self, span: StoredSpan) -> None:
        """Add a span to the store."""
        self._spans.append(span)

        # Group by trace ID
        if span.trace_id not in self._traces:
            self._traces[span.trace_id] = []
        self._traces[span.trace_id].append(span)

        # Clean up old traces if we have too many
        if len(self._traces) > self.max_spans // 10:  # Keep ~10 spans per trace avg
            oldest_trace = min(self._traces.keys(), key=lambda tid: min(s.start_time for s in self._traces[tid]))
            del self._traces[oldest_trace]

    def get_recent_spans(self, limit: int = 100) -> list[StoredSpan]:
        """Get the most recent spans."""
        return list(self._spans)[-limit:]

    def get_trace(self, trace_id: str) -> list[StoredSpan]:
        """Get all spans for a trace."""
        return self._traces.get(trace_id, [])

    def get_recent_traces(self, limit: int = 10) -> dict[str, list[StoredSpan]]:
        """Get the most recent traces."""
        # Sort traces by most recent span start time
        sorted_traces = sorted(self._traces.items(), key=lambda kv: max(s.start_time for s in kv[1]), reverse=True)
        return dict(sorted_traces[:limit])

    def clear(self) -> None:
        """Clear all stored spans."""
        self._spans.clear()
        self._traces.clear()


class VisualizationSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Span processor that stores spans for visualization."""

    def __init__(self, store: InMemorySpanStore):
        self.store = store

    def on_start(self, span: ReadableSpan, parent_context: Any | None = None) -> None:
        """Called when a span is started."""
        pass  # We only care about completed spans

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span is ended."""
        # Extract span data
        span_context = span.get_span_context()

        # Convert events
        events = []
        for event in span.events:
            events.append(
                StoredEvent(
                    name=event.name,
                    timestamp=event.timestamp / 1e9,  # Convert to seconds
                    attributes=dict(event.attributes or {}),
                )
            )

        # Calculate duration
        duration_ms = None
        if span.end_time and span.start_time:
            duration_ms = (span.end_time - span.start_time) / 1e6  # Convert to ms

        # Create stored span
        stored_span = StoredSpan(
            span_id=format(span_context.span_id, "016x"),
            trace_id=format(span_context.trace_id, "032x"),
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            name=span.name,
            start_time=span.start_time / 1e9 if span.start_time else 0,  # Convert to seconds
            end_time=span.end_time / 1e9 if span.end_time else None,
            duration_ms=duration_ms,
            kind=span.kind.name if span.kind else "INTERNAL",
            status_code=span.status.status_code.name if span.status else "UNSET",
            status_message=span.status.description if span.status else None,
            attributes=dict(span.attributes or {}),
            events=events,
            context={
                "trace_flags": span_context.trace_flags,
                "is_remote": span_context.is_remote,
            },
        )

        # Store the span
        self.store.add_span(stored_span)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True


# Global span store instance
span_store = InMemorySpanStore()
