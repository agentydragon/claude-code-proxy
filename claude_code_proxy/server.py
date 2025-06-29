"""Claude Code Proxy Server - Direct Anthropic to OpenAI conversion."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from starlette.templating import _TemplateResponse as TemplateResponse

from .config import load_config
from .converter import anthropic_to_openai_request
from .logging_utils import log_dir, session_id
from .server_otel import handle_anthropic_otel
from .telemetry import setup_telemetry
from .telemetry_store import span_store

config = load_config()
client = AsyncOpenAI(api_key=config.openai_api_key)

log_level = logging._nameToLevel[config.log_level.upper()]
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get the package directory for static files
package_dir = Path(__file__).parent.parent

app = FastAPI()
templates_dir = package_dir / "templates"
static_dir = package_dir / "static"

# Only mount if directories exist
templates: Jinja2Templates | None = None
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    logger.warning(f"Templates directory not found: {templates_dir}")

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning(f"Static directory not found: {static_dir}")


@app.post("/v1/messages", response_model=None)  # type: ignore[misc]
async def handle_messages(request: Request) -> StreamingResponse | JSONResponse:
    """Handle Anthropic Messages API requests."""
    try:
        return await handle_anthropic_otel(await request.json(), request.headers)
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/messages/count_tokens", response_model=None)  # type: ignore[misc]
async def count_tokens(request: Request) -> JSONResponse:
    """Handle token counting requests via the real OpenAI Tokens API."""
    body = await request.json()
    oai_req = anthropic_to_openai_request(body)
    # Call the /v1/tokens/count endpoint using HTTP client
    from httpx import AsyncClient

    async with AsyncClient(headers={"Authorization": f"Bearer {config.openai_api_key}"}) as http:
        resp = await http.post("https://api.openai.com/v1/tokens/count", json=oai_req, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    count = data.get("token_count")
    return JSONResponse({"input_tokens": count})


@app.get("/", response_model=None)  # type: ignore[misc]
async def index(request: Request) -> TemplateResponse:
    """Root endpoint - flow visualization landing page."""
    if templates is None:
        raise HTTPException(status_code=500, detail="Templates directory not found")
    logs_root = log_dir.parent
    return templates.TemplateResponse(
        "index.html", {"request": request, "session": session_id, "logs_root": str(logs_root)}
    )


@app.get("/health", response_model=None)  # type: ignore[misc]
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/telemetry", response_model=None)  # type: ignore[misc]
async def telemetry_ui(request: Request) -> TemplateResponse:
    """Telemetry visualization UI."""
    if templates is None:
        raise HTTPException(status_code=500, detail="Templates directory not found")
    return templates.TemplateResponse("telemetry.html", {"request": request})


@app.get("/telemetry/traces")  # type: ignore[misc]
async def get_traces() -> JSONResponse:
    """Get recent traces with statistics."""
    traces = span_store.get_recent_traces(limit=20)

    # Calculate statistics
    total_spans = len(span_store._spans)
    total_traces = len(traces)

    durations = []
    error_count = 0

    for trace_spans in traces.values():
        for span in trace_spans:
            if span.duration_ms:
                durations.append(span.duration_ms)
            if span.status_code == "ERROR":
                error_count += 1

    avg_duration = sum(durations) / len(durations) if durations else 0
    error_rate = (error_count / total_spans * 100) if total_spans > 0 else 0

    return JSONResponse(
        {
            "traces": {trace_id: [span.to_dict() for span in spans] for trace_id, spans in traces.items()},
            "stats": {
                "total_traces": total_traces,
                "total_spans": total_spans,
                "avg_duration": avg_duration,
                "error_rate": error_rate,
            },
        }
    )


@app.get("/telemetry/trace/{trace_id}")  # type: ignore[misc]
async def get_trace(trace_id: str) -> JSONResponse:
    """Get all spans for a specific trace."""
    spans = span_store.get_trace(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")

    return JSONResponse([span.to_dict() for span in spans])


@app.post("/telemetry/clear")  # type: ignore[misc]
async def clear_telemetry() -> JSONResponse:
    """Clear all stored telemetry data."""
    span_store.clear()
    return JSONResponse({"status": "cleared"})


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


@app.on_event("startup")  # type: ignore[misc]
async def startup_event() -> None:
    """Log startup information and validate configuration."""
    # Initialize OpenTelemetry
    setup_telemetry()
    logger.info("OpenTelemetry instrumentation initialized")

    logger.info(f"Starting Claude Code Proxy - Session ID: {session_id}")
    logger.info(f"Logs directory: {log_dir}")

    # Log OTLP configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        logger.info(f"OTLP endpoint configured: {otlp_endpoint}")
    else:
        logger.info("Using console exporter for telemetry (set OTEL_EXPORTER_OTLP_ENDPOINT for OTLP)")

    # Log telemetry UI URL
    logger.info(f"📊 Telemetry visualization available at: http://{config.host}:{config.port}/telemetry")

    # Warn if no custom model mappings are configured (defaults will be used)
    if not config.anthropic_to_openai_model:
        logger.warning("No custom 'anthropic_to_openai_model' mappings found; using default mappings.")
    # Fail fast if no OpenAI API key is provided
    if not config.openai_api_key:
        logger.error("Configuration error: OPENAI_API_KEY not set; some endpoints may fail")

    # Log conversation tracker state
    from .tracking import tracker

    logger.info(f"Loaded {len(tracker.cache)} conversations from persistent cache")


@app.on_event("shutdown")  # type: ignore[misc]
async def shutdown_event() -> None:
    """Save state on shutdown."""
    from .tracking import tracker

    tracker.save()
    logger.info("Saved conversation cache on shutdown")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
