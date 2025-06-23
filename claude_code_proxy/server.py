"""Claude Code Proxy Server - Direct Anthropic to OpenAI conversion."""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_config
from .converter_v2 import (anthropic_to_openai_request,
                           openai_to_anthropic_response)

# Load configuration
config = load_config()

# Configure logging
log_level = getattr(logging, config.log_level.upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Get API keys from config
OPENAI_API_KEY = config.openai_api_key

# O-series models that require temperature=1
O_SERIES_MODELS = ['o1', 'o1-mini', 'o1-preview', 'o3', 'o3-mini', 'o4-mini']


async def call_openai_responses_api_streaming(request_body: Dict[str, Any]):
    """Call OpenAI Responses API with streaming."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Ensure stream is True
    request_body["stream"] = True
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=request_body,
            timeout=300.0
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"OpenAI error: {error_text}")
                raise HTTPException(status_code=response.status_code, detail=error_text.decode())
            
            # Yield response for streaming
            async for line in response.aiter_lines():
                yield line


async def call_openai_responses_api(request_body: Dict[str, Any]) -> Dict[str, Any]:
    """Call OpenAI Responses API without streaming."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Ensure stream is False
    request_body["stream"] = False
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=request_body,
            timeout=300.0
        )
        
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"OpenAI error: {error_text}")
            raise HTTPException(status_code=response.status_code, detail=error_text)
        
        return response.json()


@app.post("/v1/messages")
async def handle_messages(request: Request):
    """Handle Anthropic Messages API requests."""
    try:
        # Parse request body
        body = await request.json()
        logger.info(f"Received Anthropic request for model: {body.get('model')}")
        
        # Convert to OpenAI format
        openai_request = anthropic_to_openai_request(body)
        
        # Map model names
        model = openai_request["model"]
        if "haiku" in model.lower():
            openai_request["model"] = config.small_model
        elif "opus" in model.lower() or "sonnet" in model.lower():
            openai_request["model"] = config.big_model
        else:
            # Use the model as-is if it's already an OpenAI model
            pass
        
        # Handle O-series temperature override
        model_name = openai_request["model"]
        if model_name in O_SERIES_MODELS and openai_request.get("temperature", 1.0) != 1.0:
            logger.info(f"Overriding temperature to 1.0 for O-series model: {model_name}")
            openai_request["temperature"] = 1.0
        
        logger.info(f"Mapped to OpenAI model: {openai_request['model']}")
        logger.debug(f"OpenAI request: {json.dumps(openai_request, default=str)[:500]}...")
        
        if body.get("stream"):
            # Streaming request
            async def stream_handler():
                async for line in call_openai_responses_api_streaming(openai_request):
                    # Convert streaming response
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        if data == "[DONE]":
                            # Convert to Anthropic's completion event
                            yield "event: message_stop\n"
                            yield "data: {\"type\": \"message_stop\"}\n\n"
                            break
                        
                        try:
                            chunk = json.loads(data)
                            
                            # Convert OpenAI Responses API chunk to Anthropic format
                            # The Responses API uses a different streaming format
                            if chunk.get("type") == "response.output.delta":
                                delta = chunk.get("delta", {})
                                
                                # Handle text output delta
                                if delta.get("type") == "output_text" and "text" in delta:
                                    anthropic_event = {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": delta["text"]
                                        }
                                    }
                                    yield f"event: content_block_delta\n"
                                    yield f"data: {json.dumps(anthropic_event)}\n\n"
                                
                                # Handle tool calls
                                if "tool_calls" in delta:
                                    for i, tool_call in enumerate(delta["tool_calls"]):
                                        if "function" in tool_call:
                                            # Tool call start
                                            if "name" in tool_call["function"]:
                                                anthropic_event = {
                                                    "type": "content_block_start",
                                                    "index": i + 1,  # After text content
                                                    "content_block": {
                                                        "type": "tool_use",
                                                        "id": tool_call.get("id", f"tool_{i}"),
                                                        "name": tool_call["function"]["name"],
                                                        "input": {}
                                                    }
                                                }
                                                yield f"event: content_block_start\n"
                                                yield f"data: {json.dumps(anthropic_event)}\n\n"
                                            
                                            # Tool call arguments delta
                                            if "arguments" in tool_call["function"]:
                                                anthropic_event = {
                                                    "type": "content_block_delta",
                                                    "index": i + 1,
                                                    "delta": {
                                                        "type": "input_json_delta",
                                                        "partial_json": tool_call["function"]["arguments"]
                                                    }
                                                }
                                                yield f"event: content_block_delta\n"
                                                yield f"data: {json.dumps(anthropic_event)}\n\n"
                        
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse streaming chunk: {data}")
                            continue
            
            # Return streaming response
            return StreamingResponse(
                stream_handler(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming request
            openai_response = await call_openai_responses_api(openai_request)
            logger.debug(f"OpenAI response: {json.dumps(openai_response, default=str)[:500]}...")
            
            # Convert response to Anthropic format
            anthropic_response = openai_to_anthropic_response(openai_response)
            
            logger.info(f"Request completed successfully")
            return JSONResponse(content=anthropic_response)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Handle token counting requests."""
    # For now, return a placeholder response
    # In a real implementation, you'd use tiktoken or similar
    return JSONResponse({
        "input_tokens": 100  # Placeholder
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
