"""Convert between Anthropic Messages API and OpenAI Responses API."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Model mapping
ANTHROPIC_TO_OPENAI_MODEL = {
  "claude-opus-4-20250514": "o3",
    "claude-sonnet-4-20250514": "o4-mini",
#    "claude-3-5-haiku-20241022": "gpt-4o-mini",
#    "claude-3-5-sonnet-20241022": "gpt-4o-mini",
#    "claude-3-opus-20240229": "gpt-4o-mini",
}

def anthropic_to_openai_request(anthropic_req: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic Messages API request to OpenAI Responses API format."""
    # Map model
    anthropic_model = anthropic_req["model"]
    openai_model = ANTHROPIC_TO_OPENAI_MODEL.get(anthropic_model, "gpt-4o-mini")
    openai_req = {"model": openai_model}

    # Map system instructions if present (Anthropic 'system' → OpenAI 'instructions')
    if "system" in anthropic_req and (system_content := _extract_text_content(anthropic_req["system"])):
        openai_req["instructions"] = system_content

    # Build input array from Anthropic messages
    input_items: List[Dict[str, Any]] = []
    for msg in anthropic_req.get("messages", []):
        # Handle messages with tool use or results
        if _contains_tool_results(msg) or _contains_tool_use(msg):
            input_items.extend(_split_tool_message(msg))
        else:
            if (item := _convert_message_to_input(msg)):
                input_items.append(item)
            else:
                logger.warning(f"Skipping unsupported message format: {msg}")

    openai_req["input"] = input_items
    # 2025-06-23 22:01:25,358 - INFO - Received Anthropic request for model: claude-3-5-haiku-20241022
    # 2025-06-23 22:01:25,358 - DEBUG - OpenAI request: {"model": "gpt-4o-mini", "input": [{"role": "user", "content": "quota"}], "max_output_tokens": 1, "metadata": {}, "user": "f35dc80505901d7cc45bb33b9d66a2ca896e6cc173285c043c932be151f45d59"}...
    # 2025-06-23 22:01:25,552 - ERROR - OpenAI error: {
    #   "error": {
    #     "message": "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 1 instead.",
    #     "type": "invalid_request_error",
    #     "param": "max_output_tokens",
    #     "code": "integer_below_min_value"
    #   }
    # }

    # Map parameters
    if "max_tokens" in anthropic_req:
        openai_req["max_output_tokens"] = anthropic_req["max_tokens"]
        if openai_req["max_output_tokens"] < 16:
            openai_req["max_output_tokens"] = 16
        
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]
        
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]
        
    if "stream" in anthropic_req:
        openai_req["stream"] = anthropic_req["stream"]
        
    # Handle tools
    if "tools" in anthropic_req:
        openai_req["tools"] = _convert_tools_to_openai(anthropic_req["tools"])
        
    if "tool_choice" in anthropic_req:
        openai_req["tool_choice"] = _convert_tool_choice_to_openai(anthropic_req["tool_choice"])

    # Handle metadata
    if "metadata" in anthropic_req:
        openai_req["metadata"] = {}
        if "user_id" in anthropic_req["metadata"]:
            openai_req["user"] = anthropic_req["metadata"]["user_id"]

    return openai_req

def openai_to_anthropic_response(openai_resp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI Responses API response to Anthropic Messages format."""
    # Build Anthropic response
    anthropic_resp = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": openai_resp.get("model", "unknown"),
        "stop_reason": _convert_stop_reason(openai_resp),
        "stop_sequence": None,
    }

    # Convert output items to content blocks
    for item in openai_resp.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            # Convert assistant message content
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    anthropic_resp["content"].append({
                        "type": "text",
                        "text": content_item["text"]
                    })
                elif content_item.get("type") == "tool_call":
                    anthropic_resp["content"].append({
                        "type": "tool_use",
                        "id": content_item["id"],
                        "name": content_item["name"],
                        "input": json.loads(content_item["arguments"]) if isinstance(content_item["arguments"], str) else content_item["arguments"]
                    })
        elif item.get("type") == "function_call":
            # Handle function calls from OpenAI
            anthropic_resp["content"].append({
                "type": "tool_use",
                "id": item.get("call_id", item.get("id", f"toolu_{uuid.uuid4().hex[:8]}")),
                "name": item.get("name", ""),
                "input": json.loads(item.get("arguments", "{}")) if isinstance(item.get("arguments", "{}"), str) else item.get("arguments", {})
            })
        elif item.get("type") == "reasoning":
            # Skip reasoning blocks for now - Anthropic doesn't have an equivalent
            pass

    # Convert usage
    if "usage" in openai_resp:
        anthropic_resp["usage"] = {
            "input_tokens": openai_resp["usage"].get("input_tokens", 0),
            "output_tokens": openai_resp["usage"].get("output_tokens", 0)
        }

    return anthropic_resp

def _extract_text_content( content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract text content from various formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts).strip()
    return ""

def _contains_tool_results( msg: Dict[str, Any]) -> bool:
    """Check if message contains tool results."""
    if not isinstance(msg.get("content"), list):
        return False
    return any(block.get("type") == "tool_result" for block in msg["content"])

def _contains_tool_use( msg: Dict[str, Any]) -> bool:
    """Check if message contains tool use."""
    if not isinstance(msg.get("content"), list):
        return False
    return any(block.get("type") == "tool_use" for block in msg["content"])

def _split_tool_message( msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split a message containing tool results into separate input items."""
    items = []
    text_parts = []
    tool_calls = []

    for block in msg.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(block)
        elif block.get("type") == "tool_result":
            # Add any accumulated content first
            if text_parts or tool_calls:
                # For the Responses API, we need to handle tool calls differently
                # They should be separate output items, not part of message content
                if text_parts:
                    items.append({
                        "role": "assistant" if msg["role"] == "assistant" else "user",
                        "content": "\n".join(text_parts)
                    })

                # Add tool calls as separate function_call items
                for tc in tool_calls:
                    items.append({
                        "type": "function_call",
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                        "call_id": tc["id"]
                    })

                text_parts = []
                tool_calls = []

            # Add tool result
            items.append({
                "type": "function_call_output",
                "call_id": block["tool_use_id"],
                "output": _extract_tool_result_content(block.get("content", ""))
            })

    # Add any remaining content
    if text_parts or tool_calls:
        if text_parts:
            items.append({
                "role": msg["role"],
                "content": "\n".join(text_parts)
            })
        
        # Add remaining tool calls as separate function_call items
        for tc in tool_calls:
            items.append({
                "type": "function_call",
                "name": tc["name"],
                "arguments": json.dumps(tc["input"]),
                "call_id": tc["id"]
            })
    
    return items

def _convert_message_to_input(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert Anthropic message to OpenAI Responses input item."""
    logger.debug(f"Converting message: {msg}")
    role = msg["role"]
    # Simple text content shorthand
    raw = msg.get("content")
    if isinstance(raw, str):
        return {"role": role, "content": raw}

    # Handle explicit text field
    if "text" in msg:
        return {"role": role, "content": msg["text"]}

    content = raw if raw is not None else []
    # Complex content blocks
    if not isinstance(content, list):
        logger.warning(f"Unsupported content format: {type(content)} in message: {msg}")
        return None
    text_parts = []
    output_content = []

    for block in content:
        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block["text"])

        elif block_type == "image":
            # Convert image to base64 URL format
            source = block.get("source", {})
            if source.get("type") == "base64":
                output_content.append({
                    "type": "input_image",
                    "image": {
                        "format": source["media_type"].split("/")[1],
                        "source": {
                            "type": "base64",
                            "data": source["data"]
                        }
                    }
                })

        elif block_type == "tool_use":
            # For the Responses API, tool uses should not be included 
            # in the content array when they're part of the input
            # They need to be separate function_call items
            logger.debug(f"Skipping tool_use block in message content: {block['name']}")
            continue

        elif block_type == "thinking":
            # Skip thinking blocks - OpenAI doesn't support them
            logger.debug("Skipping thinking block")
            continue

        else:
            logger.warning(f"Unknown content block type: {block_type}")

    # Build output
    if text_parts:
        # If we only have text and no additional structured content
        # then the payload is still the simple form without a `type`
        # key.
        if not output_content:
            return {
                "role": role,
                "content": "\n".join(text_parts)
            }
        # Mix of text and other content.  In this case we keep the
        # structured `output_content` list but we still omit the
        # super-fluous top-level `type` key.
        output_content.insert(0, {
            "type": "output_text" if role == "assistant" else "input_text",
            "text": "\n".join(text_parts)
        })

    if output_content:
        return {"role": role, "content": output_content}
    logger.warning(f"No valid content found in message: {msg}")
    return None

def _extract_tool_result_content( content: Any) -> str:
    """Extract text content from tool result."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)
    else:
        return json.dumps(content)

def _convert_tools_to_openai( tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tools to OpenAI format."""
    # For the Responses API, tools have a simpler structure
    openai_tools = []
    
    for tool in tools:
        openai_tool = {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {})
        }
        openai_tools.append(openai_tool)
        
    return openai_tools

def _convert_tool_choice_to_openai( tool_choice: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    """Convert Anthropic tool_choice to OpenAI format."""
    choice_type = tool_choice.get("type")
    
    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name")}
        }
    else:
        return "auto"

def _convert_stop_reason( openai_resp: Dict[str, Any]) -> Optional[str]:
    """Convert OpenAI stop reason to Anthropic format."""
    # The Responses API may have different stop reason handling
    status = openai_resp.get("status")
    
    if status == "completed":
        # Check if tools were used
        has_tools = any(
            item.get("type") == "message" and 
            any(c.get("type") == "tool_call" for c in item.get("content", []))
            for item in openai_resp.get("output", [])
        )
        return "tool_use" if has_tools else "end_turn"
    elif status == "incomplete":
        incomplete_details = openai_resp.get("incomplete_details", {})
        reason = incomplete_details.get("reason")
        if reason == "max_tokens" or reason == "max_output_tokens":
            return "max_tokens"
    
    return "end_turn"
