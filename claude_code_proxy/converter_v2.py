"""Convert between Anthropic Messages API and OpenAI Responses API."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def anthropic_to_openai_request(anthropic_req: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic Messages API request to OpenAI Responses API format."""
    openai_req = {
        "model": anthropic_req["model"],
    }
    
    # Build input array
    input_items = []
    
    # Add system message if present
    if "system" in anthropic_req:
        system_content = _extract_text_content(anthropic_req["system"])
        if system_content:
            input_items.append({
                "role": "developer",  # OpenAI equivalent to Anthropic "system"
                "content": system_content
            })
    
    # Convert messages to input items
    for msg in anthropic_req.get("messages", []):
        # Handle messages that contain tool results
        if _contains_tool_results(msg):
            # Split into separate items
            items = _split_tool_message(msg)
            input_items.extend(items)
        else:
            # Convert regular message
            item = _convert_message_to_input(msg)
            if item:
                input_items.append(item)
    
    openai_req["input"] = input_items
    
    # Map parameters
    if "max_tokens" in anthropic_req:
        openai_req["max_output_tokens"] = anthropic_req["max_tokens"]
        
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

def openai_to_anthropic_response( openai_resp: Dict[str, Any]) -> Dict[str, Any]:
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
                if text_parts and tool_calls:
                    # Assistant message with both text and tool calls
                    items.append({
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "\n".join(text_parts)
                        }] + [{
                            "type": "tool_call",
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"])
                        } for tc in tool_calls]
                    })
                elif text_parts:
                    items.append({
                        "role": "assistant" if msg["role"] == "assistant" else "user",
                        "content": "\n".join(text_parts)
                    })
                elif tool_calls:
                    items.append({
                        "role": "assistant",
                        "content": [{
                            "type": "tool_call",
                            "id": tc["id"],
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"])
                        } for tc in tool_calls]
                    })
                text_parts = []
                tool_calls = []
            
            # Add tool result
            items.append({
                "type": "tool_call_output",
                "id": block["tool_use_id"],
                "output": _extract_tool_result_content(block.get("content", ""))
            })
    
    # Add any remaining content
    if text_parts or tool_calls:
        if text_parts and tool_calls:
            items.append({
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "\n".join(text_parts)
                }] + [{
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"])
                } for tc in tool_calls]
            })
        elif text_parts:
            items.append({
                "role": msg["role"],
                "content": "\n".join(text_parts)
            })
        elif tool_calls:
            items.append({
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"])
                } for tc in tool_calls]
            })
    
    return items

def _convert_message_to_input( msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert Anthropic message to OpenAI Responses input item."""
    role = msg["role"]
    content = msg["content"]
    
    # Simple string content – for a basic user / assistant / developer
    # message we only need the role and content keys. The "type" field is
    # implicit (a plain message) and *must not* be included – the
    # Responses API will reject unknown keys.  Including it caused the
    # entire input array to be ignored which manifested as the
    # "One of \"input\" or ... must be provided" error that the test
    # harness reported.
    if isinstance(content, str):
        return {
            "role": role,
            "content": content
        }
    
    # Complex content blocks
    if isinstance(content, list):
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
                output_content.append({
                    "type": "tool_call",
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": json.dumps(block["input"])
                })
                
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
            else:
                # Mix of text and other content.  In this case we keep the
                # structured `output_content` list but we still omit the
                # super-fluous top-level `type` key.
                output_content.insert(0, {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": "\n".join(text_parts)
                })
        
        if output_content:
            return {
                "role": role,
                "content": output_content
            }
    
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
    openai_tools = []
    
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
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
