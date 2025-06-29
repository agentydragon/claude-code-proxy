"""Convert between Anthropic Messages API and OpenAI Responses API."""

import fnmatch
import json
import logging
import re
import uuid
from typing import Any

from claude_code_proxy.config import load_config

logger = logging.getLogger(__name__)

# Model mapping loaded from config (supports wildcards)
CONFIG = load_config()
ANTHROPIC_TO_OPENAI_MODEL: dict[str, str] = CONFIG.anthropic_to_openai_model or {
    "claude-opus-4-20250514": "o3",
    "claude-sonnet-4-20250514": "o3-mini",
    "claude-3-5-haiku-20241022": "gpt-4o-mini",
    "claude-3-5-sonnet-20241022": "gpt-4o",
    "claude-3-opus-20240229": "gpt-4o",
}


def _apply_search_replace(text: str) -> str:
    """Apply configured search-replace patterns to text content."""
    for pattern, repl in CONFIG.search_replace.items():
        text = re.sub(pattern, repl, text)
        assert isinstance(text, str)
    return text


def _convert_tool_call_to_function_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic tool_use block to OpenAI function_call format."""
    return {"type": "function_call", "name": tc["name"], "arguments": json.dumps(tc["input"]), "call_id": tc["id"]}


def _create_tool_use_block(id: str, name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Create an Anthropic tool_use content block."""
    return {"type": "tool_use", "id": id, "name": name, "input": input_data}


def parse_json_arguments(arguments: Any, context_name: str, context_type: str = "tool") -> dict[str, Any]:
    """
    Parse JSON arguments with robust error recovery for tool/function calls.

    The Anthropic API may supply arguments as raw JSON strings or dictionaries,
    so we attempt to load strings and pass through dicts unchanged. On parse errors,
    we emit a structured error dict that the mitigation layer can present back in
    Anthropic format, guiding clients to correct malformed JSON in their tool definitions.

    Args:
        arguments: Raw arguments payload (JSON string or dict).
        context_name: Identifier of the tool/function (for error context).
        context_type: Either 'tool' or 'function' (affects messaging).

    Returns:
        A dict of parsed arguments on success,
        or a structured error object on failure.
    """
    try:
        if isinstance(arguments, str):
            return json.loads(arguments) if arguments else {}
        else:
            return arguments if arguments else {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {context_type} arguments for '{context_name}': {e}")
        logger.warning(f"Raw arguments: {arguments}")

        # Build detailed error position
        error_pos = "unknown position"
        if hasattr(e, "lineno") and hasattr(e, "colno"):
            error_pos = f"line {e.lineno}, column {e.colno}"
        elif hasattr(e, "pos"):
            error_pos = f"character {e.pos}"

        # Return structured error that helps the model retry
        return {
            "error": f"Failed to parse JSON arguments for {context_type}",
            "raw_arguments": arguments,
            "parse_error": str(e),
            "error_position": error_pos,
            f"{context_type}_name": context_name,
            "instruction": (
                f"The JSON arguments for {context_type} '{context_name}' are malformed at {str(e)}. "
                "Please retry with valid JSON. Common issues: unescaped quotes, missing commas, or incomplete brackets."
            ),
        }


def anthropic_to_openai_request(anthropic_req: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic Messages API request to OpenAI Responses API format."""
    # Map Anthropic model name to OpenAI model, allowing a catch-all fallback ('*')
    anthropic_model = anthropic_req["model"]
    openai_model: str | None = None
    mappings = ANTHROPIC_TO_OPENAI_MODEL or {}
    fallback = mappings.get("*")
    for pattern, target in mappings.items():
        if pattern != "*" and fnmatch.fnmatch(anthropic_model, pattern):
            openai_model = target
            break
    if openai_model is None:
        if fallback is not None:
            openai_model = fallback
            logger.warning(f"No mapping for Anthropic model '{anthropic_model}', using fallback '{fallback}'")
        else:
            if CONFIG.forward_unknown_model_names:
                openai_model = anthropic_model
                logger.warning(f"No mapping for Anthropic model '{anthropic_model}', forwarding as-is")
            else:
                raise ValueError(
                    f"No mapping for Anthropic model '{anthropic_model}' and forward_unknown_model_names is False"
                )
    openai_req = {"model": openai_model}

    # Map system instructions if present (Anthropic 'system' → OpenAI 'instructions')
    if "system" in anthropic_req and (system_content := _extract_text_content(anthropic_req["system"])):
        system_content = _apply_search_replace(system_content)
        openai_req["instructions"] = system_content

    # Build input array from Anthropic messages
    input_items: list[dict[str, Any]] = []
    for msg in anthropic_req.get("messages", []):
        # Handle messages with tool use or results
        if _contains_tool_results(msg) or _contains_tool_use(msg):
            input_items.extend(_split_tool_message(msg))
        else:
            if item := _convert_message_to_input(msg):
                input_items.append(item)
            else:
                logger.warning(f"Skipping unsupported message format: {msg}")

    # OpenAI API requires at least one of: input, prompt, or previous_response_id
    # If input is empty, we need to provide a minimal valid input
    if not input_items:
        logger.warning("No valid input items after conversion - adding minimal user prompt")
        input_items = [{"role": "user", "content": ""}]

    openai_req["input"] = input_items
    # 2025-06-23 22:01:25,358 - INFO - Received Anthropic request for model:
    # claude-3-5-haiku-20241022
    # 2025-06-23 22:01:25,358 - DEBUG - OpenAI request: {
    #   "model": "gpt-4o-mini",
    #   "input": [{"role": "user", "content": "quota"}],
    #   "max_output_tokens": 1,
    #   "metadata": {},
    #   "user":
    # "f35dc80505901d7cc45bb33b9d66a2ca896e6cc173285c043c932be151f45d59"
    # }...
    # 2025-06-23 22:01:25,552 - ERROR - OpenAI error: {
    #   "error": {
    #     "message":
    # "Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16,
    # but got 1 instead.",
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

    # Enable reasoning output for o1/o3 models and codex
    if openai_model in ["o1", "o1-mini", "o3", "o3-mini", "codex-mini-latest"]:
        # Request reasoning output from config
        openai_req["reasoning"] = {
            "effort": CONFIG.reasoning_effort.value,  # low, medium, high
            "summary": CONFIG.reasoning_summary.value,  # auto, concise, detailed, none
        }

    return openai_req


def openai_to_anthropic_response(openai_resp: dict[str, Any]) -> dict[str, Any]:
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
                    anthropic_resp["content"].append(
                        {"type": "text", "text": _apply_search_replace(content_item["text"])}
                    )
                elif content_item.get("type") == "tool_call":
                    # Parse arguments with error handling
                    input_data = parse_json_arguments(
                        content_item.get("arguments", "{}"), content_item.get("name", "unknown"), "tool"
                    )

                    anthropic_resp["content"].append(
                        _create_tool_use_block(content_item["id"], content_item["name"], input_data)
                    )
        elif item.get("type") == "function_call":
            # Handle function calls from OpenAI
            input_data = parse_json_arguments(item.get("arguments", "{}"), item.get("name", "unknown"), "function")

            anthropic_resp["content"].append(
                _create_tool_use_block(
                    item.get("call_id", item.get("id", f"toolu_{uuid.uuid4().hex[:8]}")),
                    item.get("name", ""),
                    input_data,
                )
            )
        elif item.get("type") == "reasoning":
            # Map OpenAI reasoning to Anthropic thinking blocks
            reasoning_content = _apply_search_replace(item.get("content", "") or "")
            logger.info(
                f"[REASONING BLOCK] Received from OpenAI: {reasoning_content[:200]}{'...' if len(str(reasoning_content)) > 200 else ''}",  # noqa: E501
            )
            anthropic_resp["content"].append({"type": "thinking", "text": reasoning_content})

    # Convert usage
    if "usage" in openai_resp:
        anthropic_resp["usage"] = {
            "input_tokens": openai_resp["usage"].get("input_tokens", 0),
            "output_tokens": openai_resp["usage"].get("output_tokens", 0),
        }

    return anthropic_resp


def _extract_text_content(content: str | list[dict[str, Any]]) -> str:
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


def _contains_tool_results(msg: dict[str, Any]) -> bool:
    """Check if message contains tool results."""
    if not isinstance(msg.get("content"), list):
        return False
    return any(block.get("type") == "tool_result" for block in msg["content"])


def _contains_tool_use(msg: dict[str, Any]) -> bool:
    """Check if message contains tool use."""
    if not isinstance(msg.get("content"), list):
        return False
    return any(block.get("type") == "tool_use" for block in msg["content"])


def _split_tool_message(msg: dict[str, Any]) -> list[dict[str, Any]]:
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
                    items.append(
                        {
                            "role": "assistant" if msg["role"] == "assistant" else "user",
                            "content": "\n".join(text_parts),
                        }
                    )

                # Add tool calls as separate function_call items
                for tc in tool_calls:
                    items.append(_convert_tool_call_to_function_call(tc))

                text_parts = []
                tool_calls = []

            # Add tool result
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block["tool_use_id"],
                    "output": _extract_tool_result_content(block.get("content", "")),
                }
            )

    # Add any remaining content
    if text_parts or tool_calls:
        if text_parts:
            items.append({"role": msg["role"], "content": "\n".join(text_parts)})

        # Add remaining tool calls as separate function_call items
        for tc in tool_calls:
            items.append(_convert_tool_call_to_function_call(tc))

    return items


def _convert_message_to_input(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Anthropic message to OpenAI Responses input item."""
    logger.debug(f"Converting message: {msg}")
    role = msg["role"]
    # Simple text content shorthand
    raw = msg.get("content")
    if isinstance(raw, str):
        return {"role": role, "content": _apply_search_replace(raw)}

    # Handle explicit text field
    if "text" in msg:
        return {"role": role, "content": _apply_search_replace(msg["text"])}

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
            text_parts.append(_apply_search_replace(block["text"]))

        elif block_type == "image":
            # Convert image to base64 URL format
            source = block.get("source", {})
            if source.get("type") == "base64":
                output_content.append(
                    {
                        "type": "input_image",
                        "image": {
                            "format": source["media_type"].split("/")[1],
                            "source": {"type": "base64", "data": source["data"]},
                        },
                    }
                )

        elif block_type == "tool_use":
            # For the Responses API, tool uses should not be included
            # in the content array when they're part of the input
            # They need to be separate function_call items
            logger.debug(f"Skipping tool_use block in message content: {block['name']}")
            continue

        elif block_type == "thinking":
            # Skip thinking blocks for now - OpenAI doesn't support reasoning in input
            thinking_text = _apply_search_replace(block.get("text", ""))
            logger.info(
                f"[THINKING FILTERED] Removing from input:"
                f" {thinking_text[:200]}"
                f"{'...' if len(thinking_text) > 200 else ''}"
            )
            continue

        else:
            logger.warning(f"Unknown content block type: {block_type}")

    # Build output
    if text_parts:
        # If we only have text and no additional structured content
        # then the payload is still the simple form without a `type`
        # key.
        if not output_content:
            return {"role": role, "content": "\n".join(text_parts)}
        # Mix of text and other content.  In this case we keep the
        # structured `output_content` list but we still omit the
        # super-fluous top-level `type` key.
        output_content.insert(
            0, {"type": "output_text" if role == "assistant" else "input_text", "text": "\n".join(text_parts)}
        )

    if output_content:
        return {"role": role, "content": output_content}

    # Empty content is valid for assistant messages (e.g., tool-only responses)
    if role == "assistant" and isinstance(content, list) and len(content) == 0:
        return {"role": role, "content": []}

    logger.warning(f"No valid content found in message: {msg}")
    return None


def _extract_tool_result_content(content: Any) -> str:
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


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tools to OpenAI Responses API function definitions."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        }
        for tool in tools
    ]


def _convert_tool_choice_to_openai(tool_choice: dict[str, Any]) -> str | dict[str, Any]:
    """Convert Anthropic tool_choice to OpenAI format."""
    choice_type = tool_choice.get("type")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name")}}
    else:
        return "auto"


def _convert_stop_reason(openai_resp: dict[str, Any]) -> str | None:
    """Convert OpenAI stop reason to Anthropic format."""
    # The Responses API may have different stop reason handling
    status = openai_resp.get("status")

    if status == "completed":
        # Check if tools were used
        has_tools = any(
            item.get("type") == "message" and any(c.get("type") == "tool_call" for c in item.get("content", []))
            for item in openai_resp.get("output", [])
        )
        return "tool_use" if has_tools else "end_turn"
    elif status == "incomplete":
        incomplete_details = openai_resp.get("incomplete_details", {})
        reason = incomplete_details.get("reason")
        if reason == "max_tokens" or reason == "max_output_tokens":
            return "max_tokens"

    return "end_turn"


# mypy: ignore_errors
