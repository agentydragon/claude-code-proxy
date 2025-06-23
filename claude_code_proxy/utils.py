"""Utility functions for claude-code-proxy."""
import json
import uuid
from typing import Any, Dict, Optional, Union


def serialize_content(content: Any) -> str:
    """Serialize content to string, handling various types safely."""
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, dict):
        # Special handling for text content
        if content.get("type") == "text":
            return content.get("text", "")
    
    # For dicts and other types, try JSON serialization first
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def extract_tool_call_info(tool_call: Union[Dict[str, Any], Any]) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """Extract name, tool_id, and function from a tool call object.
    
    Args:
        tool_call: Either a dict or an object with attributes
        
    Returns:
        Tuple of (name, tool_id, function)
    """
    if isinstance(tool_call, dict):
        function = tool_call.get('function', {})
        name = function.get('name', '') if isinstance(function, dict) else ""
        tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
    else:
        # Handle object with attributes
        function = getattr(tool_call, 'function', None)
        name = getattr(function, 'name', '') if function else ''
        tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
    
    return name, tool_id, function


def safe_get_nested(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely get nested attributes or dict values.
    
    Args:
        obj: Object to get value from
        *keys: Keys/attributes to traverse
        default: Default value if not found
        
    Returns:
        The value at the nested path, or default if not found
    """
    current = obj
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                current = getattr(current, key, default)
            if current is None:
                return default
        except (AttributeError, KeyError, TypeError):
            return default
    return current
