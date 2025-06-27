"""Conversation tracking and append-detection logic."""
import json
import logging

logger = logging.getLogger(__name__)


class ConversationTracker:
    """Encapsulate conversation cache and detection of append scenarios."""
    def __init__(self):
        self._cache: dict[str, dict] = {}

    def count_thinking_blocks(self, messages: list[dict]) -> int:
        return sum(
            1
            for msg in messages
            for block in (
                msg.get("content", [])
                if isinstance(msg.get("content"), list)
                else []
            )
            if isinstance(block, dict) and block.get("type") == "thinking"
        )

    def messages_equal_ignoring_thinking(self, msg1: dict, msg2: dict) -> bool:
        if msg1.get("role") != msg2.get("role"):
            return False

        def get_non_thinking_content(msg: dict):
            content = msg.get("content", [])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return [block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")]
            return content

        return json.dumps(get_non_thinking_content(msg1), sort_keys=True) == json.dumps(
            get_non_thinking_content(msg2), sort_keys=True
        )

    @property
    def cache(self) -> dict[str, dict]:
        return self._cache


# Global tracker instance
tracker = ConversationTracker()
