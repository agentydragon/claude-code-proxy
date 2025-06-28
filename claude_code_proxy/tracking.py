"""Conversation tracking and append-detection logic."""

import json
import logging
import time

logger = logging.getLogger(__name__)


class ConversationTracker:
    """Encapsulate conversation cache, append detection, and indexing."""

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._index: dict[str, str] = {}

    def count_thinking_blocks(self, messages: list[dict]) -> int:
        return sum(
            1
            for msg in messages
            for block in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
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

    def detect_append(self, messages: list[dict], conversation_id: str) -> tuple[str, bool, int]:
        """Detect if messages append to an existing conversation using indexed prefixes."""
        cleaned = self._clean_messages(messages)
        is_append = False
        append_from_index = -1

        # Look for longest matching prefix shorter than full messages
        for i in range(len(cleaned) - 1, 0, -1):
            sig = json.dumps(cleaned[:i], sort_keys=True)
            if sig in self._index:
                conversation_id = self._index[sig]
                is_append = True
                append_from_index = i
                break

        return conversation_id, is_append, append_from_index

    def update(
        self,
        conversation_id: str,
        messages: list[dict],
        *,
        had_reasoning_filtered: bool,
        original_had_reasoning: bool,
    ) -> None:
        """Update cache and index for a conversation with cleaned messages."""
        cleaned = self._clean_messages(messages)
        self._cache[conversation_id] = {
            "messages": cleaned,
            "message_count": len(cleaned),
            "last_update": time.time(),
            "had_reasoning_filtered": had_reasoning_filtered,
            "original_had_reasoning": original_had_reasoning,
        }
        sig = json.dumps(cleaned, sort_keys=True)
        self._index[sig] = conversation_id

    def _clean_messages(self, messages: list[dict]) -> list[dict]:
        """Remove thinking blocks from messages for caching and indexing."""
        cleaned_msgs: list[dict] = []
        for msg in messages:
            entry: dict = {"role": msg.get("role")}
            content = msg.get("content", [])
            if isinstance(content, list):
                entry["content"] = [
                    block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")
                ]
            else:
                entry["content"] = content
            cleaned_msgs.append(entry)
        return cleaned_msgs


# Global tracker instance
tracker = ConversationTracker()
# mypy: ignore_errors
