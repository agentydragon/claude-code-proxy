"""Conversation tracking and append-detection logic."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConversationCacheEntry(BaseModel):
    """Model for a single conversation cache entry."""

    messages: list[dict[str, Any]]
    message_count: int
    last_update: float
    had_reasoning_filtered: bool
    original_had_reasoning: bool


class ConversationCache(BaseModel):
    """Model for the entire conversation cache."""

    cache: dict[str, ConversationCacheEntry] = Field(default_factory=dict)
    version: int = 1
    timestamp: float = Field(default_factory=time.time)


class ConversationTracker:
    """Encapsulate conversation cache, append detection, and indexing with persistence."""

    def __init__(self, max_cache_size: int = 1000):
        self._cache_data = ConversationCache()
        self._index: dict[str, str] = {}
        self.max_cache_size = max_cache_size
        self._state_dir = Path(platformdirs.user_state_dir("claude-code-proxy"))
        self._cache_file = self._state_dir / "conversation_cache.json"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

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
    def cache(self) -> dict[str, ConversationCacheEntry]:
        return self._cache_data.cache

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

        # Create new cache entry using Pydantic model
        entry = ConversationCacheEntry(
            messages=cleaned,
            message_count=len(cleaned),
            last_update=time.time(),
            had_reasoning_filtered=had_reasoning_filtered,
            original_had_reasoning=original_had_reasoning,
        )

        self._cache_data.cache[conversation_id] = entry
        sig = json.dumps(cleaned, sort_keys=True)
        self._index[sig] = conversation_id

        # Save cache after each update (could be optimized with debouncing)
        self._save_cache()

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

    def _load_cache(self) -> None:
        """Load conversation cache from persistent storage."""
        if self._cache_file.exists():
            try:
                with self._cache_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Validate and load using Pydantic
                    self._cache_data = ConversationCache.model_validate(data)
                    # Rebuild index from cache
                    self._rebuild_index()
                    logger.info(f"Loaded {len(self._cache_data.cache)} conversations from cache")
            except Exception as e:
                logger.error(f"Failed to load conversation cache: {e}")
                self._cache_data = ConversationCache()
                self._index = {}

    def _save_cache(self) -> None:
        """Save conversation cache to persistent storage."""
        try:
            # Prune old conversations if we exceed max size
            if len(self._cache_data.cache) > self.max_cache_size:
                self._prune_old_conversations()

            # Update timestamp
            self._cache_data.timestamp = time.time()

            with self._cache_file.open("w", encoding="utf-8") as f:
                # Use Pydantic's model_dump for serialization
                json.dump(self._cache_data.model_dump(), f, indent=2)
            logger.debug(f"Saved {len(self._cache_data.cache)} conversations to cache")
        except Exception as e:
            logger.error(f"Failed to save conversation cache: {e}")

    def _rebuild_index(self) -> None:
        """Rebuild the index from the loaded cache."""
        self._index = {}
        for conversation_id, entry in self._cache_data.cache.items():
            messages = entry.messages
            # Index all prefixes of the conversation
            for i in range(1, len(messages) + 1):
                sig = json.dumps(messages[:i], sort_keys=True)
                self._index[sig] = conversation_id

    def _prune_old_conversations(self) -> None:
        """Remove oldest conversations to stay under max_cache_size."""
        # Sort by last update time
        sorted_convs = sorted(self._cache_data.cache.items(), key=lambda x: x[1].last_update)
        # Remove oldest conversations
        to_remove = (
            len(self._cache_data.cache) - self.max_cache_size + 100
        )  # Remove 100 extra to avoid frequent pruning
        for conv_id, _ in sorted_convs[:to_remove]:
            del self._cache_data.cache[conv_id]
            logger.debug(f"Pruned old conversation: {conv_id}")

    def save(self) -> None:
        """Public method to trigger cache save."""
        self._save_cache()


# Global tracker instance
tracker = ConversationTracker()
# mypy: ignore_errors
