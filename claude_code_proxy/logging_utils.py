"""Logging utilities and JSONL request/response log setup."""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

# Setup XDG-compliant logging directory with session subdirectory
session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = Path(platformdirs.user_state_dir("claude-code-proxy")) / "logs" / session_id
log_dir.mkdir(parents=True, exist_ok=True)

# Session-based log files
anthropic_requests_log = log_dir / "anthropic_requests.jsonl"
anthropic_responses_log = log_dir / "anthropic_responses.jsonl"
openai_requests_log = log_dir / "openai_requests.jsonl"
openai_responses_log = log_dir / "openai_responses.jsonl"
conversation_tracking_log = log_dir / "conversation_tracking.jsonl"

def log_jsonl(filepath: Path, data: dict) -> None:
    """Append a JSON record to a file, adding timestamp and datetime if missing."""
    try:
        if "timestamp" not in data:
            data["timestamp"] = time.time()
        if "datetime" not in data:
            data["datetime"] = datetime.now().isoformat()
        with filepath.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        logger.error(f"Failed to write to {filepath}: {e}")

def truncate(obj: object, max_len: int = 10000) -> str:
    """Return a string representation of obj, truncated to max_len characters."""
   text = json.dumps(obj, ensure_ascii=False) if isinstance(obj, dict) else str(obj)
    return text if len(text) <= max_len else text[:max_len] + "..."
