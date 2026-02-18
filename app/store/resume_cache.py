"""
Simple JSON-based cache for pinned resume full text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_CACHE_DIR = PROJECT_ROOT / "data" / "resume_cache"


def _user_cache_path(user_id: str) -> Path:
    return _CACHE_DIR / f"{user_id}.json"


def save_resume_text(user_id: str, doc_id: str, text: str) -> None:
    """Save or overwrite the pinned resume text for a user."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"user_id": user_id, "doc_id": doc_id, "text": text}
    path = _user_cache_path(user_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved pinned resume for user=%s, doc_id=%s (%d chars)", user_id, doc_id, len(text))


def load_resume_text(user_id: str) -> str | None:
    """Load the pinned resume text for a user, or None if not cached."""
    path = _user_cache_path(user_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("text")
