"""
Application configuration loaded from config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # job_planning/

_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Settings:
    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_base_url: Optional[str] = None
    anthropic_base_url: Optional[str] = None

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # Chroma
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "career_assistant"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # PDF rendering
    pdf_dpi: int = 200

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def chroma_persist_path(self) -> Path:
        """Return absolute path for Chroma persistence directory."""
        p = Path(self.chroma_persist_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML, with environment variable overrides."""
    raw = _load_yaml(config_path or _CONFIG_PATH)

    # Environment variables take precedence over YAML
    env_overrides = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "anthropic_base_url": os.getenv("ANTHROPIC_BASE_URL"),
    }

    for key, env_val in env_overrides.items():
        if env_val is not None:
            raw[key] = env_val

    # Remove None values so dataclass defaults apply
    raw = {k: v for k, v in raw.items() if v is not None}

    return Settings(**raw)


# Module-level singleton
settings = load_settings()
