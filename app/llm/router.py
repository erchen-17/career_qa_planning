"""
LLM Router: maps (provider, model) to LangChain chat model instances.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from app.core.config import settings

Provider = Literal["openai", "anthropic"]


def get_chat_model(
    provider: Provider,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
):
    """Return a LangChain chat model for the given provider and model name."""
    if provider == "openai":
        kwargs = dict(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        kwargs = dict(
            model=model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        return ChatAnthropic(**kwargs)

    raise ValueError(f"Unsupported provider: {provider}")


def get_vlm_model(
    provider: Provider,
    model: str,
    max_tokens: int = 8192,
):
    """Return a VLM-capable chat model (same providers, higher max_tokens)."""
    return get_chat_model(
        provider=provider,
        model=model,
        temperature=0.0,  # deterministic for OCR
        max_tokens=max_tokens,
    )
