"""
Pydantic schemas for the chat / RAG API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    query: str
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5.1"
    top_k: int = Field(default=6, ge=1, le=20)
    retrieval_policy: Literal["auto", "resume_first", "career_first", "blended"] = "auto"
    resume_mode: Literal["rag", "pinned", "hybrid"] = "pinned"
    simple_response: bool = Field(default=True, description="为 true 时只返回 answer 文本，适配 Dify；设为 false 返回完整 JSON")
    multi_query: Literal["off", "auto", "always"] | None = Field(
        default=None,
        description="多查询扩展模式，None 时使用全局配置"
    )


class Citation(BaseModel):
    doc_id: str
    doc_type: str
    file_name: str = ""
    page: int = 0
    chunk_id: int = 0
    snippet: str = ""


class DebugInfo(BaseModel):
    retrieval_policy: str
    used_resume_pinned: bool = False
    retrieved_chunks: int = 0
    multi_query_expanded: bool = False
    sub_queries: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    debug: DebugInfo | None = None
