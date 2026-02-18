"""
Pydantic schemas for the ingest API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = "ok"
    doc_id: str
    doc_type: Literal["resume", "career_intro"]
    pages: int = 0
    chunks: int = 0
    text_preview: str = Field(default="", description="First 200 characters of extracted text")
