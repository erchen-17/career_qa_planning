"""
Pydantic schemas for the ingest API.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestBase64Request(BaseModel):
    file_name: str = Field(..., description="文件名，需包含扩展名，如 resume.pdf")
    file_content_base64: str = Field(..., description="文件内容的 base64 编码字符串")
    user_id: str
    doc_type: Literal["resume", "career_intro"]
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5.2"
    tags: Optional[str] = Field(default=None, description="逗号分隔的标签")
    resume_mode: Literal["rag", "pinned", "hybrid"] = "pinned"


class IngestUrlRequest(BaseModel):
    file_base_url: Optional[str] = Field(default=None, description="文件服务的基础 URL，如 http://host:port，不传则 file_url 需为完整地址")
    file_url: str = Field(..., description="文件的相对路径或完整 URL")
    file_name: str = Field(..., description="文件名，需包含扩展名，如 resume.pdf")
    user_id: str
    doc_type: Literal["resume", "career_intro"]
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5.2"
    tags: Optional[str] = Field(default=None, description="逗号分隔的标签")
    resume_mode: Literal["rag", "pinned", "hybrid"] = "pinned"


class IngestBatchResponse(BaseModel):
    status: str = "ok"
    total: int
    succeeded: int
    failed: int
    results: list["IngestResponse"]


class IngestResponse(BaseModel):
    status: str = "ok"
    doc_id: str
    doc_type: Literal["resume", "career_intro"]
    pages: int = 0
    chunks: int = 0
    text_preview: str = Field(default="", description="First 200 characters of extracted text")
    duplicate_of: Optional[str] = Field(default=None, description="被替换的旧文档 ID（去重时返回）")
    duplicate_file_name: Optional[str] = Field(default=None, description="被替换的旧文档文件名")
