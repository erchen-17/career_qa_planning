"""
FastAPI application entry point.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.ingest.file_utils import (
    SUPPORTED_EXTS,
    detect_file_ext,
    parse_tags,
    validate_file_ext,
)
from app.ingest.schemas import IngestBase64Request, IngestBatchResponse, IngestResponse, IngestUrlRequest
from app.ingest.service import ingest_file
from app.rag.schemas import ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Career Planning Assistant API",
    version="0.1.0",
    docs_url=None,
)

_SWAGGER_CDN = "https://unpkg.com/swagger-ui-dist@5"


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url=f"{_SWAGGER_CDN}/swagger-ui-bundle.js",
        swagger_css_url=f"{_SWAGGER_CDN}/swagger-ui.css",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    doc_type: Literal["resume", "career_intro"] = Form(...),
    provider: Literal["openai", "anthropic"] = Form("openai"),
    model: str = Form("gpt-5.2"),
    tags: Optional[str] = Form(None),
    resume_mode: Literal["rag", "pinned", "hybrid"] = Form("pinned"),
):
    """
    Upload a file (PDF/DOCX/PNG/JPG), extract text via VLM OCR,
    chunk, embed, and store in the vector database.
    """
    file_name = file.filename or "unknown"
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = detect_file_ext(file_name, file_bytes=file_bytes)
    validate_file_ext(ext, file_name=file_name)

    result = await ingest_file(
        file_bytes=file_bytes,
        file_name=file_name,
        user_id=user_id,
        doc_type=doc_type,
        provider=provider,
        model=model,
        tags=parse_tags(tags),
        resume_mode=resume_mode,
    )
    return IngestResponse(**result)


@app.post("/v1/ingest_base64", response_model=IngestResponse)
async def ingest_base64(req: IngestBase64Request):
    """
    JSON-based ingest endpoint (base64-encoded file).
    Designed for Dify and other platforms that don't support multipart uploads.
    """
    import base64

    try:
        file_bytes = base64.b64decode(req.file_content_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 content")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    ext = detect_file_ext(req.file_name, file_bytes=file_bytes)
    validate_file_ext(ext, file_name=req.file_name)

    result = await ingest_file(
        file_bytes=file_bytes,
        file_name=req.file_name,
        user_id=req.user_id,
        doc_type=req.doc_type,
        provider=req.provider,
        model=req.model,
        tags=parse_tags(req.tags),
        resume_mode=req.resume_mode,
    )
    return IngestResponse(**result)


@app.post("/v1/ingest_url", response_model=IngestResponse)
async def ingest_url(req: IngestUrlRequest):
    """
    URL-based ingest endpoint.
    Downloads the file from the given URL, then processes it.
    Designed for Dify file variables to bypass base64 length limits.
    """
    import httpx

    # 拼接完整下载 URL
    if req.file_base_url:
        base = req.file_base_url.rstrip("/")
        path = req.file_url.lstrip("/")
        download_url = f"{base}/{path}"
    else:
        download_url = req.file_url

    logger.info("ingest_url: file_name=%s, 下载地址=%s", req.file_name, download_url)

    # 下载文件
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(download_url)
            resp.raise_for_status()
            file_bytes = resp.content
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"下载文件失败: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载文件失败: {e}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="下载的文件为空")

    # 多级扩展名检测：文件名 → URL → Content-Type → magic bytes
    ext = detect_file_ext(
        file_name=req.file_name,
        file_bytes=file_bytes,
        url=req.file_url,
        content_type=resp.headers.get("content-type", ""),
    )
    validate_file_ext(ext, file_name=req.file_name)

    # 如果 file_name 没有扩展名，补上检测到的扩展名
    file_name = req.file_name
    if "." not in file_name:
        file_name = f"{file_name}.{ext}"

    result = await ingest_file(
        file_bytes=file_bytes,
        file_name=file_name,
        user_id=req.user_id,
        doc_type=req.doc_type,
        provider=req.provider,
        model=req.model,
        tags=parse_tags(req.tags),
        resume_mode=req.resume_mode,
    )
    return IngestResponse(**result)


@app.post("/v1/ingest_batch", response_model=IngestBatchResponse)
async def ingest_batch(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    doc_type: Literal["resume", "career_intro"] = Form(...),
    provider: Literal["openai", "anthropic"] = Form("openai"),
    model: str = Form("gpt-5.2"),
    tags: Optional[str] = Form(None),
    resume_mode: Literal["rag", "pinned", "hybrid"] = Form("pinned"),
):
    """
    Batch upload multiple files. All files share the same user_id, doc_type, etc.
    Each file is processed independently; partial failures won't block others.
    """
    tag_list = parse_tags(tags)
    results: list[IngestResponse] = []
    succeeded = 0
    failed = 0

    for file in files:
        file_name = file.filename or "unknown"
        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Batch ingest: skipping empty file %s", file_name)
            failed += 1
            continue

        ext = detect_file_ext(file_name, file_bytes=file_bytes)
        if ext not in SUPPORTED_EXTS:
            logger.warning("Batch ingest: skipping unsupported file %s", file_name)
            failed += 1
            continue

        try:
            result = await ingest_file(
                file_bytes=file_bytes,
                file_name=file_name,
                user_id=user_id,
                doc_type=doc_type,
                provider=provider,
                model=model,
                tags=tag_list,
                resume_mode=resume_mode,
            )
            results.append(IngestResponse(**result))
            succeeded += 1
        except Exception as e:
            logger.error("Batch ingest: failed for %s: %s", file_name, e)
            failed += 1

    return IngestBatchResponse(
        total=succeeded + failed,
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.post("/v1/chat", response_model=None)
async def chat_endpoint(req: ChatRequest) -> ChatResponse | PlainTextResponse:
    """
    RAG-based Q&A: retrieve relevant chunks from the knowledge base,
    optionally inject pinned resume, and generate an answer with citations.
    """
    from app.rag.service import chat

    result = await chat(
        user_id=req.user_id,
        query=req.query,
        provider=req.provider,
        model=req.model,
        retrieval_policy=req.retrieval_policy,
        resume_mode=req.resume_mode,
        top_k=req.top_k,
    )

    if req.simple_response:
        answer = result.answer if hasattr(result, "answer") else result.get("answer", "")
        return PlainTextResponse(answer)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
