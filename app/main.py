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
from app.ingest.schemas import IngestBase64Request, IngestBatchResponse, IngestResponse, IngestUrlRequest
from app.rag.schemas import ChatRequest

_SUPPORTED_EXTS = {"pdf", "docx", "doc", "png", "jpg", "jpeg"}

_CONTENT_TYPE_MAP = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
}


def _ext_from_name(name: str) -> str:
    """从文件名提取扩展名"""
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _ext_from_url(url: str) -> str:
    """从 URL 路径提取扩展名"""
    path = url.split("?")[0].split("#")[0]
    last_segment = path.rsplit("/", 1)[-1]
    return last_segment.rsplit(".", 1)[-1].lower() if "." in last_segment else ""


def _ext_from_content_type(content_type: str) -> str:
    """从 HTTP Content-Type 推断扩展名"""
    ct = content_type.split(";")[0].strip().lower()
    return _CONTENT_TYPE_MAP.get(ct, "")


def _ext_from_magic_bytes(data: bytes) -> str:
    """从文件头 magic bytes 推断扩展名"""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:4] == b"PK\x03\x04":  # ZIP-based (DOCX)
        return "docx"
    return ""

logging.basicConfig(  #全局配置logger
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
    resume_mode: Literal["rag", "pinned", "hybrid"] = Form("rag"),
):
    """
    Upload a file (PDF/DOCX/PNG/JPG), extract text via VLM OCR,
    chunk, embed, and store in the vector database.
    """
    from app.ingest.service import ingest_file

    # Validate file extension
    file_name = file.filename or "unknown"
    ext = _ext_from_name(file_name)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 文件名无扩展名时，用 magic bytes 检测
    if ext not in _SUPPORTED_EXTS:
        ext = _ext_from_magic_bytes(file_bytes)
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: pdf, docx, png, jpg",
        )

    # Parse tags
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

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

    return IngestResponse(**result)


@app.post("/v1/ingest_base64", response_model=IngestResponse)
async def ingest_base64(req: IngestBase64Request):
    """
    JSON-based ingest endpoint (base64-encoded file).
    Designed for Dify and other platforms that don't support multipart uploads.
    """
    import base64

    from app.ingest.service import ingest_file

    # Validate file extension
    ext = _ext_from_name(req.file_name)

    try:
        file_bytes = base64.b64decode(req.file_content_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 content")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 文件名无扩展名时，用 magic bytes 检测
    if ext not in _SUPPORTED_EXTS:
        ext = _ext_from_magic_bytes(file_bytes)
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type (file_name={req.file_name}). Supported: pdf, docx, png, jpg",
        )

    tag_list = None
    if req.tags:
        tag_list = [t.strip() for t in req.tags.split(",") if t.strip()]

    result = await ingest_file(
        file_bytes=file_bytes,
        file_name=req.file_name,
        user_id=req.user_id,
        doc_type=req.doc_type,
        provider=req.provider,
        model=req.model,
        tags=tag_list,
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

    from app.ingest.service import ingest_file

    # 多级扩展名检测：file_name → file_url
    ext = _ext_from_name(req.file_name)
    if ext not in _SUPPORTED_EXTS:
        ext = _ext_from_url(req.file_url)

    # 拼接完整下载 URL
    if req.file_base_url:
        base = req.file_base_url.rstrip("/")
        path = req.file_url.lstrip("/")
        download_url = f"{base}/{path}"
    else:
        download_url = req.file_url

    logger.info("ingest_url: file_name=%s, ext=%s, 下载地址=%s", req.file_name, ext, download_url)

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

    # 扩展名仍未确定时，用 Content-Type 和 magic bytes 回退检测
    if ext not in _SUPPORTED_EXTS:
        ext = _ext_from_content_type(resp.headers.get("content-type", ""))
    if ext not in _SUPPORTED_EXTS:
        ext = _ext_from_magic_bytes(file_bytes)
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"无法识别文件类型 (file_name={req.file_name}). Supported: pdf, docx, png, jpg",
        )

    # 如果 file_name 没有扩展名，补上检测到的扩展名（确保后续处理正确）
    file_name = req.file_name
    if "." not in file_name:
        file_name = f"{file_name}.{ext}"

    tag_list = None
    if req.tags:
        tag_list = [t.strip() for t in req.tags.split(",") if t.strip()]

    result = await ingest_file(
        file_bytes=file_bytes,
        file_name=file_name,
        user_id=req.user_id,
        doc_type=req.doc_type,
        provider=req.provider,
        model=req.model,
        tags=tag_list,
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
    resume_mode: Literal["rag", "pinned", "hybrid"] = Form("rag"),
):
    """
    Batch upload multiple files. All files share the same user_id, doc_type, etc.
    Each file is processed independently; partial failures won't block others.
    """
    import logging

    from app.ingest.service import ingest_file

    logger = logging.getLogger(__name__)
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    results: list[IngestResponse] = []
    succeeded = 0
    failed = 0

    for file in files:
        file_name = file.filename or "unknown"
        ext = _ext_from_name(file_name)

        file_bytes = await file.read()
        if not file_bytes:
            logger.warning("Batch ingest: skipping empty file %s", file_name)
            failed += 1
            continue

        # 文件名无扩展名时，用 magic bytes 检测
        if ext not in _SUPPORTED_EXTS:
            ext = _ext_from_magic_bytes(file_bytes)
        if ext not in _SUPPORTED_EXTS:
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


@app.post("/v1/chat")
async def chat_endpoint(req: ChatRequest):
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
