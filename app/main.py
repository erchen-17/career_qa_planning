"""
FastAPI application entry point.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.ingest.schemas import IngestResponse
from app.rag.schemas import ChatRequest, ChatResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Career Planning Assistant API",
    version="0.1.0",
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
    model: str = Form("gpt-4o-mini"),
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
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext not in ("pdf", "docx", "doc", "png", "jpg", "jpeg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: pdf, docx, png, jpg",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

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


@app.post("/v1/chat", response_model=ChatResponse)
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
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
