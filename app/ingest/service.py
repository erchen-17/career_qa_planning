"""
Ingest service: orchestrates file → text extraction → clean → chunk → embed → store.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from app.core.config import settings
from app.ingest.chunkers import chunk_text
from app.ingest.cleaners import clean_text
from app.ingest.converters import convert_file
from app.llm.vlm import extract_text_from_images
from app.store.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)


async def ingest_file(
    file_bytes: bytes,
    file_name: str,
    user_id: str,
    doc_type: Literal["resume", "career_intro"],
    provider: str,
    model: str,
    tags: Optional[list[str]] = None,
    resume_mode: str = "rag",
) -> dict:
    """
    Full ingest pipeline:
      1. Convert file to images or text
      2. OCR via VLM (if images)
      3. Clean text
      4. Chunk text
      5. Store in Chroma with metadata

    Returns dict compatible with IngestResponse.
    """
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    file_ext = file_name.rsplit(".", 1)[-1] if "." in file_name else ""
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Convert
    logger.info("Ingest: converting %s (ext=%s)", file_name, file_ext)
    converted = convert_file(file_bytes, file_ext, dpi=settings.pdf_dpi)
    pages = converted["pages"]

    # Step 2: Extract text
    if converted["mode"] == "images":
        logger.info("Ingest: OCR via VLM for %d page(s)", pages)
        page_texts = await extract_text_from_images(
            images=converted["images"],
            provider=provider,
            model=model,
        )
        full_text = "\n\n".join(page_texts)
    else:
        full_text = converted["text"]

    # Step 3: Clean
    full_text = clean_text(full_text)

    # Step 4: Chunk
    chunks = chunk_text(
        full_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Ingest: %d chunks from %s", len(chunks), file_name)

    # Step 5: Build metadata and store
    store = get_chroma_store()
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        meta = {
            "user_id": user_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "file_name": file_name,
            "file_ext": file_ext,
            "page": _guess_page(i, len(chunks), pages),
            "chunk_id": i,
            "created_at": now,
            "source": "upload",
        }
        if tags:
            meta["tags"] = ",".join(tags)
        metadatas.append(meta)
        ids.append(f"{doc_id}_chunk_{i}")

    store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    # Step 5.5: 去重 — 检测并删除已有的重复旧文档
    dedup_result = None
    try:
        from app.ingest.dedup import check_and_remove_duplicates
        dedup_result = check_and_remove_duplicates(
            store=store,
            user_id=user_id,
            new_doc_id=doc_id,
            chunks=chunks,
            doc_type=doc_type,
        )
    except Exception as e:
        logger.warning("去重检查失败（不影响入库）: %s", e)

    # Step 6: If resume and pinned/hybrid, cache full text
    if doc_type == "resume" and resume_mode in ("pinned", "hybrid"):
        from app.store.resume_cache import save_resume_text
        save_resume_text(user_id=user_id, doc_id=doc_id, text=full_text)

    text_preview = full_text[:200]
    result = {
        "status": "ok",
        "doc_id": doc_id,
        "doc_type": doc_type,
        "pages": pages,
        "chunks": len(chunks),
        "text_preview": text_preview,
    }
    if dedup_result:
        result["duplicate_of"] = dedup_result["removed_doc_id"]
        result["duplicate_file_name"] = dedup_result["file_name"]
    return result


def _guess_page(chunk_idx: int, total_chunks: int, total_pages: int) -> int:
    """Rough estimate of which page a chunk belongs to."""
    if total_pages <= 1:
        return 1
    return min(chunk_idx * total_pages // total_chunks + 1, total_pages)
