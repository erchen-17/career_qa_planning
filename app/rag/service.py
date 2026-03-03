"""
RAG service: retrieval policy routing, prompt assembly, LLM generation.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.llm.router import get_chat_model, Provider
from app.rag.prompts import build_messages
from app.rag.schemas import ChatResponse, Citation, DebugInfo
from app.store.chroma_store import get_chroma_store
from app.store.resume_cache import load_resume_text

logger = logging.getLogger(__name__)

# Keywords for auto policy detection
_RESUME_KEYWORDS = re.compile(r"(我|我的|简历|经历|擅长|技能|适合|优势|背景)")
_CAREER_KEYWORDS = re.compile(r"(岗位|职业|工作内容|发展路径|要求|JD|招聘|职位|行业)")


def _detect_policy(query: str) -> str:
    """Auto-detect retrieval policy based on query keywords."""
    resume_score = len(_RESUME_KEYWORDS.findall(query))
    career_score = len(_CAREER_KEYWORDS.findall(query))

    if resume_score > career_score:
        return "resume_first"
    if career_score > resume_score:
        return "career_first"
    return "blended"


def _retrieve(
    query: str,
    user_id: str,
    policy: str,
    top_k: int,
) -> list[dict]:
    """
    Retrieve chunks from Chroma according to the retrieval policy.
    """
    store = get_chroma_store()
    base_filter = {"user_id": user_id}

    if policy == "blended":
        return store.query(query_text=query, top_k=top_k, where=base_filter)

    if policy == "resume_first":
        first_type, second_type = "resume", "career_intro"
    else:  # career_first
        first_type, second_type = "career_intro", "resume"

    half = max(top_k // 2, 1)
    remaining = top_k - half

    first_results = store.query(
        query_text=query,
        top_k=half,
        where={"$and": [{"user_id": user_id}, {"doc_type": first_type}]},
    )
    second_results = store.query(
        query_text=query,
        top_k=remaining,
        where={"$and": [{"user_id": user_id}, {"doc_type": second_type}]},
    )

    return first_results + second_results


def _chunks_to_citations(chunks: list[dict]) -> list[Citation]:
    """Convert retrieved chunks to Citation objects."""
    citations = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        citations.append(Citation(
            doc_id=meta.get("doc_id", ""),
            doc_type=meta.get("doc_type", ""),
            file_name=meta.get("file_name", ""),
            page=meta.get("page", 0),
            chunk_id=meta.get("chunk_id", 0),
            snippet=chunk.get("content", "")[:200],
        ))
    return citations


async def chat(
    user_id: str,
    query: str,
    provider: Provider,
    model: str,
    retrieval_policy: str = "auto",
    resume_mode: str = "pinned",
    top_k: int = 6,
) -> ChatResponse:
    """
    Full RAG pipeline:
      1. Detect/apply retrieval policy
      2. Retrieve relevant chunks
      3. Optionally inject pinned resume
      4. Build prompt and call LLM
      5. Return answer + citations
    """
    # Step 1: Resolve policy
    if retrieval_policy == "auto":
        effective_policy = _detect_policy(query)
    else:
        effective_policy = retrieval_policy
    logger.info("Chat: user=%s, policy=%s (effective=%s)", user_id, retrieval_policy, effective_policy)

    # Step 2: Retrieve
    chunks = _retrieve(query, user_id, effective_policy, top_k)
    logger.info("Chat: retrieved %d chunks", len(chunks))

    # Step 3: Pinned resume
    pinned_text = None
    used_pinned = False
    if resume_mode in ("pinned", "hybrid"):
        pinned_text = load_resume_text(user_id)
        if pinned_text:
            used_pinned = True
            logger.info("Chat: injecting pinned resume (%d chars)", len(pinned_text))

    # If resume_mode == "pinned", don't use RAG chunks for resume type
    if resume_mode == "pinned" and used_pinned:
        chunks = [c for c in chunks if c.get("metadata", {}).get("doc_type") != "resume"]

    # Step 4: Build messages and call LLM
    messages_raw = build_messages(query=query, chunks=chunks, pinned_resume_text=pinned_text)

    # Convert to LangChain message objects
    lc_messages = []
    for msg in messages_raw:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        else:
            lc_messages.append(HumanMessage(content=msg["content"]))

    llm = get_chat_model(provider=provider, model=model)
    response = await llm.ainvoke(lc_messages)
    answer = response.content

    # Step 5: Build response
    citations = _chunks_to_citations(chunks)
    debug = DebugInfo(
        retrieval_policy=effective_policy,
        used_resume_pinned=used_pinned,
        retrieved_chunks=len(chunks),
    )

    return ChatResponse(answer=answer, citations=citations, debug=debug)
