"""
RAG service: retrieval policy routing, prompt assembly, LLM generation.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.llm.router import get_chat_model, Provider
from app.rag.expander import force_expand, maybe_expand, merge_results
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
    multi_query: str | None = None,
) -> ChatResponse:
    """
    Full RAG pipeline:
      1. Detect/apply retrieval policy
      2. Multi-query expansion (if enabled)
      3. Retrieve relevant chunks
      4. Optionally inject pinned resume
      5. Build prompt and call LLM
      6. Return answer + citations
    """
    # Step 1: Resolve policy
    if retrieval_policy == "auto":
        effective_policy = _detect_policy(query)
    else:
        effective_policy = retrieval_policy
    logger.info("Chat: user=%s, policy=%s (effective=%s)", user_id, retrieval_policy, effective_policy)

    # Step 2: Multi-query expansion
    effective_mq_mode = multi_query or settings.multi_query_mode
    expanded = False
    sub_queries: list[str] = []

    if effective_mq_mode in ("auto", "always"):
        expansion_provider = settings.multi_query_provider or provider or settings.chat_provider or "openai"
        expansion_model = settings.multi_query_model or model or settings.chat_model or "gpt-5.1"
        try:
            if effective_mq_mode == "always":
                sub_queries = await force_expand(
                    query=query,
                    provider=expansion_provider,
                    model=expansion_model,
                    max_queries=settings.multi_query_max_queries,
                )
            else:  # auto
                sub_queries = await maybe_expand(
                    query=query,
                    provider=expansion_provider,
                    model=expansion_model,
                    max_queries=settings.multi_query_max_queries,
                )
            expanded = bool(sub_queries)
            if expanded:
                logger.info("Chat: 多查询扩展生成 %d 个子查询: %s", len(sub_queries), sub_queries)
            else:
                logger.info("Chat: LLM 判断无需扩展")
        except Exception as e:
            logger.warning("Chat: 多查询扩展失败，回退到单查询: %s", e)

    # Step 3: Retrieve
    if expanded:
        all_queries = [query] + sub_queries
        per_query_k = max(top_k // len(all_queries) + 1, 2)
        results_per_query = [
            _retrieve(q, user_id, effective_policy, per_query_k)
            for q in all_queries
        ]
        chunks = merge_results(results_per_query, top_k)
    else:
        chunks = _retrieve(query, user_id, effective_policy, top_k)
    logger.info("Chat: retrieved %d chunks (expanded=%s)", len(chunks), expanded)

    # Step 4: Pinned resume
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

    # Step 5: Build messages and call LLM
    messages_raw = build_messages(
        query=query, chunks=chunks,
        pinned_resume_text=pinned_text,
        context_injection=settings.context_injection,
    )

    # Convert to LangChain message objects
    lc_messages = []
    for msg in messages_raw:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        else:
            lc_messages.append(HumanMessage(content=msg["content"]))

    # config 作为 fallback，请求传入的值优先
    effective_provider = provider or settings.chat_provider or "openai"
    effective_model = model or settings.chat_model or "gpt-5.1"

    logger.info("Chat: using provider=%s, model=%s", effective_provider, effective_model)
    llm = get_chat_model(provider=effective_provider, model=effective_model)

    # 联网搜索：仅 OpenAI provider 支持
    if settings.web_search_enabled and effective_provider == "openai":
        llm = llm.bind_tools([{"type": "web_search_preview"}])
        logger.info("Chat: web_search_preview enabled")

    response = await llm.ainvoke(lc_messages)
    answer = response.content
    logger.info("Chat: answer length=%d", len(answer) if answer else 0)

    # Step 6: Build response
    citations = _chunks_to_citations(chunks)
    debug = DebugInfo(
        retrieval_policy=effective_policy,
        used_resume_pinned=used_pinned,
        retrieved_chunks=len(chunks),
        multi_query_expanded=expanded,
        sub_queries=sub_queries,
    )

    return ChatResponse(answer=answer, citations=citations, debug=debug)
