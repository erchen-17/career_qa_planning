"""
入库后文档去重：检测新文档是否与已有文档重复，若重复则删除旧文档。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.core.config import settings
from app.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# 最多取前 N 个 chunks 做去重查询，避免对大文档做过多查询
_MAX_QUERY_CHUNKS = 3


def check_and_remove_duplicates(
    store: ChromaStore,
    user_id: str,
    new_doc_id: str,
    chunks: list[str],
    doc_type: str,
) -> dict[str, Any] | None:
    """
    入库后去重：用新文档的 chunks 查询已有文档，
    如果发现高相似度的旧文档，则删除旧文档。

    Returns:
        去重结果 dict (removed_doc_id, similarity, file_name) 或 None（无重复）。
    """
    if not settings.dedup_enabled or not chunks:
        return None

    threshold = settings.dedup_similarity_threshold
    query_chunks = chunks[:_MAX_QUERY_CHUNKS]

    # 收集所有匹配结果，按旧 doc_id 分组
    doc_scores: dict[str, list[float]] = defaultdict(list)
    doc_meta: dict[str, dict] = {}

    for chunk_text in query_chunks:
        results = store.query(
            query_text=chunk_text,
            top_k=5,
            where={"user_id": user_id},
        )
        for hit in results:
            hit_doc_id = hit["metadata"].get("doc_id", "")
            if hit_doc_id == new_doc_id:
                continue  # 跳过自身
            doc_scores[hit_doc_id].append(hit["score"])
            if hit_doc_id not in doc_meta:
                doc_meta[hit_doc_id] = hit["metadata"]

    if not doc_scores:
        return None

    # 找平均相似度最高的旧文档
    best_doc_id = None
    best_avg_score = 0.0
    for doc_id, scores in doc_scores.items():
        avg = sum(scores) / len(scores)
        if avg > best_avg_score:
            best_avg_score = avg
            best_doc_id = doc_id

    if best_avg_score < threshold or best_doc_id is None:
        logger.info(
            "去重: 未发现重复（最高相似度 %.3f < 阈值 %.2f）",
            best_avg_score, threshold,
        )
        return None

    # 删除旧文档
    old_meta = doc_meta[best_doc_id]
    old_file_name = old_meta.get("file_name", "")
    old_doc_type = old_meta.get("doc_type", "")

    logger.info(
        "去重: 发现重复文档 %s (%s)，平均相似度 %.3f，删除旧文档",
        best_doc_id, old_file_name, best_avg_score,
    )
    store.delete_by_doc_id(best_doc_id)

    # 清理 resume_cache（如果旧文档是 resume 类型）
    if old_doc_type == "resume":
        try:
            import json
            from app.store.resume_cache import _user_cache_path
            cache_path = _user_cache_path(user_id)
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if data.get("doc_id") == best_doc_id:
                    cache_path.unlink(missing_ok=True)
                    logger.info("去重: 清理了旧 resume 缓存 user=%s", user_id)
        except Exception as e:
            logger.warning("去重: 清理 resume 缓存失败: %s", e)

    return {
        "removed_doc_id": best_doc_id,
        "similarity": round(best_avg_score, 3),
        "file_name": old_file_name,
    }
