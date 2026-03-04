"""
多查询扩展模块：通过 LLM 判断是否需要扩展查询，并生成多个子查询以提升召回率。
适用于比较型/探索型问题（如"A还是B"、"有哪些方向"等）。
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.llm.router import get_chat_model

logger = logging.getLogger(__name__)

# auto 模式：LLM 同时判断是否需要扩展 + 生成子查询
_MAYBE_EXPAND_PROMPT = """\
你是一个搜索查询改写助手。

背景：用户正在使用一个职业规划知识库。知识库中存储了多种不同岗位的介绍文档（每个文档描述一个具体岗位的职责、要求、发展路径等）。
你的任务是判断用户的问题是否需要检索多个不同岗位的文档，如果需要，生成对应的搜索子查询。

需要扩展的情况：
- 用户在比较或犹豫多个方向（如"A还是B"、"该选哪个"）
- 用户提到了模糊范围（如"其他方向"、"还有什么选择"、"有哪些岗位"）
- 回答问题需要参考多个岗位的信息

不需要扩展的情况：
- 只问某一个具体岗位的信息
- 只问用户自身背景/简历相关的问题

如果不需要扩展，只回复：NO_EXPAND

如果需要扩展，输出 {max_queries} 个搜索子查询，每行一个。
每个子查询应该对应某个或某多个相关的岗位类别，写法像是在搜索该岗位的介绍。

用户问题：{query}
"""

# always 模式：强制生成子查询，不含判断逻辑
_FORCE_EXPAND_PROMPT = """\
你是一个搜索查询改写助手。

背景：用户正在使用一个职业规划知识库。知识库中存储了多种不同岗位的介绍文档（每个文档描述一个具体岗位的职责、要求、发展路径等）。
你的任务是将用户的问题拆解为多个搜索子查询，每个子查询对应知识库中一个具体岗位的文档。

请生成 {max_queries} 个搜索子查询，每行一个。
每个子查询的写法应该像是在搜索某个具体岗位的介绍文档。
例如：
- 大模型算法工程师岗位介绍
- 计算机视觉算法工程师岗位介绍
- 推荐系统算法工程师岗位要求

子查询之间应覆盖不同的岗位方向，不要重复。
不要输出编号、不要解释、不要输出抽象的分析性查询。

用户问题：{query}
"""


def _parse_sub_queries(text: str, max_queries: int) -> list[str]:
    """解析 LLM 返回的子查询文本，按行分割并过滤。"""
    if "NO_EXPAND" in text.upper():
        return []
    lines = [
        line.strip().lstrip("0123456789.-、) ")
        for line in text.strip().split("\n")
    ]
    sub_queries = [line for line in lines if line and len(line) > 2]
    return sub_queries[:max_queries]


async def maybe_expand(
    query: str,
    provider: str,
    model: str,
    max_queries: int = 3,
) -> list[str]:
    """
    auto 模式：一次 LLM 调用同时判断是否需要扩展 + 生成子查询。
    返回空列表表示不需要扩展。
    """
    llm = get_chat_model(provider=provider, model=model, temperature=0.3, max_tokens=512)
    prompt = _MAYBE_EXPAND_PROMPT.format(max_queries=max_queries, query=query)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return _parse_sub_queries(response.content, max_queries)


async def force_expand(
    query: str,
    provider: str,
    model: str,
    max_queries: int = 3,
) -> list[str]:
    """
    always 模式：强制生成子查询，不含判断逻辑。
    """
    llm = get_chat_model(provider=provider, model=model, temperature=0.3, max_tokens=512)
    prompt = _FORCE_EXPAND_PROMPT.format(max_queries=max_queries, query=query)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return _parse_sub_queries(response.content, max_queries)


def merge_results(
    results_per_query: list[list[dict]],
    top_k: int,
) -> list[dict]:
    """
    合并多个子查询的检索结果，去重后按相关性排序。
    去重依据：(doc_id, chunk_id) 元数据对。
    """
    seen: set[tuple[str, int]] = set()
    merged: list[dict] = []

    for results in results_per_query:
        for chunk in results:
            meta = chunk.get("metadata", {})
            key = (meta.get("doc_id", ""), meta.get("chunk_id", 0))
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

    # 按 score 升序排列（ChromaDB 距离越小越相似）
    merged.sort(key=lambda c: c.get("score", float("inf")))
    return merged[:top_k]
