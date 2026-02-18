"""
Prompt templates for RAG answer generation.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
你是一个专业的职业规划助手。你的任务是基于用户的简历信息和职业介绍资料，为用户提供个性化的职业规划建议。

规则：
1. 仅基于提供的上下文信息回答，不要编造内容。
2. 如果上下文信息不足以回答问题，请如实告知用户。
3. 回答中引用的关键信息需标注来源（如"根据您的简历…"或"根据职业介绍资料…"）。
4. 回答应当具体、有针对性，避免泛泛而谈。
5. 使用中文回答。
"""

CONTEXT_TEMPLATE = """\
以下是检索到的相关资料片段，请基于这些内容回答用户的问题。

{context_blocks}

---
用户问题：{query}
"""

PINNED_RESUME_TEMPLATE = """\
以下是用户的完整简历内容（作为固定上下文）：

<resume>
{resume_text}
</resume>
"""


def format_context_blocks(chunks: list[dict]) -> str:
    """Format retrieved chunks into numbered context blocks."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc_type_label = "简历" if meta.get("doc_type") == "resume" else "职业介绍"
        source = f"[来源{i}: {doc_type_label} - {meta.get('file_name', '未知')} 第{meta.get('page', '?')}页]"
        blocks.append(f"{source}\n{chunk['content']}")
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    chunks: list[dict],
    pinned_resume_text: str | None = None,
) -> list[dict]:
    """Build the LangChain message list for the chat model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add pinned resume if available
    if pinned_resume_text:
        messages.append({
            "role": "system",
            "content": PINNED_RESUME_TEMPLATE.format(resume_text=pinned_resume_text),
        })

    # Add context + query
    context_str = format_context_blocks(chunks) if chunks else "（未检索到相关资料）"
    user_msg = CONTEXT_TEMPLATE.format(context_blocks=context_str, query=query)
    messages.append({"role": "user", "content": user_msg})

    return messages
