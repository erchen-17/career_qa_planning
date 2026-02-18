"""
Text chunking for Chinese + English mixed documents.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chinese-aware separators (ordered from most to least significant)
_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    with Chinese-aware separators.

    Returns a list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,  # character-based for Chinese
    )
    chunks = splitter.split_text(text)
    return chunks
