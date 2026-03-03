"""
Chroma vector store wrapper with metadata-filtered search.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper around Chroma for add / query with metadata filters."""

    def __init__(self):
        emb_api_key = settings.embedding_api_key or settings.openai_api_key
        emb_base_url = settings.embedding_base_url or settings.openai_base_url

        self._embedding_fn = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=emb_api_key,
            **({"base_url": emb_base_url} if emb_base_url else {}),
        )
        persist_dir = str(settings.chroma_persist_path)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.info("ChromaStore initialised – persist_dir=%s", persist_dir)

    # ---- write ----

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Embed and store text chunks with metadata. Returns stored IDs."""
        doc_ids = self._vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d chunks to Chroma", len(doc_ids))
        return doc_ids

    # ---- read ----

    def query(
        self,
        query_text: str,
        top_k: int = 6,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Similarity search with optional metadata filter.

        Returns a list of dicts:
          {"content": str, "metadata": dict, "score": float}
        """
        results = self._vectorstore.similarity_search_with_relevance_scores(
            query=query_text,
            k=top_k,
            filter=where,
        )
        out = []
        for doc, score in results:
            out.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
            })
        return out

    # ---- utils ----

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a specific doc_id."""
        self._vectorstore.delete(where={"doc_id": doc_id})
        logger.info("Deleted chunks for doc_id=%s", doc_id)

    def count(self) -> int:
        """Return total number of stored embeddings."""
        collection = self._client.get_collection(settings.chroma_collection_name)
        return collection.count()


# Module-level singleton (lazy)
_store: ChromaStore | None = None


def get_chroma_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
