"""Brave Web Retriever — optional web search augmentation for retrieval stage.

This retriever uses Brave Search API as an additional source alongside
vector (RAG) and KG retrieval. Results are wrapped into EvidenceNodes
for downstream cascade fusion.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType
from app.infra.external import BraveSearchClient
from app.infra.vector_store import vector_store


class BraveWebRetriever:
    """Brave web search retriever."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.enabled = bool(settings.BRAVE_SEARCH_ENABLED and settings.BRAVE_SEARCH_API_KEY)
        self.auto_index_enabled = bool(
            self.enabled and settings.BRAVE_AUTO_INDEX_TO_VECTOR
        )
        self._table_ensured = False
        self._client = BraveSearchClient(
            api_key=settings.BRAVE_SEARCH_API_KEY,
            base_url=settings.BRAVE_SEARCH_BASE_URL,
            timeout_seconds=settings.BRAVE_SEARCH_TIMEOUT_SECONDS,
        )
        embed_kwargs = {
            "model": settings.EMBEDDING_MODEL,
            "api_key": settings.OPENAI_API_KEY,
        }
        if settings.OPENAI_API_BASE:
            embed_kwargs["base_url"] = settings.OPENAI_API_BASE
        self._embeddings = OpenAIEmbeddings(**embed_kwargs)

        if settings.BRAVE_SEARCH_ENABLED and not settings.BRAVE_SEARCH_API_KEY:
            logger.warning("brave_search_enabled_without_key")

    async def retrieve(
        self,
        sub_query_nodes: List[EvidenceNode],
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """Retrieve web results for sub-query/tree-step nodes."""
        if not self.enabled:
            return []

        all_web_nodes: List[EvidenceNode] = []
        retrieval_types = ("sub_query", "tree_step")

        for sq_node in sub_query_nodes:
            if sq_node.evidence.content_type not in retrieval_types:
                continue

            query_text = sq_node.evidence.content
            results = await self._client.web_search(query=query_text, count=self.top_k)
            if not results:
                continue

            for item in results:
                rank = int(item.get("rank", 0))
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("description", "")
                # Heuristic confidence decays with rank.
                confidence = max(0.35, 0.9 - rank * 0.08)
                content = (title + "\n" + snippet).strip()

                node = evidence_graph.add_evidence(
                    evidence=Evidence(
                        content=content,
                        content_type="web_result",
                        source=url,
                        metadata={
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "rank": rank,
                            "query": query_text,
                            "engine": "brave_search",
                        },
                    ),
                    belief=Belief(
                        content=f"Brave web result rank={rank}: {title}",
                        confidence=confidence,
                        source_agent="retrieval_agent",
                    ),
                    intent=Intent(
                        intent_type=IntentType.RETRIEVE,
                        description=f"Brave web search result (rank={rank})",
                        parameters={"rank": rank, "engine": "brave_search"},
                        source_agent="retrieval_agent",
                    ),
                    parent_node_ids=[sq_node.node_id],
                    relation=EdgeRelation.DERIVED_FROM,
                    tags=["web_result", "brave_search", f"rank_{rank}"],
                )
                all_web_nodes.append(node)

            # Optionally persist top web results into vector DB for future reuse.
            if self.auto_index_enabled:
                await self._index_results_to_vector(query_text=query_text, results=results)

        logger.info("web_retrieval_complete", total_results=len(all_web_nodes))
        return all_web_nodes

    async def _index_results_to_vector(
        self,
        query_text: str,
        results: List[dict],
    ) -> None:
        """Persist Brave results into PGVector for later RAG retrieval."""
        if not results:
            return

        if not self._table_ensured:
            await vector_store.ensure_table()
            self._table_ensured = True

        max_items = max(0, settings.BRAVE_AUTO_INDEX_MAX_ITEMS)
        min_chars = max(1, settings.BRAVE_AUTO_INDEX_MIN_CONTENT_CHARS)

        items = results[:max_items]
        documents = []
        for item in items:
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("description") or "").strip()
            rank = int(item.get("rank", 0))
            content = (title + "\n" + snippet).strip()
            if len(content) < min_chars:
                continue

            doc_id = self._build_doc_id(url=url, query=query_text, title=title)
            metadata = {
                "source_type": "brave_search",
                "engine": "brave_search",
                "url": url,
                "title": title,
                "rank": rank,
                "query": query_text,
            }
            documents.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                }
            )

        if not documents:
            return

        texts = [d["content"] for d in documents]
        embeddings = await self._embeddings.aembed_documents(texts)
        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb

        inserted = await vector_store.batch_upsert(documents)
        logger.info(
            "brave_auto_index_complete",
            query=query_text[:120],
            indexed=inserted,
            attempted=len(documents),
        )

    @staticmethod
    def _build_doc_id(url: str, query: str, title: str) -> str:
        """Build stable doc id for Brave result upsert."""
        base = url or (query + "::" + title)
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]
        return f"brave-{digest}"
