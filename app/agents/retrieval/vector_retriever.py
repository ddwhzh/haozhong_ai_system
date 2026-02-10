"""Vector Retriever — embedding-based similarity search for breadth expansion.

For each sub-query from the query decomposer, performs vector similarity
search against PGVector to find relevant document chunks.  Results are
wrapped as EvidenceNodes with belief + intent.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
from app.infra.vector_store import vector_store


class VectorRetriever:
    """Embedding-based retrieval using PGVector."""

    def __init__(
        self,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> None:
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.retrieval_mode = settings.RETRIEVAL_VECTOR_MODE if settings.RETRIEVAL_VECTOR_MODE in ("dense", "hybrid") else "dense"
        self.hybrid_dense_weight = settings.RETRIEVAL_HYBRID_DENSE_WEIGHT
        self.hybrid_sparse_weight = settings.RETRIEVAL_HYBRID_SPARSE_WEIGHT
        self.hybrid_rrf_k = settings.RETRIEVAL_HYBRID_RRF_K
        self.hybrid_candidates = settings.RETRIEVAL_HYBRID_CANDIDATES
        self.hybrid_max_terms = settings.RETRIEVAL_HYBRID_MAX_TERMS
        embed_kwargs = {
            "model": settings.EMBEDDING_MODEL,
            "api_key": settings.OPENAI_API_KEY,
        }
        if settings.OPENAI_API_BASE:
            embed_kwargs["base_url"] = settings.OPENAI_API_BASE
        self._embeddings = OpenAIEmbeddings(**embed_kwargs)

    async def retrieve(
        self,
        sub_query_nodes: List[EvidenceNode],
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """For each sub-query node, run vector similarity search.

        Returns:
            New EvidenceNodes representing retrieved document chunks.
        """
        all_vector_nodes: List[EvidenceNode] = []

        # Accept both classic "sub_query" and QDMR "tree_step" content types
        _retrieval_types = ("sub_query", "tree_step")
        for sq_node in sub_query_nodes:
            if sq_node.evidence.content_type not in _retrieval_types:
                continue

            query_text = sq_node.evidence.content
            try:
                nodes = await self._retrieve_for_query(
                    query_text, sq_node, evidence_graph
                )
                all_vector_nodes.extend(nodes)
            except Exception as e:
                logger.error(
                    "vector_retrieval_failed",
                    query=query_text,
                    error=str(e),
                )

        logger.info("vector_retrieval_complete", total_chunks=len(all_vector_nodes))
        return all_vector_nodes

    async def _retrieve_for_query(
        self,
        query: str,
        parent_node: EvidenceNode,
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """Run vector search for a single sub-query."""
        # Generate embedding
        query_embedding = await self._embeddings.aembed_query(query)

        if self.retrieval_mode == "hybrid":
            results = await vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
                dense_weight=self.hybrid_dense_weight,
                sparse_weight=self.hybrid_sparse_weight,
                rrf_k=self.hybrid_rrf_k,
                candidate_k=self.hybrid_candidates,
                sparse_max_terms=self.hybrid_max_terms,
            )
            if not results:
                logger.warning("hybrid_search_empty_fallback_to_dense", query=query[:120])

        else:
            results = []

        # Fallback to dense mode when hybrid disabled or hybrid misses.
        if not results:
            results = await vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
            )

        if not results:
            return []

        # Wrap each result as an EvidenceNode
        nodes: List[EvidenceNode] = []
        for i, result in enumerate(results):
            score = result.get("score", 0.0)

            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=result.get("content", ""),
                    content_type="document_chunk",
                    source=result.get("id", ""),
                    metadata={
                        "similarity_score": score,
                        "doc_metadata": result.get("metadata", {}),
                        "rank": i,
                        "retrieval_mode": result.get("retrieval_mode", self.retrieval_mode),
                        "dense_score": result.get("dense_score"),
                        "sparse_score": result.get("sparse_score"),
                        "dense_rank": result.get("dense_rank"),
                        "sparse_rank": result.get("sparse_rank"),
                    },
                ),
                belief=Belief(
                    content=f"Vector search result (score={score:.3f}): {result.get('content', '')[:80]}...",
                    confidence=score,
                    source_agent="retrieval_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.RETRIEVE,
                    description=f"Vector similarity search result (rank={i})",
                    parameters={"similarity_score": score, "rank": i},
                    source_agent="retrieval_agent",
                ),
                parent_node_ids=[parent_node.node_id],
                relation=EdgeRelation.DERIVED_FROM,
                tags=["vector_result", f"rank_{i}"],
            )
            nodes.append(node)

        return nodes
