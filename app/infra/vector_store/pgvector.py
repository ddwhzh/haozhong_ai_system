"""PGVector store for embedding-based retrieval.

Provides async similarity search, upsert, and hybrid (dense + sparse)
retrieval primitives used by the retrieval agent's breadth pipeline.
"""

from __future__ import annotations

import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.infra.database.connection import connection_manager


class VectorStore:
    """Async PGVector operations backed by the shared connection pool."""

    def __init__(self, collection_name: str = "documents") -> None:
        self.collection_name = collection_name

    @staticmethod
    def _build_filter_clause(metadata_filter: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Build SQL filter clause for metadata JSONB fields."""
        filter_clause = ""
        filter_params: List[str] = []

        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>'{key}' = %s")
                filter_params.append(str(value))
            filter_clause = "WHERE " + " AND ".join(conditions)

        return filter_clause, filter_params

    @staticmethod
    def _extract_sparse_terms(query_text: str, max_terms: int = 8) -> List[str]:
        """Extract lexical terms for sparse retrieval.

        Uses a lightweight regex split strategy to support both zh/en queries.
        """
        if not query_text:
            return []

        normalized = re.sub(r"[，。！？、,:;；()\[\]{}\-_/]+", " ", query_text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []

        terms: List[str] = []
        terms.extend(re.findall(r"[A-Za-z0-9_]{3,}", normalized))

        zh_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
        split_pattern = (
            r"(能不能|是否|可不可以|可否|会不会|打不打得过|打得过|打不过|打过|"
            r"谁更强|谁更厉害|强不强|厉不厉害|和|与|跟|对|比|及|还是|或者|并且|以及)"
        )
        for chunk in zh_chunks:
            parts = re.split(split_pattern, chunk)
            for part in parts:
                if not part:
                    continue
                if re.fullmatch(split_pattern, part):
                    continue
                part = part.strip()
                if len(part) >= 2:
                    terms.append(part)

        seen = set()
        unique_terms: List[str] = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        if not unique_terms and normalized:
            unique_terms = [normalized]

        return unique_terms[:max_terms]

    @staticmethod
    def _fuse_hybrid_results(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse dense+sparse result lists via weighted RRF."""
        fused: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            doc_id = item.get("id")
            if not doc_id:
                continue
            fused[doc_id] = {
                "id": doc_id,
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "dense_score": float(item.get("score", 0.0) or 0.0),
                "sparse_score": 0.0,
                "dense_rank": rank,
                "sparse_rank": None,
            }

        for rank, item in enumerate(sparse_results, start=1):
            doc_id = item.get("id")
            if not doc_id:
                continue
            entry = fused.get(doc_id)
            if entry is None:
                entry = {
                    "id": doc_id,
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "dense_score": 0.0,
                    "sparse_score": float(item.get("score", 0.0) or 0.0),
                    "dense_rank": None,
                    "sparse_rank": rank,
                }
                fused[doc_id] = entry
            else:
                entry["sparse_score"] = float(item.get("score", 0.0) or 0.0)
                entry["sparse_rank"] = rank

        def _rrf(rank: Optional[int], weight: float) -> float:
            if rank is None:
                return 0.0
            return weight * (1.0 / (rrf_k + rank))

        items: List[Dict[str, Any]] = []
        for _, entry in fused.items():
            fused_score = _rrf(entry["dense_rank"], dense_weight) + _rrf(entry["sparse_rank"], sparse_weight)
            entry["score"] = fused_score
            entry["retrieval_mode"] = "hybrid"
            items.append(entry)

        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return items[:top_k]

    async def ensure_table(self) -> None:
        """Create the vector table + index if not present."""
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({settings.EMBEDDING_DIMENSION}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # IVFFlat index supports at most 2000 dimensions.
            # For larger dimensions (e.g. 2048), skip index creation and fall back
            # to exact search to avoid runtime errors.
            if settings.EMBEDDING_DIMENSION <= 2000:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_embedding
                    ON {self.collection_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            else:
                logger.warning(
                    "vector_index_skipped_due_to_dimension",
                    table=self.collection_name,
                    embedding_dimension=settings.EMBEDDING_DIMENSION,
                )
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_content_trgm
                ON {self.collection_name}
                USING gin (content gin_trgm_ops)
            """)
            logger.info("vector_table_ensured", table=self.collection_name)

    async def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a document with its embedding."""
        import json as _json
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.collection_name} (id, content, embedding, metadata)
                VALUES (%s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (doc_id, content, str(embedding), _json.dumps(metadata or {})),
            )

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search.

        Returns list of {id, content, score, metadata} dicts ordered by
        descending similarity.
        """
        filter_clause, filter_params = self._build_filter_clause(metadata_filter)

        emb_str = str(query_embedding)

        query = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.collection_name}
            {filter_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # params: emb (for score calc), filter_params, emb (for ORDER BY), top_k
        params = [emb_str] + filter_params + [emb_str, top_k]
        try:
            pool = await connection_manager.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    results = []
                    for row in rows:
                        score = float(row[3]) if row[3] else 0.0
                        if score >= score_threshold:
                            results.append({
                                "id": row[0],
                                "content": row[1],
                                "metadata": row[2],
                                "score": score,
                            })
                    return results
        except Exception as e:
            logger.error(
                "vector_search_failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return []

    async def sparse_search(
        self,
        query_text: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
        max_terms: int = 8,
    ) -> List[Dict[str, Any]]:
        """Lexical sparse retrieval using SQL ILIKE keyword matching."""
        terms = self._extract_sparse_terms(query_text=query_text, max_terms=max_terms)
        if not terms:
            return []

        score_parts: List[str] = []
        score_params: List[str] = []
        match_parts: List[str] = []
        match_params: List[str] = []

        for term in terms:
            pattern = f"%{term}%"
            score_parts.append("CASE WHEN content ILIKE %s THEN 1 ELSE 0 END")
            score_params.append(pattern)
            match_parts.append("content ILIKE %s")
            match_params.append(pattern)

        score_expr = f"(({ ' + '.join(score_parts) })::float / {len(terms)})"
        filter_clause, filter_params = self._build_filter_clause(metadata_filter)
        keyword_clause = "(" + " OR ".join(match_parts) + ")"
        if filter_clause:
            filter_clause = f"{filter_clause} AND {keyword_clause}"
        else:
            filter_clause = f"WHERE {keyword_clause}"

        query = f"""
            SELECT id, content, metadata,
                   {score_expr} AS score
            FROM {self.collection_name}
            {filter_clause}
            ORDER BY score DESC, created_at DESC
            LIMIT %s
        """
        params = score_params + filter_params + match_params + [top_k]
        try:
            pool = await connection_manager.get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    results = []
                    for row in rows:
                        score = float(row[3]) if row[3] else 0.0
                        if score >= score_threshold:
                            results.append({
                                "id": row[0],
                                "content": row[1],
                                "metadata": row[2],
                                "score": score,
                                "retrieval_mode": "sparse",
                            })
                    return results
        except Exception as e:
            logger.error(
                "sparse_search_failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return []

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
        candidate_k: int = 30,
        sparse_max_terms: int = 8,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: dense vector + sparse lexical fused with weighted RRF."""
        candidate_limit = max(top_k, candidate_k)
        dense_results = await self.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_limit,
            metadata_filter=metadata_filter,
            score_threshold=0.0,
        )
        sparse_results = await self.sparse_search(
            query_text=query_text,
            top_k=candidate_limit,
            metadata_filter=metadata_filter,
            score_threshold=0.0,
            max_terms=sparse_max_terms,
        )
        fused_results = self._fuse_hybrid_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )
        if score_threshold <= 0.0:
            return fused_results
        return [r for r in fused_results if float(r.get("score", 0.0)) >= score_threshold]

    async def batch_upsert(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Batch upsert documents. Each dict needs: id, content, embedding, metadata."""
        count = 0
        for doc in documents:
            try:
                await self.upsert(
                    doc_id=doc["id"],
                    content=doc["content"],
                    embedding=doc["embedding"],
                    metadata=doc.get("metadata"),
                )
                count += 1
            except Exception as e:
                logger.error("batch_upsert_item_failed", doc_id=doc.get("id"), error=str(e))
        return count

    async def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"DELETE FROM {self.collection_name} WHERE id = %s",
                (doc_id,),
            )


# Default singleton
vector_store = VectorStore(collection_name=settings.VECTOR_COLLECTION_NAME)
