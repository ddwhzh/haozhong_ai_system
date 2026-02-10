"""Unit tests for PGVector ensure_table behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infra.vector_store import pgvector as pgvector_module
from app.infra.vector_store.pgvector import VectorStore


class _FakeConnContext:
    """Async context manager wrapper for a mocked DB connection."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False


class _FakePool:
    """Pool stub exposing connection() async context manager."""

    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        return _FakeConnContext(self._conn)


def _executed_sqls(mock_conn: AsyncMock) -> list[str]:
    """Collect executed SQL strings from AsyncMock calls."""
    sqls = []
    for call in mock_conn.execute.await_args_list:
        if call.args:
            sqls.append(call.args[0])
    return sqls


class TestVectorStoreEnsureTable:
    """Verify ivfflat index creation guard by embedding dimension."""

    @pytest.mark.asyncio
    async def test_create_ivfflat_index_when_dimension_supported(self, monkeypatch):
        """EMBEDDING_DIMENSION<=2000时应创建ivfflat索引."""
        mock_conn = AsyncMock()
        fake_pool = _FakePool(mock_conn)

        monkeypatch.setattr(
            pgvector_module.connection_manager,
            "get_pool",
            AsyncMock(return_value=fake_pool),
        )
        monkeypatch.setattr(pgvector_module.settings, "EMBEDDING_DIMENSION", 1536)

        store = VectorStore(collection_name="documents_unit_test")
        await store.ensure_table()

        sqls = _executed_sqls(mock_conn)
        assert any("USING ivfflat" in s for s in sqls)
        assert any("CREATE EXTENSION IF NOT EXISTS pg_trgm" in s for s in sqls)
        assert any("gin_trgm_ops" in s for s in sqls)

    @pytest.mark.asyncio
    async def test_skip_ivfflat_index_when_dimension_too_large(self, monkeypatch):
        """EMBEDDING_DIMENSION>2000时应跳过ivfflat并记录告警."""
        mock_conn = AsyncMock()
        fake_pool = _FakePool(mock_conn)
        warn_mock = MagicMock()

        monkeypatch.setattr(
            pgvector_module.connection_manager,
            "get_pool",
            AsyncMock(return_value=fake_pool),
        )
        monkeypatch.setattr(pgvector_module.settings, "EMBEDDING_DIMENSION", 2048)
        monkeypatch.setattr(pgvector_module.logger, "warning", warn_mock)

        store = VectorStore(collection_name="documents_unit_test")
        await store.ensure_table()

        sqls = _executed_sqls(mock_conn)
        assert not any("USING ivfflat" in s for s in sqls)
        assert any("CREATE EXTENSION IF NOT EXISTS pg_trgm" in s for s in sqls)
        assert any("gin_trgm_ops" in s for s in sqls)
        warn_mock.assert_called_once()
        assert warn_mock.call_args.kwargs["embedding_dimension"] == 2048


class TestVectorStoreHybridFusion:
    """Verify dense+sparse hybrid rank fusion behavior."""

    def test_extract_sparse_terms_supports_zh_comparison(self):
        terms = VectorStore._extract_sparse_terms("鹿紫云能不能打过五条悟", max_terms=8)
        assert "鹿紫云" in terms
        assert "五条悟" in terms

    def test_fuse_hybrid_results_prefers_dual_hit_documents(self):
        dense_results = [
            {"id": "a", "content": "doc-a", "metadata": {}, "score": 0.91},
            {"id": "b", "content": "doc-b", "metadata": {}, "score": 0.88},
        ]
        sparse_results = [
            {"id": "b", "content": "doc-b", "metadata": {}, "score": 1.0},
            {"id": "c", "content": "doc-c", "metadata": {}, "score": 0.8},
        ]
        fused = VectorStore._fuse_hybrid_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=3,
            dense_weight=0.6,
            sparse_weight=0.4,
            rrf_k=60,
        )
        assert fused
        assert fused[0]["id"] == "b"
        assert fused[0]["retrieval_mode"] == "hybrid"
        assert fused[0]["dense_rank"] == 2
        assert fused[0]["sparse_rank"] == 1
