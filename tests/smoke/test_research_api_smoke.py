"""Smoke tests for Research API endpoints.

These tests verify API surface and response shape with mocked research graph.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.types.research_state import ResearchPhase


def _mock_research_state() -> dict:
    """Build a mock final research state."""
    return {
        "research_topic": "How to improve RAG for multi-hop reasoning",
        "current_phase": ResearchPhase.CONCLUDE.value,
        "iteration_count": 2,
        "max_iterations": 5,
        "literature": [
            {"title": "Paper A", "year": 2025, "source": "arxiv"},
            {"title": "Paper B", "year": 2024, "source": "semantic_scholar"},
        ],
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "statement": "Graph-augmented RAG improves recall",
                "status": "confirmed",
                "confidence": 0.85,
            },
        ],
        "experiments": [
            {
                "experiment_id": "e1",
                "name": "graph_rag_recall_test",
                "status": "completed",
            },
        ],
        "research_journal": [
            {"phase": "scout", "summary": "Explored domain"},
            {"phase": "survey", "summary": "Retrieved 10 papers"},
        ],
        "final_report": "# Research Report\n\nGraph-augmented RAG improves recall by 12%.",
        "domain_context": {"keywords": ["RAG", "multi-hop"]},
        "synthesis_report": "RAG benefits from graph structure.",
        "analysis_report": "Experiment confirms hypothesis.",
        "metadata": {"session_id": "test-session"},
    }


@pytest.mark.asyncio
async def test_research_start_endpoint():
    """POST /api/v1/research/start 返回正确格式."""
    mock_state = _mock_research_state()

    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_state)

    with patch(
        "app.presentation.api.v1.research.research_graph"
    ) as mock_rg:
        mock_rg.create = AsyncMock(return_value=mock_graph)

        with patch(
            "app.presentation.api.v1.research.get_langfuse_callback",
            return_value=None,
        ):
            from app.main import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    "/api/v1/research/start",
                    json={
                        "topic": "How to improve RAG for multi-hop reasoning",
                        "max_iterations": 2,
                    },
                )

    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["research_topic"] == "How to improve RAG for multi-hop reasoning"
    assert data["final_report"] != ""
    assert isinstance(data["hypotheses"], list)
    assert isinstance(data["experiments"], list)


@pytest.mark.asyncio
async def test_research_start_validation():
    """POST /api/v1/research/start 空topic返回422."""
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/research/start",
            json={"topic": "ab"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_research_status_not_found():
    """GET /api/v1/research/{id}/status 不存在会话返回404."""
    mock_graph = AsyncMock()
    mock_graph.aget_state = AsyncMock(return_value=None)

    with patch(
        "app.presentation.api.v1.research.research_graph"
    ) as mock_rg:
        mock_rg.create = AsyncMock(return_value=mock_graph)

        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/api/v1/research/nonexistent-session/status")

    assert response.status_code == 404
