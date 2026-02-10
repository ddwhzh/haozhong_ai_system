"""Smoke tests for API endpoints.

These tests verify API surface and response shape with mocked orchestration.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.types.belief import Belief
from app.core.types.evidence import EdgeRelation, Evidence, EvidenceGraph
from app.core.types.intent import Intent, IntentType


def _build_graph_with_web_result() -> EvidenceGraph:
    """Create a small graph including one Brave web_result node."""
    graph = EvidenceGraph()

    query_node = graph.add_evidence(
        evidence=Evidence(
            content="鹿自云一能不能打过五条悟",
            content_type="query",
            source="user",
        ),
        belief=Belief(
            content="user query",
            confidence=1.0,
            source_agent="retrieval_agent",
        ),
        intent=Intent(
            intent_type=IntentType.RETRIEVE,
            description="input query",
            source_agent="retrieval_agent",
        ),
        tags=["query"],
    )

    graph.add_evidence(
        evidence=Evidence(
            content="示例标题\n示例摘要",
            content_type="web_result",
            source="https://example.com",
            metadata={
                "title": "示例标题",
                "url": "https://example.com",
                "depends_on": [1, 2],
                "rank": 1,
            },
        ),
        belief=Belief(
            content="web result",
            confidence=0.77,
            source_agent="retrieval_agent",
        ),
        intent=Intent(
            intent_type=IntentType.RETRIEVE,
            description="brave web retrieval",
            source_agent="retrieval_agent",
        ),
        parent_node_ids=[query_node.node_id],
        relation=EdgeRelation.DERIVED_FROM,
        tags=["web_result", "brave_search"],
    )

    return graph


class TestRootEndpointSmoke:
    """Smoke: root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_system_info(self):
        """根端点返回系统信息."""
        with patch("app.infra.neo4j.client.AsyncGraphDatabase"):
            from app.main import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "agents" in data
        assert "retrieval_agent" in data["agents"]


class TestHealthEndpointSmoke:
    """Smoke: health endpoint."""

    @pytest.mark.asyncio
    async def test_health_degraded_without_services(self):
        """无外部服务时健康检查返回degraded."""
        with patch("app.infra.neo4j.client.AsyncGraphDatabase"):
            from app.main import app
            from app.infra.database import database_service
            from app.infra.neo4j import neo4j_client

            database_service.health_check = AsyncMock(return_value=False)
            neo4j_client.health_check = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert "components" in data


class TestChatEndpointSmoke:
    """Smoke: chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """chat端点mock管道后返回有效响应."""
        with patch("app.infra.neo4j.client.AsyncGraphDatabase"):
            from app.main import app

            with patch("app.presentation.api.v1.agent.pipeline_graph") as mock_pg:
                mock_graph = AsyncMock()
                mock_pg.create = AsyncMock(return_value=mock_graph)
                mock_graph.ainvoke = AsyncMock(return_value={
                    "final_output": "这是测试回答",
                    "evidence_graph": EvidenceGraph(),
                    "agent_outputs": [],
                    "current_phase": "done",
                })
                with patch("app.presentation.api.v1.agent._run_in_presentation_pool", new_callable=AsyncMock) as mock_runner:
                    async def _passthrough(func, *args):
                        return func(*args)

                    mock_runner.side_effect = _passthrough

                    transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                        response = await client.post(
                            "/api/v1/agent/chat",
                            json={"query": "测试查询"},
                        )
                    assert mock_runner.await_count >= 3

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "这是测试回答"
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_chat_intermediate_steps_keep_list_metadata(self):
        """web_result metadata中的列表字段应保持为list类型."""
        with patch("app.infra.neo4j.client.AsyncGraphDatabase"):
            from app.main import app

            with patch("app.presentation.api.v1.agent.pipeline_graph") as mock_pg:
                mock_graph = AsyncMock()
                mock_pg.create = AsyncMock(return_value=mock_graph)
                mock_graph.ainvoke = AsyncMock(return_value={
                    "final_output": "ok",
                    "evidence_graph": _build_graph_with_web_result(),
                    "agent_outputs": [],
                    "current_phase": "done",
                })

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/agent/chat",
                        json={"query": "测试web metadata"},
                    )

        assert response.status_code == 200
        payload = response.json()
        web_step = next(
            (s for s in payload["intermediate_steps"] if "Brave Web检索结果" in s["label"]),
            None,
        )
        assert web_step is not None
        node = web_step["nodes"][0]
        assert isinstance(node["metadata"].get("depends_on"), list)
        assert node["metadata"]["depends_on"] == [1, 2]

    @pytest.mark.asyncio
    async def test_chat_rejects_empty_query(self):
        """空查询应返回422验证错误."""
        with patch("app.infra.neo4j.client.AsyncGraphDatabase"):
            from app.main import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/agent/chat",
                    json={"query": ""},
                )

        assert response.status_code == 422
