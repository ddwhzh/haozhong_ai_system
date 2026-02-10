"""Unit tests for orchestration helper nodes."""

import pytest

from app.core.types.belief import Belief
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode
from app.core.types.intent import Intent, IntentType
from app.core.types.state import PipelineState
from app.orchestration.nodes import _extract_query_keywords, assemble_output, freeze_context


def _node(
    *,
    content: str,
    content_type: str,
    source: str,
    source_agent: str,
    confidence: float = 0.8,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> EvidenceNode:
    """Create an EvidenceNode helper for tests."""
    return EvidenceNode(
        evidence=Evidence(
            content=content,
            content_type=content_type,
            source=source,
            metadata=metadata or {},
        ),
        belief=Belief(
            content=f"belief:{content_type}",
            confidence=confidence,
            source_agent=source_agent,
        ),
        intent=Intent(
            intent_type=IntentType.RETRIEVE,
            description=f"intent:{content_type}",
            source_agent=source_agent,
        ),
        tags=tags or [],
    )


class TestAssembleOutputDynamicGap:
    """Verify dynamic evidence-gap fallback behavior."""

    @pytest.mark.asyncio
    async def test_dynamic_gap_message_zh(self):
        """中文查询在证据不足时返回动态主题提示."""
        graph = EvidenceGraph()
        graph.add_node(_node(
            content="## 综合回答\n\n**问题:** 鹿自云一能不能打过五条悟",
            content_type="generated_content",
            source="content_generator",
            source_agent="generation_agent",
            metadata={"filled_count": 0},
            tags=["generated"],
        ))
        state = PipelineState(
            query="鹿自云一能不能打过五条悟",
            evidence_graph=graph,
        )

        result = await assemble_output(state)
        text = result["final_output"]

        assert "当前知识库中缺少与你问题直接相关的证据" in text
        assert "建议: 补充" in text
        assert "东北经济" not in text

    @pytest.mark.asyncio
    async def test_dynamic_gap_message_en(self):
        """英文查询在证据不足时返回英文建议."""
        graph = EvidenceGraph()
        graph.add_node(_node(
            content="## answer\n\nQ: can A beat B?",
            content_type="generated_content",
            source="content_generator",
            source_agent="generation_agent",
            metadata={"filled_count": 0},
            tags=["generated"],
        ))
        state = PipelineState(
            query="Can character A beat character B?",
            evidence_graph=graph,
        )

        result = await assemble_output(state)
        text = result["final_output"]

        assert "The current knowledge base lacks directly relevant evidence" in text
        assert "Suggestion: add more evidence about" in text

    def test_dynamic_gap_keywords_extract_entities_for_comparison_query(self):
        """关键词提取应把比较句拆成实体词, 避免整句提示."""
        keywords = _extract_query_keywords("鹿紫云能不能打过五条悟")
        assert "鹿紫云" in keywords
        assert "五条悟" in keywords
        assert "鹿紫云能不能打过五条悟" not in keywords


class TestFreezeContext:
    """Verify context manifest aggregation."""

    @pytest.mark.asyncio
    async def test_freeze_context_counts_web_nodes(self):
        """freeze_context应统计web_result数量."""
        graph = EvidenceGraph()
        graph.add_node(_node(
            content="doc chunk",
            content_type="document_chunk",
            source="vector_store",
            source_agent="retrieval_agent",
        ))
        graph.add_node(_node(
            content="web title\nweb snippet",
            content_type="web_result",
            source="https://example.com",
            source_agent="retrieval_agent",
        ))
        graph.add_node(_node(
            content="kg chain",
            content_type="reasoning_chain",
            source="neo4j",
            source_agent="retrieval_agent",
        ))

        state = PipelineState(query="测试查询", evidence_graph=graph)
        result = await freeze_context(state)
        manifest = result["context_manifest"]

        assert manifest["rag_count"] == 1
        assert manifest["web_count"] == 1
        assert manifest["kg_count"] == 1
        assert manifest["sufficient"] is True
