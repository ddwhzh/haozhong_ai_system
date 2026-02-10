"""Unit tests for Evidence Graph."""

import pytest

from app.core.types.belief import Belief
from app.core.types.evidence import (
    Evidence,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType


def _make_node(
    content: str = "test",
    agent: str = "test_agent",
    confidence: float = 0.8,
    intent_type: IntentType = IntentType.RETRIEVE,
) -> EvidenceNode:
    """Helper: create a test EvidenceNode."""
    return EvidenceNode(
        evidence=Evidence(content=content, source="test"),
        belief=Belief(content=content, confidence=confidence, source_agent=agent),
        intent=Intent(intent_type=intent_type, source_agent=agent),
    )


class TestEvidenceGraph:
    """Tests for EvidenceGraph container."""

    def test_empty_graph(self):
        """空图初始状态."""
        g = EvidenceGraph()
        assert g.node_count == 0
        assert g.edge_count == 0
        assert g.get_leaf_nodes() == []

    def test_add_node(self):
        """添加节点."""
        g = EvidenceGraph()
        node = _make_node("hello")
        result = g.add_node(node)
        assert result.node_id == node.node_id
        assert g.node_count == 1
        assert node.node_id in g.nodes

    def test_add_edge_valid(self):
        """添加有效边(两端节点已存在)."""
        g = EvidenceGraph()
        n1 = _make_node("parent")
        n2 = _make_node("child")
        g.add_node(n1)
        g.add_node(n2)

        edge = EvidenceEdge(
            source_node_id=n1.node_id,
            target_node_id=n2.node_id,
            relation=EdgeRelation.DERIVED_FROM,
        )
        g.add_edge(edge)
        assert g.edge_count == 1

    def test_add_edge_missing_source_raises(self):
        """添加边时源节点不存在应报错."""
        g = EvidenceGraph()
        n2 = _make_node("child")
        g.add_node(n2)

        edge = EvidenceEdge(
            source_node_id="nonexistent",
            target_node_id=n2.node_id,
            relation=EdgeRelation.DERIVED_FROM,
        )
        with pytest.raises(ValueError, match="Source node"):
            g.add_edge(edge)

    def test_add_edge_missing_target_raises(self):
        """添加边时目标节点不存在应报错."""
        g = EvidenceGraph()
        n1 = _make_node("parent")
        g.add_node(n1)

        edge = EvidenceEdge(
            source_node_id=n1.node_id,
            target_node_id="nonexistent",
            relation=EdgeRelation.DERIVED_FROM,
        )
        with pytest.raises(ValueError, match="Target node"):
            g.add_edge(edge)

    def test_add_evidence_convenience(self):
        """add_evidence一次性创建节点+边."""
        g = EvidenceGraph()
        parent = _make_node("parent")
        g.add_node(parent)

        child = g.add_evidence(
            evidence=Evidence(content="child", source="test"),
            belief=Belief(content="child", confidence=0.7, source_agent="a"),
            intent=Intent(intent_type=IntentType.RETRIEVE, source_agent="a"),
            parent_node_ids=[parent.node_id],
            relation=EdgeRelation.DERIVED_FROM,
        )
        assert g.node_count == 2
        assert g.edge_count == 1
        assert child.node_id in g.nodes

    def test_get_nodes_by_agent(self):
        """按agent名称筛选节点."""
        g = EvidenceGraph()
        g.add_node(_make_node("a", agent="agent_a"))
        g.add_node(_make_node("b", agent="agent_b"))
        g.add_node(_make_node("c", agent="agent_a"))

        result = g.get_nodes_by_agent("agent_a")
        assert len(result) == 2

    def test_get_nodes_by_intent(self):
        """按intent类型筛选节点."""
        g = EvidenceGraph()
        g.add_node(_make_node("r", intent_type=IntentType.RETRIEVE))
        g.add_node(_make_node("g", intent_type=IntentType.GENERATE))
        g.add_node(_make_node("r2", intent_type=IntentType.RETRIEVE))

        result = g.get_nodes_by_intent("retrieve")
        assert len(result) == 2

    def test_get_children_and_parents(self):
        """获取子节点和父节点."""
        g = EvidenceGraph()
        parent = _make_node("parent")
        child1 = _make_node("child1")
        child2 = _make_node("child2")
        g.add_node(parent)
        g.add_node(child1)
        g.add_node(child2)

        g.add_edge(EvidenceEdge(
            source_node_id=parent.node_id,
            target_node_id=child1.node_id,
            relation=EdgeRelation.DECOMPOSES_TO,
        ))
        g.add_edge(EvidenceEdge(
            source_node_id=parent.node_id,
            target_node_id=child2.node_id,
            relation=EdgeRelation.DECOMPOSES_TO,
        ))

        children = g.get_children(parent.node_id)
        assert len(children) == 2

        parents = g.get_parents(child1.node_id)
        assert len(parents) == 1
        assert parents[0].node_id == parent.node_id

    def test_get_leaf_nodes(self):
        """叶子节点: 没有出边的节点."""
        g = EvidenceGraph()
        root = _make_node("root")
        leaf = _make_node("leaf")
        g.add_node(root)
        g.add_node(leaf)
        g.add_edge(EvidenceEdge(
            source_node_id=root.node_id,
            target_node_id=leaf.node_id,
            relation=EdgeRelation.DERIVED_FROM,
        ))

        leaves = g.get_leaf_nodes()
        assert len(leaves) == 1
        assert leaves[0].node_id == leaf.node_id

    def test_get_high_confidence_nodes(self):
        """高置信度筛选."""
        g = EvidenceGraph()
        g.add_node(_make_node("low", confidence=0.3))
        g.add_node(_make_node("mid", confidence=0.6))
        g.add_node(_make_node("high", confidence=0.9))

        result = g.get_high_confidence_nodes(threshold=0.7)
        assert len(result) == 1

        result = g.get_high_confidence_nodes(threshold=0.5)
        assert len(result) == 2

    def test_summary(self):
        """summary返回正确结构."""
        g = EvidenceGraph()
        g.add_node(_make_node("a", agent="agent_a", confidence=0.8))
        g.add_node(_make_node("b", agent="agent_b", confidence=0.6))

        s = g.summary()
        assert s["node_count"] == 2
        assert s["edge_count"] == 0
        assert set(s["agents_involved"]) == {"agent_a", "agent_b"}
        assert abs(s["avg_confidence"] - 0.7) < 0.01
