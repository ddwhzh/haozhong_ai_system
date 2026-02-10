"""Unit tests for AgentInput / AgentOutput contracts."""

import pytest

from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus
from app.core.types.belief import Belief, BeliefUpdate
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode
from app.core.types.intent import Intent, IntentType


class TestAgentInput:
    """Tests for AgentInput model."""

    def test_create_minimal(self):
        """最小创建: 仅需query."""
        inp = AgentInput(query="test query")
        assert inp.query == "test query"
        assert inp.evidence_graph.node_count == 0
        assert inp.context == {}
        assert inp.max_iterations == 3

    def test_create_with_evidence_graph(self):
        """可以携带已有证据图."""
        g = EvidenceGraph()
        g.add_node(EvidenceNode(
            evidence=Evidence(content="e", source="s"),
            belief=Belief(content="b", confidence=0.8, source_agent="a"),
            intent=Intent(intent_type=IntentType.RETRIEVE, source_agent="a"),
        ))
        inp = AgentInput(query="q", evidence_graph=g)
        assert inp.evidence_graph.node_count == 1


class TestAgentOutput:
    """Tests for AgentOutput model."""

    def test_create_success(self):
        """成功状态输出."""
        out = AgentOutput(
            agent_name="test_agent",
            status=AgentStatus.SUCCESS,
            summary="Completed successfully",
        )
        assert out.agent_name == "test_agent"
        assert out.status == AgentStatus.SUCCESS
        assert out.new_nodes == []
        assert out.belief_updates == []

    def test_create_with_nodes(self):
        """携带新节点."""
        node = EvidenceNode(
            evidence=Evidence(content="content", source="src"),
            belief=Belief(content="b", confidence=0.9, source_agent="agent"),
            intent=Intent(intent_type=IntentType.GENERATE, source_agent="agent"),
        )
        out = AgentOutput(
            agent_name="gen",
            new_nodes=[node],
        )
        assert len(out.new_nodes) == 1

    def test_create_with_belief_updates(self):
        """携带信念更新提案."""
        update = BeliefUpdate(
            target_belief_id="b-1",
            new_confidence=0.3,
            proposed_by="eval",
        )
        out = AgentOutput(
            agent_name="eval",
            belief_updates=[update],
        )
        assert len(out.belief_updates) == 1

    def test_error_output(self):
        """错误状态输出."""
        out = AgentOutput(
            agent_name="test",
            status=AgentStatus.ERROR,
            error_message="Something went wrong",
        )
        assert out.status == AgentStatus.ERROR
        assert out.error_message == "Something went wrong"

    def test_all_status_values(self):
        """所有状态枚举值有效."""
        assert AgentStatus.SUCCESS == "success"
        assert AgentStatus.PARTIAL == "partial"
        assert AgentStatus.NEEDS_MORE_INFO == "needs_more_info"
        assert AgentStatus.ERROR == "error"

    def test_metrics_default_empty(self):
        """metrics默认为空字典."""
        out = AgentOutput(agent_name="t")
        assert out.metrics == {}

    def test_metrics_accepts_mixed_value_types(self):
        """metrics支持字符串分类标签与数值并存."""
        out = AgentOutput(
            agent_name="retrieval_agent",
            metrics={
                "q_type": "QSelect",
                "classification_confidence": 0.92,
                "web_result_count": 5,
            },
        )
        assert out.metrics["q_type"] == "QSelect"
        assert out.metrics["classification_confidence"] == 0.92
        assert out.metrics["web_result_count"] == 5
