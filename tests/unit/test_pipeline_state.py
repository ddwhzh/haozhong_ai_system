"""Unit tests for PipelineState."""

import pytest

from app.core.types.agent_io import AgentOutput, AgentStatus
from app.core.types.evidence import EvidenceGraph
from app.core.types.state import PipelineState


class TestPipelineState:
    """Tests for PipelineState model."""

    def test_default_state(self):
        """默认状态正确初始化."""
        state = PipelineState()
        assert state.query == ""
        assert state.current_phase == "explore"
        assert state.next_agent is None
        assert state.iteration_count == 0
        assert state.max_iterations == 10
        assert state.final_output == ""
        assert isinstance(state.evidence_graph, EvidenceGraph)
        assert state.evidence_graph.node_count == 0

    def test_state_with_query(self):
        """携带查询创建."""
        state = PipelineState(query="test query")
        assert state.query == "test query"

    def test_agent_outputs_merge(self):
        """agent_outputs支持列表合并."""
        out1 = AgentOutput(agent_name="a1", status=AgentStatus.SUCCESS)
        out2 = AgentOutput(agent_name="a2", status=AgentStatus.SUCCESS)

        state = PipelineState(agent_outputs=[out1])
        # 模拟reducer行为
        merged = state.agent_outputs + [out2]
        assert len(merged) == 2
        assert merged[0].agent_name == "a1"
        assert merged[1].agent_name == "a2"

    def test_metadata_default_empty(self):
        """metadata默认为空字典."""
        state = PipelineState()
        assert state.metadata == {}

    def test_state_with_metadata(self):
        """携带元数据创建."""
        state = PipelineState(
            metadata={"user_id": "u1", "session_id": "s1"}
        )
        assert state.metadata["user_id"] == "u1"
