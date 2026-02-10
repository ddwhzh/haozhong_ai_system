"""Smoke tests for state machine routing and agent execution wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.base import BaseAgent
from app.agents.evaluation.agent import EvaluationAgent
from app.agents.generation.agent import GenerationAgent
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.query_classifier import (
    ClassificationResult,
    QType,
    StrategyConfig,
)
from app.agents.retrieval.tree_explorer import TreeStep
from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus
from app.core.types.belief import Belief
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode
from app.core.types.intent import Intent, IntentType
from app.core.types.state import PipelinePhase, PipelineState
from app.orchestration.gateway import gateway
from app.orchestration.nodes import assemble_output, phase_transition


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
    """Create an evidence node for smoke tests."""
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


class TestGatewaySmoke:
    """Verify gateway routes correctly for new phase machine."""

    @pytest.mark.asyncio
    async def test_explore_routes_retrieval(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.EXPLORE.value)
        cmd = await gateway(state)
        assert cmd.goto == "retrieval_agent"

    @pytest.mark.asyncio
    async def test_freeze_routes_freeze_context(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.FREEZE.value)
        cmd = await gateway(state)
        assert cmd.goto == "freeze_context"

    @pytest.mark.asyncio
    async def test_plan_routes_plan_phase(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.PLAN.value)
        cmd = await gateway(state)
        assert cmd.goto == "plan_phase"

    @pytest.mark.asyncio
    async def test_execute_routes_generation_agent(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.EXECUTE.value)
        cmd = await gateway(state)
        assert cmd.goto == "generation_agent"

    @pytest.mark.asyncio
    async def test_validate_routes_evaluation_agent(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.VALIDATE.value)
        cmd = await gateway(state)
        assert cmd.goto == "evaluation_agent"

    @pytest.mark.asyncio
    async def test_done_routes_assemble(self):
        state = PipelineState(query="test", current_phase=PipelinePhase.DONE.value)
        cmd = await gateway(state)
        assert cmd.goto == "assemble_output"

    @pytest.mark.asyncio
    async def test_max_iterations_forces_assemble(self):
        state = PipelineState(
            query="test",
            current_phase=PipelinePhase.EXPLORE.value,
            iteration_count=10,
            max_iterations=10,
        )
        cmd = await gateway(state)
        assert cmd.goto == "assemble_output"

    @pytest.mark.asyncio
    async def test_rollback_routes_to_rollback_phase_when_retry_allowed(self):
        state = PipelineState(
            query="test",
            current_phase=PipelinePhase.ROLLBACK.value,
            rollback_count=0,
            max_rollbacks=2,
            iteration_count=1,
            max_iterations=10,
            validation_feedback="need retry",
        )
        cmd = await gateway(state)
        assert cmd.goto == "rollback_phase"

    @pytest.mark.asyncio
    async def test_rollback_routes_to_assemble_when_exhausted(self):
        state = PipelineState(
            query="test",
            current_phase=PipelinePhase.ROLLBACK.value,
            rollback_count=2,
            max_rollbacks=2,
            iteration_count=9,
            max_iterations=10,
        )
        cmd = await gateway(state)
        assert cmd.goto == "assemble_output"


class TestPhaseTransitionSmoke:
    """Verify phase transition helper."""

    @pytest.mark.asyncio
    async def test_explore_to_freeze(self):
        state = PipelineState(current_phase=PipelinePhase.EXPLORE.value)
        result = await phase_transition(state)
        assert result["current_phase"] == PipelinePhase.FREEZE.value

    @pytest.mark.asyncio
    async def test_freeze_to_plan(self):
        state = PipelineState(current_phase=PipelinePhase.FREEZE.value)
        result = await phase_transition(state)
        assert result["current_phase"] == PipelinePhase.PLAN.value

    @pytest.mark.asyncio
    async def test_plan_to_execute(self):
        state = PipelineState(current_phase=PipelinePhase.PLAN.value)
        result = await phase_transition(state)
        assert result["current_phase"] == PipelinePhase.EXECUTE.value

    @pytest.mark.asyncio
    async def test_execute_to_validate(self):
        state = PipelineState(current_phase=PipelinePhase.EXECUTE.value)
        result = await phase_transition(state)
        assert result["current_phase"] == PipelinePhase.VALIDATE.value


class TestAssembleOutputSmoke:
    """Verify output assembly behavior."""

    @pytest.mark.asyncio
    async def test_assemble_with_generated_content(self):
        graph = EvidenceGraph()
        graph.add_node(_node(
            content="这是生成内容",
            content_type="generated_content",
            source="content_generator",
            source_agent="generation_agent",
            metadata={"priority": 1, "filled_count": 1},
            tags=["generated"],
        ))
        state = PipelineState(evidence_graph=graph)

        result = await assemble_output(state)
        assert result["final_output"] != ""
        assert result["current_phase"] == PipelinePhase.DONE.value
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_assemble_empty_graph(self):
        state = PipelineState(evidence_graph=EvidenceGraph())
        result = await assemble_output(state)
        assert "No output generated" in result["final_output"]


class TestRetrievalAgentSmoke:
    """Verify retrieval agent wrapper runs with mocked sub-components."""

    @pytest.mark.asyncio
    @patch("app.agents.retrieval.vector_retriever.OpenAIEmbeddings")
    @patch("app.agents.retrieval.web_retriever.OpenAIEmbeddings")
    async def test_retrieval_agent_runs_with_component_mocks(
        self, mock_web_embeddings_cls, mock_vector_embeddings_cls
    ):
        mock_embedder = MagicMock()
        mock_vector_embeddings_cls.return_value = mock_embedder
        mock_web_embeddings_cls.return_value = mock_embedder

        agent = RetrievalAgent()
        graph = EvidenceGraph()

        classification = ClassificationResult(
            q_type=QType.Q_SELECT,
            confidence=0.9,
            reasoning="simple select query",
            decomposition_hint=["single-hop"],
            strategy=StrategyConfig.for_type(QType.Q_SELECT),
        )
        class_node = _node(
            content='{"q_type":"QSelect"}',
            content_type="query_classification",
            source="query_classifier",
            source_agent="retrieval_agent",
        )
        tree_step = TreeStep(
            step_idx=1,
            question="什么是知识图谱",
            operator="select",
            retrieval_strategy="both",
            depends_on=[],
        )
        step_node = _node(
            content="什么是知识图谱",
            content_type="tree_step",
            source="tree_explorer",
            source_agent="retrieval_agent",
        )
        doc_node = _node(
            content="知识图谱是语义网络",
            content_type="document_chunk",
            source="doc-1",
            source_agent="retrieval_agent",
            confidence=0.82,
        )
        web_node = _node(
            content="百科条目\n示例摘要",
            content_type="web_result",
            source="https://example.com",
            source_agent="retrieval_agent",
            confidence=0.73,
        )
        fused_node = _node(
            content="融合后证据摘要",
            content_type="fused_evidence",
            source="cascade_fuser",
            source_agent="retrieval_agent",
            confidence=0.79,
        )

        agent.query_classifier.classify = AsyncMock(return_value=(classification, class_node))
        agent.query_decomposer.qdmr_decompose = AsyncMock(return_value=[tree_step])
        agent.tree_explorer.explore = AsyncMock(return_value=[step_node, doc_node, web_node])
        agent.cascade_fuser.fuse = AsyncMock(return_value=[fused_node])

        output = await agent._execute(AgentInput(query="什么是知识图谱?", evidence_graph=graph))

        assert output.agent_name == "retrieval_agent"
        assert output.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)
        assert output.metrics["q_type"] == "QSelect"
        assert output.metrics["web_result_count"] == 1
        assert len(output.new_nodes) == 5


class TestGenerationAgentSmoke:
    """Verify generation agent runs with mocked two-stage components."""

    @pytest.mark.asyncio
    async def test_generation_agent_runs_with_component_mocks(self):
        agent = GenerationAgent()
        graph = EvidenceGraph()

        proposal_node = _node(
            content='{"aspect":"综合回答"}',
            content_type="proposal",
            source="proposal_decomposer",
            source_agent="generation_agent",
            metadata={"aspect": "综合回答", "priority": 1, "slot_count": 2},
        )
        generated_node = _node(
            content="## 综合回答\n\n这是基于证据生成的内容",
            content_type="generated_content",
            source="content_generator",
            source_agent="generation_agent",
            metadata={"aspect": "综合回答", "priority": 1, "filled_count": 1},
        )

        agent.proposal_decomposer.generate_proposals = AsyncMock(return_value=[proposal_node])
        agent.content_generator.generate = AsyncMock(return_value=[generated_node])

        output = await agent._execute(AgentInput(query="测试查询", evidence_graph=graph))

        assert output.agent_name == "generation_agent"
        assert output.status == AgentStatus.SUCCESS
        assert output.metrics["proposal_count"] == 1
        assert output.metrics["generated_count"] == 1
        assert len(output.new_nodes) == 2


class TestEvaluationAgentSmoke:
    """Verify evaluation agent runs with mocked matching outputs."""

    @pytest.mark.asyncio
    @patch("app.agents.evaluation.agent.OpenAIEmbeddings")
    async def test_evaluation_agent_runs_with_component_mocks(self, mock_embeddings_cls):
        mock_embeddings_cls.return_value = MagicMock()
        agent = EvaluationAgent()

        graph = EvidenceGraph()
        generated_node = _node(
            content="这是生成答案",
            content_type="generated_content",
            source="content_generator",
            source_agent="generation_agent",
            metadata={"priority": 1, "filled_count": 1},
            confidence=0.8,
        )
        evidence_node = _node(
            content="这是检索证据",
            content_type="fused_evidence",
            source="cascade_fuser",
            source_agent="retrieval_agent",
            confidence=0.85,
        )
        graph.add_node(generated_node)
        graph.add_node(evidence_node)

        agent._prepare_for_matching = AsyncMock(side_effect=[
            [{"id": generated_node.node_id, "content": generated_node.evidence.content, "tokens": [], "embedding": [0.1]}],
            [{"id": evidence_node.node_id, "content": evidence_node.evidence.content, "tokens": [], "embedding": [0.2]}],
        ])
        agent.matcher.evaluate_faithfulness = MagicMock(return_value={
            "overall_score": 0.86,
            "faithfulness_rate": 1.0,
            "section_scores": [
                {"generated_idx": 0, "score": 0.86, "is_faithful": True}
            ],
        })
        agent.auto_prompt.optimize = AsyncMock(return_value=[])

        output = await agent._execute(AgentInput(query="测试查询", evidence_graph=graph))

        assert output.agent_name == "evaluation_agent"
        assert output.status == AgentStatus.SUCCESS
        assert output.metrics["overall_score"] == 0.86
        assert output.metrics["sections_evaluated"] == 1
        assert len(output.belief_updates) == 1


class TestBaseAgentRunSmoke:
    """Verify BaseAgent.run wraps _execute and handles failures."""

    @pytest.mark.asyncio
    async def test_run_wraps_execution(self):
        class DummyAgent(BaseAgent):
            agent_name = "dummy_agent"

            async def _execute(self, agent_input: AgentInput) -> AgentOutput:
                return AgentOutput(
                    agent_name=self.agent_name,
                    status=AgentStatus.SUCCESS,
                    summary="dummy done",
                )

        agent = DummyAgent()
        state = PipelineState(query="test")
        result = await agent.run(state)

        assert "evidence_graph" in result
        assert "agent_outputs" in result
        assert len(result["agent_outputs"]) == 1
        assert result["agent_outputs"][0].agent_name == "dummy_agent"
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_run_handles_exception(self):
        class FailingAgent(BaseAgent):
            agent_name = "failing_agent"

            async def _execute(self, agent_input: AgentInput) -> AgentOutput:
                raise RuntimeError("intentional failure")

        agent = FailingAgent()
        state = PipelineState(query="test")
        result = await agent.run(state)

        output = result["agent_outputs"][0]
        assert output.status == AgentStatus.ERROR
        assert "intentional failure" in output.error_message
