"""Main LangGraph graph definition — the orchestration backbone.

Wires together the disciplined agent workflow state machine:

    gateway -> retrieval_agent  -> phase_transition     -> gateway  (Explore)
    gateway -> freeze_context   -> phase_transition     -> gateway  (Freeze)
    gateway -> plan_phase       -> phase_transition     -> gateway  (Plan)
    gateway -> generation_agent -> phase_transition     -> gateway  (Execute)
    gateway -> evaluation_agent -> post_validate        -> gateway  (Validate)
    gateway -> rollback_phase                           -> gateway  (Rollback)
    gateway -> human_review     -> phase_transition     -> gateway  (HITL)
    gateway -> assemble_output  -> END                              (Done)
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote_plus

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.evaluation import EvaluationAgent
from app.agents.generation import GenerationAgent
from app.agents.retrieval import RetrievalAgent
from app.core.config import Environment, settings
from app.core.logging import logger
from app.core.types.state import PipelineState
from app.infra.database.connection import connection_manager
from app.orchestration.gateway import gateway
from app.orchestration.nodes import (
    assemble_output,
    freeze_context,
    human_review,
    phase_transition,
    plan_phase,
    post_validate_transition,
    rollback_phase,
)


class PipelineGraph:
    """Manages the LangGraph pipeline lifecycle."""

    def __init__(self) -> None:
        self._graph: Optional[CompiledStateGraph] = None

        # Instantiate worker agents
        self._retrieval_agent = RetrievalAgent()
        self._generation_agent = GenerationAgent()
        self._evaluation_agent = EvaluationAgent()

    async def create(self) -> CompiledStateGraph:
        """Build and compile the LangGraph graph."""
        if self._graph is not None:
            return self._graph

        builder = StateGraph(PipelineState)

        # ── Add nodes ────────────────────────────────────────────────────
        # Gateway: the ONLY node that returns Command.goto
        builder.add_node("gateway", gateway)

        # Worker agents: return plain dicts, NEVER Command
        builder.add_node("retrieval_agent", self._retrieval_agent.run)
        builder.add_node("generation_agent", self._generation_agent.run)
        builder.add_node("evaluation_agent", self._evaluation_agent.run)

        # Helper nodes: return plain dicts
        builder.add_node("phase_transition", phase_transition)
        builder.add_node("post_validate", post_validate_transition)
        builder.add_node("freeze_context", freeze_context)
        builder.add_node("plan_phase", plan_phase)
        builder.add_node("rollback_phase", rollback_phase)
        builder.add_node("assemble_output", assemble_output)
        builder.add_node("human_review", human_review)

        # ── Add edges ────────────────────────────────────────────────────
        # Entry point
        builder.set_entry_point("gateway")

        # Explore: retrieval_agent -> phase_transition -> gateway
        builder.add_edge("retrieval_agent", "phase_transition")

        # Freeze: freeze_context -> phase_transition -> gateway
        builder.add_edge("freeze_context", "phase_transition")

        # Plan: plan_phase -> phase_transition -> gateway
        builder.add_edge("plan_phase", "phase_transition")

        # Execute: generation_agent -> phase_transition -> gateway
        builder.add_edge("generation_agent", "phase_transition")

        # Validate: evaluation_agent -> post_validate -> gateway
        # (post_validate decides done/rollback instead of generic phase_transition)
        builder.add_edge("evaluation_agent", "post_validate")
        builder.add_edge("post_validate", "gateway")

        # Rollback: rollback_phase -> gateway (re-enters explore)
        builder.add_edge("rollback_phase", "gateway")

        # Generic phase_transition -> gateway (loop back)
        builder.add_edge("phase_transition", "gateway")

        # Human review -> phase_transition -> gateway
        builder.add_edge("human_review", "phase_transition")

        # Output assembly -> END
        builder.add_edge("assemble_output", END)

        # Gateway routing is handled by Command.goto (dynamic edges)

        # ── Compile with checkpointer ────────────────────────────────────
        checkpointer = await self._get_checkpointer()

        compile_kwargs = {
            "name": f"{settings.PROJECT_NAME} Pipeline ({settings.ENVIRONMENT.value})",
        }
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer

        # Add HITL interrupt if enabled
        if settings.HITL_ENABLED:
            compile_kwargs["interrupt_before"] = ["human_review"]

        self._graph = builder.compile(**compile_kwargs)

        logger.info(
            "pipeline_graph_created",
            nodes=[
                "gateway", "retrieval_agent", "generation_agent",
                "evaluation_agent", "phase_transition", "post_validate",
                "freeze_context", "plan_phase", "rollback_phase",
                "assemble_output", "human_review",
            ],
            hitl_enabled=settings.HITL_ENABLED,
            has_checkpointer=checkpointer is not None,
        )

        return self._graph

    async def _get_checkpointer(self) -> Optional[AsyncPostgresSaver]:
        """Create a PostgreSQL checkpointer for state persistence."""
        try:
            pool = await connection_manager.get_pool()
            if pool is None:
                return None
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            return checkpointer
        except Exception as e:
            logger.error("checkpointer_setup_failed", error=str(e))
            if settings.ENVIRONMENT != Environment.PRODUCTION:
                raise
            return None

    @property
    def graph(self) -> Optional[CompiledStateGraph]:
        return self._graph


# Singleton
pipeline_graph = PipelineGraph()
