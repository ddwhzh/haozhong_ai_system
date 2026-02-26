"""Research LangGraph — the orchestration backbone for automated research.

Wires together the research loop state machine:

    gateway -> scout               -> phase_transition -> gateway  (Scout)
    gateway -> survey              -> phase_transition -> gateway  (Survey)
    gateway -> synthesize          -> phase_transition -> gateway  (Synthesize)
    gateway -> design              -> phase_transition -> gateway  (Design)
    gateway -> experiment          -> phase_transition -> gateway  (Experiment)
    gateway -> analyze             -> phase_transition -> gateway  (Analyze)
    gateway -> decide              -> post_decide      -> gateway  (Decide)
    gateway -> conclude            -> END                          (Conclude)

This graph is completely independent of the main PipelineGraph.
"""

from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.research import ResearchAgent
from app.core.config import Environment, settings
from app.core.logging import logger
from app.core.types.research_state import ResearchState
from app.infra.database.connection import connection_manager
from app.orchestration.research_gateway import research_gateway
from app.orchestration.research_nodes import (
    research_conclude,
    research_phase_transition,
    research_post_decide,
)


class ResearchGraph:
    """Manages the research LangGraph lifecycle."""

    def __init__(self) -> None:
        self._graph: Optional[CompiledStateGraph] = None
        self._research_agent = ResearchAgent()

    async def create(self) -> CompiledStateGraph:
        """Build and compile the research LangGraph."""
        if self._graph is not None:
            return self._graph

        builder = StateGraph(ResearchState)

        # -- Gateway: the ONLY node that returns Command.goto --
        builder.add_node("gateway", research_gateway)

        # -- Worker nodes: return plain dicts, NEVER Command --
        builder.add_node("scout", self._research_agent.run_scout)
        builder.add_node("survey", self._research_agent.run_survey)
        builder.add_node("synthesize", self._research_agent.run_synthesize)
        builder.add_node("design", self._research_agent.run_design)
        builder.add_node("experiment", self._research_agent.run_experiment)
        builder.add_node("analyze", self._research_agent.run_analyze)
        builder.add_node("decide", self._research_agent.run_decide)

        # -- Helper nodes --
        builder.add_node("phase_transition", research_phase_transition)
        builder.add_node("post_decide", research_post_decide)
        builder.add_node("conclude", research_conclude)

        # -- Entry point --
        builder.set_entry_point("gateway")

        # -- Edges --
        # Scout -> phase_transition -> gateway
        builder.add_edge("scout", "phase_transition")

        # Survey -> phase_transition -> gateway
        builder.add_edge("survey", "phase_transition")

        # Synthesize -> phase_transition -> gateway
        builder.add_edge("synthesize", "phase_transition")

        # Design -> phase_transition -> gateway
        builder.add_edge("design", "phase_transition")

        # Experiment -> phase_transition -> gateway
        builder.add_edge("experiment", "phase_transition")

        # Analyze -> phase_transition -> gateway
        builder.add_edge("analyze", "phase_transition")

        # Decide -> post_decide -> gateway (handles iterate/pivot/conclude routing)
        builder.add_edge("decide", "post_decide")
        builder.add_edge("post_decide", "gateway")

        # Generic phase_transition -> gateway
        builder.add_edge("phase_transition", "gateway")

        # Conclude -> END
        builder.add_edge("conclude", END)

        # -- Compile --
        checkpointer = await self._get_checkpointer()

        compile_kwargs = {
            "name": f"{settings.PROJECT_NAME} Research Pipeline ({settings.ENVIRONMENT.value})",
        }
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer

        self._graph = builder.compile(**compile_kwargs)

        logger.info(
            "research_graph_created",
            nodes=[
                "gateway", "scout", "survey", "synthesize", "design",
                "experiment", "analyze", "decide", "phase_transition",
                "post_decide", "conclude",
            ],
            has_checkpointer=checkpointer is not None,
        )

        return self._graph

    async def _get_checkpointer(self) -> Optional[AsyncPostgresSaver]:
        """Create a PostgreSQL checkpointer for research state persistence."""
        try:
            pool = await connection_manager.get_pool()
            if pool is None:
                return None
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            return checkpointer
        except Exception as e:
            logger.error("research_checkpointer_setup_failed", error=str(e))
            if settings.ENVIRONMENT != Environment.PRODUCTION:
                raise
            return None

    @property
    def graph(self) -> Optional[CompiledStateGraph]:
        return self._graph


# Singleton
research_graph = ResearchGraph()
