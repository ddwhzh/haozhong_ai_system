"""Base agent interface — contract that all worker agents must implement.

Design rule (from architecture constraints):
- Worker nodes ONLY return State or dict, NEVER Command.
- Worker nodes NEVER decide "next node" — that's the Gateway's job.
- All inter-agent communication goes through the EvidenceGraph.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict

from app.core.logging import logger
from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus
from app.core.types.state import PipelineState


class BaseAgent(ABC):
    """Abstract base for all worker agents.

    Subclasses implement `_execute` which receives an AgentInput and returns
    an AgentOutput.  The `run` method wraps execution with logging, timing,
    and error handling.
    """

    agent_name: str = "base_agent"

    @abstractmethod
    async def _execute(self, agent_input: AgentInput) -> AgentOutput:
        """Core agent logic — to be implemented by subclasses.

        Args:
            agent_input: Typed input envelope with query, evidence graph, context.

        Returns:
            AgentOutput with new evidence nodes and/or belief updates.
        """
        ...

    async def run(self, state: PipelineState) -> Dict[str, Any]:
        """Entry point called by LangGraph node.

        Converts PipelineState -> AgentInput, calls _execute, and returns
        a state update dict.  This method NEVER returns a Command.

        Returns:
            dict suitable for LangGraph state update (merged by reducer).
        """
        start = time.monotonic()
        logger.info(
            "agent_started",
            agent=self.agent_name,
            phase=state.current_phase,
            iteration=state.iteration_count,
        )

        agent_input = AgentInput(
            query=state.query,
            evidence_graph=state.evidence_graph,
            context=state.metadata,
            conversation_history=[],
            max_iterations=state.max_iterations,
        )

        try:
            output = await self._execute(agent_input)
        except Exception as e:
            logger.error(
                "agent_execution_failed",
                agent=self.agent_name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            output = AgentOutput(
                agent_name=self.agent_name,
                status=AgentStatus.ERROR,
                error_message=str(e),
            )

        elapsed = time.monotonic() - start
        output.metrics["elapsed_seconds"] = round(elapsed, 3)

        logger.info(
            "agent_finished",
            agent=self.agent_name,
            status=output.status.value,
            new_nodes=len(output.new_nodes),
            elapsed=elapsed,
        )

        # Merge new evidence nodes into the graph
        updated_graph = state.evidence_graph.model_copy(deep=True)
        for node in output.new_nodes:
            updated_graph.add_node(node)

        return {
            "evidence_graph": updated_graph,
            "agent_outputs": [output],
            "iteration_count": state.iteration_count + 1,
        }
