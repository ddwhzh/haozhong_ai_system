"""Gateway — the ONLY node allowed to use Command.goto for routing.

Implements the disciplined agent workflow state machine:

    [*] --> Explore       (retrieval: RAG + KG)
    Explore --> Freeze    (snapshot context_manifest + evidence)
    Freeze --> Plan       (proposal decomposition)
    Plan --> Execute      (content generation)
    Execute --> Validate  (evaluation + scoring)
    Validate --> Done     (pass -> assemble output)
    Validate --> Rollback (fail -> expand/repair)
    Rollback --> Explore  (retry with feedback)

Architecture constraint (from LangGraph rules):
- Worker nodes ONLY return State/dict, NEVER Command.
- ONLY this Gateway node returns Command[Literal[...]] with goto.
- All routing decisions are centralised here for auditability.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph.state import Command

from app.core.config import settings
from app.core.logging import logger
from app.core.types.state import PipelinePhase, PipelineState

# Type-safe routing targets
ROUTE_TARGETS = Literal[
    "retrieval_agent",
    "freeze_context",
    "plan_phase",
    "generation_agent",
    "evaluation_agent",
    "human_review",
    "rollback_phase",
    "assemble_output",
    "__end__",
]


async def gateway(state: PipelineState) -> Command[ROUTE_TARGETS]:
    """Central routing node — decides next step based on pipeline state machine.

    Phase transitions follow the disciplined workflow:
      explore -> freeze -> plan -> execute -> validate -> done
      validate -> rollback -> explore (if quality fails and retries remain)
    """
    phase = state.current_phase
    iteration = state.iteration_count
    max_iter = state.max_iterations

    logger.info(
        "gateway_routing",
        phase=phase,
        iteration=iteration,
        max_iterations=max_iter,
        rollback_count=state.rollback_count,
    )

    # Safety: hard iteration limit
    if iteration >= max_iter:
        logger.warning("gateway_max_iterations_reached", iteration=iteration)
        return Command(
            update={"current_phase": PipelinePhase.DONE.value},
            goto="assemble_output",
        )

    # ── Phase-based routing (state machine) ───────────────────────────────

    if phase == PipelinePhase.EXPLORE.value:
        # Explore: run retrieval agent (RAG + KG + cascade fusion)
        return Command(
            update={"next_agent": "retrieval_agent"},
            goto="retrieval_agent",
        )

    if phase == PipelinePhase.FREEZE.value:
        # Freeze: snapshot context manifest + evidence
        return Command(goto="freeze_context")

    if phase == PipelinePhase.PLAN.value:
        # Plan: proposal decomposition (generation planning)
        return Command(goto="plan_phase")

    if phase == PipelinePhase.EXECUTE.value:
        # Execute: content generation per proposal
        return Command(
            update={"next_agent": "generation_agent"},
            goto="generation_agent",
        )

    if phase == PipelinePhase.VALIDATE.value:
        # Validate: evaluation + scoring + Hungarian matching
        return Command(
            update={"next_agent": "evaluation_agent"},
            goto="evaluation_agent",
        )

    if phase == PipelinePhase.ROLLBACK.value:
        # Rollback: repair and retry
        return _rollback_routing(state)

    if phase == PipelinePhase.HUMAN_REVIEW.value:
        # After human review, proceed to assembly
        return Command(
            update={"current_phase": PipelinePhase.DONE.value},
            goto="assemble_output",
        )

    if phase == PipelinePhase.DONE.value:
        return Command(goto="assemble_output")

    # Default: start with explore
    return Command(
        update={"current_phase": PipelinePhase.EXPLORE.value},
        goto="retrieval_agent",
    )


def _rollback_routing(state: PipelineState) -> Command[ROUTE_TARGETS]:
    """Handle rollback: decide whether to retry or give up.

    Rules:
    1. If rollback_count < max_rollbacks and iterations remain -> Explore again
    2. Otherwise -> assemble best-effort output
    """
    if (
        state.rollback_count < state.max_rollbacks
        and state.iteration_count < state.max_iterations - 3
    ):
        logger.info(
            "gateway_rollback_to_explore",
            rollback_count=state.rollback_count,
            feedback=state.validation_feedback[:100],
        )
        return Command(goto="rollback_phase")

    # Exhausted rollbacks — deliver best effort
    logger.warning(
        "gateway_exhausted_rollbacks",
        rollback_count=state.rollback_count,
        iterations=state.iteration_count,
    )
    return Command(
        update={"current_phase": PipelinePhase.DONE.value},
        goto="assemble_output",
    )
