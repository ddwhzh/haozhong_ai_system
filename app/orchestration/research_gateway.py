"""Research Gateway — the ONLY routing node in the research graph.

Implements the research loop state machine:

    [*] --> Scout        (domain exploration)
    Scout --> Survey     (literature retrieval)
    Survey --> Synthesize(knowledge synthesis)
    Synthesize --> Design(experiment design)
    Design --> Experiment(code execution)
    Experiment --> Analyze(result analysis)
    Analyze --> Decide   (iteration decision)
    Decide --> Survey    (iterate: refine approach)
    Decide --> Scout     (pivot: change direction)
    Decide --> Conclude  (conclude: generate report)
    Conclude --> END

Architecture constraint: ONLY this node returns Command[Literal[...]] with goto.
All worker nodes return plain dicts.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph.state import Command

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_state import ResearchPhase, ResearchState
from app.core.types.research_types import ResearchDecision

RESEARCH_ROUTE_TARGETS = Literal[
    "scout",
    "survey",
    "synthesize",
    "design",
    "experiment",
    "analyze",
    "decide",
    "conclude",
    "__end__",
]


async def research_gateway(state: ResearchState) -> Command[RESEARCH_ROUTE_TARGETS]:
    """Central routing node for the research pipeline.

    Routes based on current_phase, enforcing the research state machine.
    """
    phase = state.current_phase
    iteration = state.iteration_count
    max_iter = state.max_iterations

    logger.info(
        "research_gateway_routing",
        phase=phase,
        iteration=iteration,
        max_iterations=max_iter,
    )

    if iteration > max_iter:
        logger.warning("research_gateway_max_iterations_exceeded", iteration=iteration)
        return Command(
            update={"current_phase": ResearchPhase.CONCLUDE.value},
            goto="conclude",
        )

    if phase == ResearchPhase.SCOUT.value:
        return Command(goto="scout")

    if phase == ResearchPhase.SURVEY.value:
        return Command(goto="survey")

    if phase == ResearchPhase.SYNTHESIZE.value:
        return Command(goto="synthesize")

    if phase == ResearchPhase.DESIGN.value:
        return Command(goto="design")

    if phase == ResearchPhase.EXPERIMENT.value:
        return Command(goto="experiment")

    if phase == ResearchPhase.ANALYZE.value:
        return Command(goto="analyze")

    if phase == ResearchPhase.DECIDE.value:
        return Command(goto="decide")

    if phase == ResearchPhase.CONCLUDE.value:
        return Command(goto="conclude")

    return Command(
        update={"current_phase": ResearchPhase.SCOUT.value},
        goto="scout",
    )
