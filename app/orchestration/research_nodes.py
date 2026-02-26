"""Research helper nodes — phase transitions and report assembly.

These are the non-agent nodes in the research graph:
- research_phase_transition: advance to the next phase after each worker
- research_conclude: assemble the final research report
"""

from __future__ import annotations

import json
from typing import Any, Dict

from app.core.logging import logger
from app.core.types.research_state import ResearchPhase, ResearchState
from app.core.types.research_types import (
    JournalEntry,
    ResearchDecision,
)


PHASE_SEQUENCE = {
    ResearchPhase.SCOUT.value: ResearchPhase.SURVEY.value,
    ResearchPhase.SURVEY.value: ResearchPhase.SYNTHESIZE.value,
    ResearchPhase.SYNTHESIZE.value: ResearchPhase.DESIGN.value,
    ResearchPhase.DESIGN.value: ResearchPhase.EXPERIMENT.value,
    ResearchPhase.EXPERIMENT.value: ResearchPhase.ANALYZE.value,
    ResearchPhase.ANALYZE.value: ResearchPhase.DECIDE.value,
}


async def research_phase_transition(state: ResearchState) -> Dict[str, Any]:
    """Advance to the next research phase in the standard sequence."""
    current = state.current_phase
    next_phase = PHASE_SEQUENCE.get(current, ResearchPhase.DECIDE.value)

    logger.info(
        "research_phase_transition",
        from_phase=current,
        to_phase=next_phase,
        iteration=state.iteration_count,
    )

    return {"current_phase": next_phase}


async def research_post_decide(state: ResearchState) -> Dict[str, Any]:
    """Handle the outcome of the DECIDE phase.

    Routes based on the latest iteration record's decision:
    - iterate -> SURVEY (loop back for more research)
    - pivot   -> SCOUT (restart with new direction)
    - conclude -> CONCLUDE (generate final report)
    """
    if not state.iteration_history:
        logger.warning("research_post_decide_no_history")
        return {"current_phase": ResearchPhase.CONCLUDE.value}

    latest = state.iteration_history[-1]
    decision = latest.decision

    if decision == ResearchDecision.ITERATE:
        logger.info(
            "research_post_decide_iterate",
            direction=latest.direction[:80],
            iteration=state.iteration_count,
        )
        return {"current_phase": ResearchPhase.SURVEY.value}

    if decision == ResearchDecision.PIVOT:
        logger.info(
            "research_post_decide_pivot",
            direction=latest.direction[:80],
            iteration=state.iteration_count,
        )
        return {"current_phase": ResearchPhase.SCOUT.value}

    logger.info("research_post_decide_conclude", iteration=state.iteration_count)
    return {"current_phase": ResearchPhase.CONCLUDE.value}


async def research_conclude(state: ResearchState) -> Dict[str, Any]:
    """Assemble the final research report from accumulated state."""
    logger.info(
        "research_conclude_start",
        topic=state.research_topic[:80],
        iterations=state.iteration_count,
    )

    if state.final_report:
        report = state.final_report
    else:
        report = _build_report(state)

    journal = JournalEntry(
        phase="conclude",
        summary=f"Research concluded after {state.iteration_count} iterations",
        details={
            "report_length": len(report),
            "total_papers": len(state.literature),
            "total_hypotheses": len(state.hypotheses),
            "total_experiments": len(state.experiments),
        },
    )

    logger.info(
        "research_conclude_complete",
        report_length=len(report),
        total_papers=len(state.literature),
    )

    return {
        "final_report": report,
        "research_journal": [journal],
    }


def _build_report(state: ResearchState) -> str:
    """Build a structured research report from the accumulated state."""
    sections = []

    sections.append(f"# Research Report: {state.research_topic}\n")
    sections.append(f"**Iterations**: {state.iteration_count}")
    sections.append(f"**Papers Reviewed**: {len(state.literature)}")
    sections.append(f"**Hypotheses Tested**: {len(state.hypotheses)}")
    sections.append(f"**Experiments Run**: {len(state.experiments)}\n")

    if state.synthesis_report:
        sections.append("## Synthesis\n")
        sections.append(state.synthesis_report)
        sections.append("")

    if state.hypotheses:
        sections.append("## Hypotheses\n")
        for h in state.hypotheses:
            sections.append(
                f"- **[{h.status.value}]** {h.statement} "
                f"(confidence: {h.confidence:.2f})"
            )
        sections.append("")

    if state.experiments:
        sections.append("## Experiments\n")
        for e in state.experiments:
            result_str = ""
            if e.result:
                result_str = f" | exit={e.result.exit_code}, elapsed={e.result.elapsed_seconds}s"
            sections.append(
                f"- **{e.name}** [{e.status.value}]{result_str}"
            )
        sections.append("")

    if state.analysis_report:
        sections.append("## Latest Analysis\n")
        sections.append(state.analysis_report)
        sections.append("")

    if state.iteration_history:
        sections.append("## Iteration History\n")
        for r in state.iteration_history:
            sections.append(
                f"- Iteration {r.iteration}: {r.decision.value}"
                + (f" -> {r.direction}" if r.direction else "")
            )
            for f in r.key_findings[:3]:
                sections.append(f"  - {f}")
        sections.append("")

    if state.literature:
        sections.append("## Key References\n")
        top_papers = sorted(
            state.literature, key=lambda p: p.relevance_score, reverse=True
        )[:10]
        for p in top_papers:
            sections.append(
                f"- {p.title} ({p.year or 'N/A'}) [{p.source.value}]"
            )

    return "\n".join(sections)
