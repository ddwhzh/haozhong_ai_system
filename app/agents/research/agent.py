"""Research Agent — the main agent wiring all research sub-components.

Unlike the retrieval/generation/evaluation agents that inherit from BaseAgent
(which assumes PipelineState), the ResearchAgent works with ResearchState
and exposes individual phase methods called by the research LangGraph nodes.

Each method reads from ResearchState, delegates to the appropriate
sub-component, and returns a state update dict for LangGraph to merge.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

from app.agents.research.domain_scout import DomainScout
from app.agents.research.experiment_designer import ExperimentDesigner
from app.agents.research.experiment_runner import ExperimentRunner
from app.agents.research.iteration_planner import IterationPlanner
from app.agents.research.knowledge_synthesizer import KnowledgeSynthesizer
from app.agents.research.literature_surveyor import LiteratureSurveyor
from app.agents.research.result_analyzer import ResultAnalyzer
from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_state import ResearchPhase, ResearchState
from app.core.types.research_types import (
    HypothesisStatus,
    ResearchDecision,
)


class ResearchAgent:
    """Orchestrates the research loop sub-components.

    Each public method corresponds to one phase in the research LangGraph.
    Methods receive ResearchState and return a dict for state update.
    """

    def __init__(self) -> None:
        self.domain_scout = DomainScout()
        self.literature_surveyor = LiteratureSurveyor()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.experiment_designer = ExperimentDesigner()
        self.experiment_runner = ExperimentRunner()
        self.result_analyzer = ResultAnalyzer()
        self.iteration_planner = IterationPlanner()

    async def run_scout(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: SCOUT — explore the research domain."""
        start = time.monotonic()
        logger.info("research_agent_scout_start", topic=state.research_topic[:80])

        try:
            result = await self.domain_scout.scout(state.research_topic)
        except Exception as exc:
            logger.exception("research_agent_scout_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("scout", f"Scout failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        logger.info("research_agent_scout_complete", elapsed=round(elapsed, 2))

        return {
            "domain_context": result["domain_context"],
            "research_journal": [result["journal_entry"]],
        }

    async def run_survey(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: SURVEY — retrieve academic literature."""
        start = time.monotonic()
        logger.info("research_agent_survey_start", iteration=state.iteration_count)

        try:
            result = await self.literature_surveyor.survey(
                research_topic=state.research_topic,
                domain_context=state.domain_context,
                iteration=state.iteration_count,
            )
        except Exception as exc:
            logger.exception("research_agent_survey_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("survey", f"Survey failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        logger.info(
            "research_agent_survey_complete",
            papers=len(result["papers"]),
            elapsed=round(elapsed, 2),
        )

        return {
            "literature": result["papers"],
            "research_journal": [result["journal_entry"]],
            "metadata": {
                **state.metadata,
                "survey_assessment": result.get("survey_assessment", {}),
            },
        }

    async def run_synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: SYNTHESIZE — synthesize literature into hypotheses."""
        start = time.monotonic()
        logger.info("research_agent_synthesize_start", iteration=state.iteration_count)

        survey_assessment = state.metadata.get("survey_assessment", {})

        try:
            result = await self.knowledge_synthesizer.synthesize(
                research_topic=state.research_topic,
                domain_context=state.domain_context,
                papers=state.literature,
                survey_assessment=survey_assessment,
                iteration=state.iteration_count,
                previous_findings=state.synthesis_report,
            )
        except Exception as exc:
            logger.exception("research_agent_synthesize_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("synthesize", f"Synthesis failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        logger.info(
            "research_agent_synthesize_complete",
            hypotheses=len(result["hypotheses"]),
            elapsed=round(elapsed, 2),
        )

        return {
            "hypotheses": result["hypotheses"],
            "synthesis_report": result["synthesis_report"],
            "research_journal": [result["journal_entry"]],
        }

    async def run_design(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: DESIGN — design experiments for hypotheses."""
        start = time.monotonic()
        logger.info("research_agent_design_start", iteration=state.iteration_count)

        testable = [
            h for h in state.hypotheses
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.INCONCLUSIVE)
        ]
        if not testable:
            logger.warning("research_agent_design_no_testable_hypotheses")
            return {
                "research_journal": [
                    _error_journal("design", "No testable hypotheses available")
                ],
            }

        hypothesis = testable[0]
        previous_results = state.analysis_report

        try:
            result = await self.experiment_designer.design(
                hypothesis=hypothesis,
                research_topic=state.research_topic,
                previous_results=previous_results,
                iteration=state.iteration_count,
            )
        except Exception as exc:
            logger.exception("research_agent_design_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("design", f"Design failed: {exc}")
                ],
            }

        hypothesis.status = HypothesisStatus.TESTING
        elapsed = time.monotonic() - start

        logger.info(
            "research_agent_design_complete",
            experiment=result["experiment"].name,
            elapsed=round(elapsed, 2),
        )

        return {
            "experiments": [result["experiment"]],
            "hypotheses": [hypothesis],
            "research_journal": [result["journal_entry"]],
        }

    async def run_experiment(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: EXPERIMENT — execute the latest experiment in sandbox."""
        start = time.monotonic()
        logger.info("research_agent_experiment_start", iteration=state.iteration_count)

        designed = [e for e in state.experiments if e.status.value == "designed"]
        if not designed:
            running = [e for e in state.experiments if e.status.value == "running"]
            if running:
                designed = running
            else:
                logger.warning("research_agent_experiment_no_designed")
                return {
                    "research_journal": [
                        _error_journal("experiment", "No experiment to run")
                    ],
                }

        experiment = designed[-1]

        try:
            result = await self.experiment_runner.run(experiment)
        except Exception as exc:
            logger.exception("research_agent_experiment_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("experiment", f"Execution failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        logger.info(
            "research_agent_experiment_complete",
            experiment=result["experiment"].name,
            status=result["experiment"].status.value,
            elapsed=round(elapsed, 2),
        )

        return {
            "experiments": [result["experiment"]],
            "research_journal": [result["journal_entry"]],
        }

    async def run_analyze(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: ANALYZE — analyze experiment results."""
        start = time.monotonic()
        logger.info("research_agent_analyze_start", iteration=state.iteration_count)

        completed = [
            e for e in state.experiments
            if e.status.value in ("completed", "failed", "timeout")
        ]
        if not completed:
            logger.warning("research_agent_analyze_no_completed_experiments")
            return {
                "research_journal": [
                    _error_journal("analyze", "No completed experiments to analyze")
                ],
            }

        experiment = completed[-1]

        hypothesis = None
        if experiment.hypothesis_id:
            for h in state.hypotheses:
                if h.hypothesis_id == experiment.hypothesis_id:
                    hypothesis = h
                    break

        if hypothesis is None and state.hypotheses:
            hypothesis = state.hypotheses[0]

        if hypothesis is None:
            logger.warning("research_agent_analyze_no_hypothesis")
            return {
                "research_journal": [
                    _error_journal("analyze", "No hypothesis to analyze against")
                ],
            }

        try:
            result = await self.result_analyzer.analyze(
                experiment=experiment,
                hypothesis=hypothesis,
                iteration=state.iteration_count,
            )
        except Exception as exc:
            logger.exception("research_agent_analyze_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("analyze", f"Analysis failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        logger.info(
            "research_agent_analyze_complete",
            hypothesis_status=result["hypothesis_update"].status.value,
            elapsed=round(elapsed, 2),
        )

        return {
            "hypotheses": [result["hypothesis_update"]],
            "analysis_report": result["analysis_report"],
            "research_journal": [result["journal_entry"]],
        }

    async def run_decide(self, state: ResearchState) -> Dict[str, Any]:
        """Phase: DECIDE — decide next research action."""
        start = time.monotonic()
        logger.info("research_agent_decide_start", iteration=state.iteration_count)

        try:
            result = await self.iteration_planner.decide(
                research_topic=state.research_topic,
                hypotheses=state.hypotheses,
                analysis_report=state.analysis_report,
                iteration=state.iteration_count,
                max_iterations=state.max_iterations,
                iteration_history=state.iteration_history,
            )
        except Exception as exc:
            logger.exception("research_agent_decide_failed", error=str(exc))
            return {
                "research_journal": [
                    _error_journal("decide", f"Decision failed: {exc}")
                ],
            }

        elapsed = time.monotonic() - start
        decision = result["decision"]

        logger.info(
            "research_agent_decide_complete",
            decision=decision.value,
            iteration=state.iteration_count,
            elapsed=round(elapsed, 2),
        )

        update: Dict[str, Any] = {
            "iteration_history": [result["iteration_record"]],
            "research_journal": [result["journal_entry"]],
        }

        if decision == ResearchDecision.CONCLUDE:
            update["final_report"] = result.get("final_report", "")
        elif decision == ResearchDecision.ITERATE:
            update["iteration_count"] = state.iteration_count + 1

            direction = result["iteration_record"].direction
            if direction:
                ctx = dict(state.domain_context)
                existing_queries = ctx.get("refined_queries", [])
                ctx["refined_queries"] = [direction] + existing_queries[:4]
                update["domain_context"] = ctx
        elif decision == ResearchDecision.PIVOT:
            update["iteration_count"] = state.iteration_count + 1

        return update


def _error_journal(phase: str, message: str) -> JournalEntry:
    """Create an error journal entry."""
    from app.core.types.research_types import JournalEntry as JE

    return JE(
        phase=phase,
        summary=f"ERROR: {message}",
        details={"error": True},
    )
