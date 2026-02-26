"""Iteration Planner — decides whether to iterate, pivot, or conclude.

Input: full research state summary
Output: decision (iterate/pivot/conclude), direction, key findings
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.research_types import (
    Hypothesis,
    HypothesisStatus,
    IterationRecord,
    JournalEntry,
    ResearchDecision,
)
from app.infra.llm.service import llm_service
from app.prompts import load_prompt


class IterationPlanner:
    """Decide the next action in the research loop."""

    async def decide(
        self,
        research_topic: str,
        hypotheses: List[Hypothesis],
        analysis_report: str,
        iteration: int,
        max_iterations: int,
        iteration_history: List[IterationRecord],
    ) -> Dict[str, Any]:
        """Make iteration decision.

        Returns:
            dict with keys: decision, iteration_record, journal_entry, final_report
        """
        logger.info(
            "iteration_planner_start",
            iteration=iteration,
            max_iterations=max_iterations,
            hypothesis_count=len(hypotheses),
        )

        if iteration >= max_iterations:
            logger.info("iteration_planner_force_conclude", iteration=iteration)
            return self._force_conclude(
                research_topic, hypotheses, analysis_report, iteration, iteration_history
            )

        hypotheses_summary = self._summarize_hypotheses(hypotheses)
        history_summary = self._summarize_history(iteration_history)

        system_prompt = load_prompt(
            "research_decide_system",
            research_topic=research_topic,
            iteration=iteration,
            max_iterations=max_iterations,
            hypotheses_summary=hypotheses_summary,
            results_summary=analysis_report[:2000],
            iteration_history=history_summary,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Based on the research context above, what should we do next?"),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_decide_system")
            result = self._parse_json_response(response.content)
        except Exception as exc:
            logger.exception("iteration_planner_llm_error", error=str(exc))
            result = {
                "decision": "conclude",
                "reasoning": f"LLM error, concluding: {exc}",
                "direction": "",
                "key_findings": [],
                "report_summary": "",
            }

        decision_str = result.get("decision", "conclude")
        decision_map = {
            "iterate": ResearchDecision.ITERATE,
            "pivot": ResearchDecision.PIVOT,
            "conclude": ResearchDecision.CONCLUDE,
        }
        decision = decision_map.get(decision_str, ResearchDecision.CONCLUDE)

        confirmed = sum(1 for h in hypotheses if h.status == HypothesisStatus.CONFIRMED)

        iteration_record = IterationRecord(
            iteration=iteration,
            decision=decision,
            direction=result.get("direction", ""),
            key_findings=result.get("key_findings", []),
            hypotheses_tested=len(hypotheses),
            hypotheses_confirmed=confirmed,
            experiments_run=sum(len(h.test_results) for h in hypotheses),
        )

        raw_final = result.get("report_summary", "") if decision == ResearchDecision.CONCLUDE else ""
        if isinstance(raw_final, list):
            final_report = "\n".join(str(item) for item in raw_final)
        else:
            final_report = str(raw_final)

        journal = JournalEntry(
            phase="decide",
            summary=f"Decision: {decision.value} (iteration {iteration}/{max_iterations})",
            details={
                "decision": decision.value,
                "reasoning": result.get("reasoning", ""),
                "direction": result.get("direction", ""),
            },
        )

        logger.info(
            "iteration_planner_complete",
            decision=decision.value,
            iteration=iteration,
        )

        return {
            "decision": decision,
            "iteration_record": iteration_record,
            "journal_entry": journal,
            "final_report": final_report,
        }

    def _force_conclude(
        self,
        research_topic: str,
        hypotheses: List[Hypothesis],
        analysis_report: str,
        iteration: int,
        iteration_history: List[IterationRecord],
    ) -> Dict[str, Any]:
        """Force conclusion when max iterations reached."""
        confirmed = sum(1 for h in hypotheses if h.status == HypothesisStatus.CONFIRMED)
        key_findings = [
            f"Hypothesis: {h.statement} -> {h.status.value}"
            for h in hypotheses
        ]

        iteration_record = IterationRecord(
            iteration=iteration,
            decision=ResearchDecision.CONCLUDE,
            direction="",
            key_findings=key_findings,
            hypotheses_tested=len(hypotheses),
            hypotheses_confirmed=confirmed,
        )

        journal = JournalEntry(
            phase="decide",
            summary=f"Forced conclude at max iteration {iteration}",
            details={"reason": "max_iterations_reached"},
        )

        return {
            "decision": ResearchDecision.CONCLUDE,
            "iteration_record": iteration_record,
            "journal_entry": journal,
            "final_report": "",
        }

    def _summarize_hypotheses(self, hypotheses: List[Hypothesis]) -> str:
        """Build a text summary of hypothesis statuses."""
        if not hypotheses:
            return "No hypotheses generated"
        lines = []
        for h in hypotheses:
            lines.append(f"- [{h.status.value}] {h.statement} (confidence={h.confidence:.2f})")
        return "\n".join(lines)

    def _summarize_history(self, history: List[IterationRecord]) -> str:
        """Build a text summary of past iterations."""
        if not history:
            return "First iteration"
        lines = []
        for r in history:
            lines.append(
                f"Iteration {r.iteration}: {r.decision.value} -> {r.direction or 'N/A'}"
            )
        return "\n".join(lines)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("iteration_planner_json_parse_failed", text=cleaned[:200])
            return {
                "decision": "conclude",
                "reasoning": "Failed to parse LLM response",
                "direction": "",
                "key_findings": [],
                "report_summary": "",
            }
