"""Result Analyzer — analyzes experiment results against hypotheses.

Input: experiment with results, hypothesis
Output: analysis report, hypothesis status update
"""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.research_types import (
    Experiment,
    Hypothesis,
    HypothesisStatus,
    JournalEntry,
)
from app.infra.llm.service import llm_service
from app.prompts import load_prompt


class ResultAnalyzer:
    """Analyze experiment results and validate hypotheses."""

    async def analyze(
        self,
        experiment: Experiment,
        hypothesis: Hypothesis,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """Analyze experiment results.

        Returns:
            dict with keys: analysis_report, hypothesis_update, journal_entry
        """
        logger.info(
            "result_analysis_start",
            experiment_name=experiment.name,
            hypothesis=hypothesis.statement[:80],
        )

        if experiment.result is None:
            return self._empty_analysis(experiment, hypothesis)

        system_prompt = load_prompt(
            "research_analyze_system",
            hypothesis=hypothesis.statement,
            experiment_name=experiment.name,
            exit_code=experiment.result.exit_code,
            stdout=experiment.result.stdout[:3000],
            stderr=experiment.result.stderr[:1000],
            iteration=iteration,
        )

        user_content = (
            f"Experiment '{experiment.name}' has completed.\n\n"
            f"Hypothesis: {hypothesis.statement}\n"
            f"Success criteria: {experiment.success_criteria}\n"
            f"Expected metrics: {experiment.expected_metrics}\n\n"
            f"Exit code: {experiment.result.exit_code}\n"
            f"Elapsed: {experiment.result.elapsed_seconds}s\n\n"
            f"Parsed metrics: {json.dumps(experiment.result.metrics, indent=2)}\n\n"
            "Analyze these results against the hypothesis."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_analyze_system")
            result = self._parse_json_response(response.content)
        except Exception as exc:
            logger.exception("result_analysis_llm_error", error=str(exc))
            result = {
                "hypothesis_status": "inconclusive",
                "confidence": 0.3,
                "analysis_summary": f"Analysis failed: {exc}",
                "findings": [],
            }

        status_str = result.get("hypothesis_status", "inconclusive")
        status_map = {
            "confirmed": HypothesisStatus.CONFIRMED,
            "rejected": HypothesisStatus.REJECTED,
            "inconclusive": HypothesisStatus.INCONCLUSIVE,
        }
        new_status = status_map.get(status_str, HypothesisStatus.INCONCLUSIVE)

        hypothesis.status = new_status
        hypothesis.confidence = float(result.get("confidence", 0.5))
        hypothesis.test_results.append(experiment.experiment_id)

        raw_report = result.get("analysis_summary", "")
        if isinstance(raw_report, list):
            analysis_report = "\n".join(str(item) for item in raw_report)
        else:
            analysis_report = str(raw_report)

        journal = JournalEntry(
            phase="analyze",
            summary=(
                f"Analyzed experiment '{experiment.name}': "
                f"hypothesis {new_status.value} (confidence={hypothesis.confidence:.2f})"
            ),
            details={
                "hypothesis_status": new_status.value,
                "confidence": hypothesis.confidence,
                "key_metrics": result.get("key_metrics", {}),
                "findings_count": len(result.get("findings", [])),
            },
        )

        logger.info(
            "result_analysis_complete",
            hypothesis_status=new_status.value,
            confidence=hypothesis.confidence,
        )

        return {
            "analysis_report": analysis_report,
            "hypothesis_update": hypothesis,
            "improvement_suggestions": result.get("improvement_suggestions", []),
            "journal_entry": journal,
        }

    def _empty_analysis(
        self, experiment: Experiment, hypothesis: Hypothesis
    ) -> Dict[str, Any]:
        """Return analysis for an experiment with no results."""
        hypothesis.status = HypothesisStatus.INCONCLUSIVE
        return {
            "analysis_report": f"Experiment '{experiment.name}' produced no results",
            "hypothesis_update": hypothesis,
            "improvement_suggestions": ["Rerun the experiment or check code"],
            "journal_entry": JournalEntry(
                phase="analyze",
                summary=f"No results for experiment '{experiment.name}'",
            ),
        }

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
            logger.warning("result_analysis_json_parse_failed", text=cleaned[:200])
            return {
                "hypothesis_status": "inconclusive",
                "confidence": 0.3,
                "analysis_summary": cleaned[:500],
                "findings": [],
            }
