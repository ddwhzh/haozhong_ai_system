"""Knowledge Synthesizer — synthesizes literature into hypotheses.

Input: research topic, domain context, literature, survey assessment
Output: synthesis report, hypotheses, updated domain context
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.research_types import Hypothesis, JournalEntry, Paper
from app.infra.llm.service import llm_service
from app.prompts import load_prompt


class KnowledgeSynthesizer:
    """Synthesize literature findings and generate testable hypotheses."""

    async def synthesize(
        self,
        research_topic: str,
        domain_context: Dict[str, Any],
        papers: List[Paper],
        survey_assessment: Dict[str, Any],
        iteration: int = 1,
        previous_findings: str = "",
    ) -> Dict[str, Any]:
        """Run knowledge synthesis.

        Returns:
            dict with keys: synthesis_report, hypotheses, journal_entry
        """
        logger.info(
            "knowledge_synthesis_start",
            topic=research_topic[:80],
            paper_count=len(papers),
            iteration=iteration,
        )

        gaps = survey_assessment.get("gaps", [])

        system_prompt = load_prompt(
            "research_synthesize_system",
            research_topic=research_topic,
            domain_context=json.dumps(domain_context, ensure_ascii=False)[:1000],
            gaps=json.dumps(gaps, ensure_ascii=False),
            iteration=iteration,
            previous_findings=f"Previous findings:\n{previous_findings}" if previous_findings else "",
        )

        papers_text = "\n\n".join(
            f"[{p.title}] ({p.year or 'N/A'}): {p.abstract[:300]}"
            for p in sorted(papers, key=lambda x: x.relevance_score, reverse=True)[:20]
        )

        themes = survey_assessment.get("common_themes", [])
        themes_text = "\n".join(f"- {t}" for t in themes) if themes else "None identified"

        user_content = (
            f"Literature ({len(papers)} papers, top 20 shown):\n\n{papers_text}\n\n"
            f"Common themes:\n{themes_text}\n\n"
            f"Identified gaps:\n{json.dumps(gaps, ensure_ascii=False)}\n\n"
            "Synthesize these findings and generate testable hypotheses."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_synthesize_system")
            result = self._parse_json_response(response.content)
        except Exception as exc:
            logger.exception("knowledge_synthesis_llm_error", error=str(exc))
            result = {
                "synthesis": f"Synthesis failed: {exc}",
                "hypotheses": [],
                "contradictions": [],
            }

        hypotheses = self._build_hypotheses(result.get("hypotheses", []))
        raw_synthesis = result.get("synthesis", "")
        if isinstance(raw_synthesis, list):
            synthesis_report = "\n".join(str(item) for item in raw_synthesis)
        else:
            synthesis_report = str(raw_synthesis)

        journal = JournalEntry(
            phase="synthesize",
            summary=f"Synthesized {len(papers)} papers, generated {len(hypotheses)} hypotheses (iteration {iteration})",
            details={
                "paper_count": len(papers),
                "hypothesis_count": len(hypotheses),
                "contradictions_found": len(result.get("contradictions", [])),
            },
        )

        logger.info(
            "knowledge_synthesis_complete",
            hypothesis_count=len(hypotheses),
            iteration=iteration,
        )

        return {
            "synthesis_report": synthesis_report,
            "hypotheses": hypotheses,
            "journal_entry": journal,
        }

    def _build_hypotheses(self, raw_hypotheses: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Convert raw LLM hypothesis dicts to Hypothesis objects."""
        hypotheses: List[Hypothesis] = []
        for h in raw_hypotheses[:5]:
            hypotheses.append(Hypothesis(
                statement=h.get("statement", ""),
                rationale=h.get("rationale", ""),
                supporting_papers=h.get("supporting_papers", []),
                confidence=min(
                    float(h.get("testability", 0.5)) * float(h.get("impact", 0.5)),
                    1.0,
                ),
            ))
        return hypotheses

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
            logger.warning("knowledge_synthesis_json_parse_failed", text=cleaned[:200])
            return {"synthesis": cleaned[:500], "hypotheses": [], "contradictions": []}
