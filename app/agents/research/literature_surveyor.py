"""Literature Surveyor — retrieves academic papers from arXiv + Semantic Scholar.

Input: domain context with refined queries, research topic
Output: list of Paper objects, survey assessment journal entry
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_types import JournalEntry, Paper, PaperSource
from app.infra.external.arxiv_client import ArxivClient
from app.infra.external.semantic_scholar import SemanticScholarClient
from app.infra.llm.service import llm_service
from app.prompts import load_prompt


class LiteratureSurveyor:
    """Retrieve and assess academic papers from multiple sources."""

    def __init__(self) -> None:
        self.arxiv_client = ArxivClient(
            timeout_seconds=settings.RESEARCH_ARXIV_TIMEOUT,
            max_results=settings.RESEARCH_MAX_PAPERS_PER_SOURCE,
        )
        self.s2_client = SemanticScholarClient(
            api_key=settings.RESEARCH_S2_API_KEY,
            timeout_seconds=settings.RESEARCH_S2_TIMEOUT,
            max_results=settings.RESEARCH_MAX_PAPERS_PER_SOURCE,
        )

    async def survey(
        self,
        research_topic: str,
        domain_context: Dict[str, Any],
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """Run literature survey.

        Returns:
            dict with keys: papers, survey_assessment, journal_entry
        """
        logger.info("literature_survey_start", topic=research_topic[:80], iteration=iteration)

        queries = domain_context.get("refined_queries", [research_topic])
        if not queries:
            queries = [research_topic]

        raw_papers = await self._retrieve_papers(queries)
        papers = self._deduplicate(raw_papers)
        assessment = await self._assess_papers(research_topic, papers, iteration)

        self._apply_relevance_scores(papers, assessment)

        papers.sort(key=lambda p: p.relevance_score, reverse=True)

        journal = JournalEntry(
            phase="survey",
            summary=f"Retrieved {len(papers)} papers across {len(queries)} queries (iteration {iteration})",
            details={
                "query_count": len(queries),
                "total_papers": len(papers),
                "arxiv_papers": sum(1 for p in papers if p.source == PaperSource.ARXIV),
                "s2_papers": sum(1 for p in papers if p.source == PaperSource.SEMANTIC_SCHOLAR),
            },
        )

        logger.info(
            "literature_survey_complete",
            paper_count=len(papers),
            iteration=iteration,
        )

        return {
            "papers": papers,
            "survey_assessment": assessment,
            "journal_entry": journal,
        }

    async def _retrieve_papers(self, queries: List[str]) -> List[Paper]:
        """Retrieve papers from all sources for all queries."""
        all_papers: List[Paper] = []

        for query in queries[:5]:
            arxiv_results = await self.arxiv_client.search(query)
            for r in arxiv_results:
                all_papers.append(Paper(
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    abstract=r.get("abstract", ""),
                    year=r.get("year"),
                    url=r.get("url", ""),
                    arxiv_id=r.get("arxiv_id"),
                    categories=r.get("categories", []),
                    source=PaperSource.ARXIV,
                ))

            s2_results = await self.s2_client.search(query)
            for r in s2_results:
                all_papers.append(Paper(
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    abstract=r.get("abstract", ""),
                    year=r.get("year"),
                    url=r.get("url", ""),
                    doi=r.get("doi"),
                    arxiv_id=r.get("arxiv_id"),
                    citation_count=r.get("citation_count", 0),
                    categories=r.get("categories", []),
                    source=PaperSource.SEMANTIC_SCHOLAR,
                    metadata={"s2_paper_id": r.get("s2_paper_id", "")},
                ))

        return all_papers

    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """Deduplicate papers by DOI, arXiv ID, or normalized title."""
        seen = set()
        unique: List[Paper] = []
        for p in papers:
            key = p.dedup_key()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    async def _assess_papers(
        self,
        research_topic: str,
        papers: List[Paper],
        iteration: int,
    ) -> Dict[str, Any]:
        """Use LLM to assess paper relevance and survey quality."""
        if not papers:
            return {"paper_assessments": [], "common_themes": [], "gaps": []}

        system_prompt = load_prompt(
            "research_survey_system",
            research_topic=research_topic,
            iteration=iteration,
        )

        papers_text = "\n\n".join(
            f"Title: {p.title}\nAuthors: {', '.join(p.authors[:3])}\n"
            f"Year: {p.year or 'N/A'}\nAbstract: {p.abstract[:300]}"
            for p in papers[:30]
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Papers to assess:\n\n{papers_text}"),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_survey_system")
            return self._parse_json_response(response.content)
        except Exception as exc:
            logger.exception("literature_survey_llm_error", error=str(exc))
            return {"paper_assessments": [], "common_themes": [], "gaps": []}

    def _apply_relevance_scores(
        self, papers: List[Paper], assessment: Dict[str, Any]
    ) -> None:
        """Apply LLM-assessed relevance scores back to papers."""
        title_to_score = {}
        for pa in assessment.get("paper_assessments", []):
            title = pa.get("title", "").strip().lower()
            score = pa.get("relevance", 0.5)
            if title:
                title_to_score[title] = score

        for p in papers:
            key = p.title.strip().lower()
            if key in title_to_score:
                p.relevance_score = title_to_score[key]

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
            logger.warning("literature_survey_json_parse_failed", text=cleaned[:200])
            return {"paper_assessments": [], "common_themes": [], "gaps": []}
