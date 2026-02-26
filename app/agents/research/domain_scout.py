"""Domain Scout — explores the research landscape via web search + LLM analysis.

Input: research topic
Output: domain context (keywords, research directions, core problems, refined queries)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_types import JournalEntry
from app.infra.external.brave_search import BraveSearchClient
from app.infra.llm.service import llm_service
from app.prompts import load_prompt


class DomainScout:
    """Explore a research domain and produce structured domain context."""

    def __init__(self) -> None:
        self.brave_client = BraveSearchClient(
            api_key=settings.BRAVE_SEARCH_API_KEY,
            base_url=settings.BRAVE_SEARCH_BASE_URL,
            timeout_seconds=settings.BRAVE_SEARCH_TIMEOUT_SECONDS,
        )

    async def scout(self, research_topic: str) -> Dict[str, Any]:
        """Run domain exploration.

        Returns:
            dict with keys: domain_context, journal_entry
        """
        logger.info("domain_scout_start", topic=research_topic[:100])

        web_results = await self._search_web(research_topic)
        domain_context = await self._analyze_with_llm(research_topic, web_results)

        journal = JournalEntry(
            phase="scout",
            summary=f"Explored domain for '{research_topic[:80]}', "
            f"found {len(domain_context.get('directions', []))} directions, "
            f"{len(domain_context.get('keywords', []))} keywords",
            details={
                "web_results_count": len(web_results),
                "directions_count": len(domain_context.get("directions", [])),
            },
        )

        logger.info(
            "domain_scout_complete",
            directions=len(domain_context.get("directions", [])),
            keywords=len(domain_context.get("keywords", [])),
        )

        return {
            "domain_context": domain_context,
            "journal_entry": journal,
        }

    async def _search_web(self, topic: str) -> List[Dict[str, Any]]:
        """Search the web for the research topic."""
        if not self.brave_client.enabled:
            logger.warning("domain_scout_brave_disabled")
            return []

        queries = [
            topic,
            f"{topic} survey 2025 2026",
            f"{topic} state of the art recent advances",
        ]

        all_results: List[Dict[str, Any]] = []
        for q in queries:
            results = await self.brave_client.web_search(q, count=5)
            all_results.extend(results)

        seen_urls = set()
        deduped: List[Dict[str, Any]] = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(r)

        return deduped[:15]

    async def _analyze_with_llm(
        self, topic: str, web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to analyze web results and extract domain context."""
        system_prompt = load_prompt("research_scout_system", research_topic=topic)

        results_text = "\n".join(
            f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}): {r.get('description', '')}"
            for r in web_results
        )

        user_content = (
            f"Research topic: {topic}\n\n"
            f"Web search results:\n{results_text}\n\n"
            "Analyze these results and extract the domain context as specified."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_scout_system")
            return self._parse_json_response(response.content)
        except Exception as exc:
            logger.exception("domain_scout_llm_error", error=str(exc))
            return {
                "keywords": [],
                "directions": [],
                "core_problems": [],
                "refined_queries": [topic],
                "summary": f"LLM analysis failed: {exc}",
            }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
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
            logger.warning("domain_scout_json_parse_failed", text=cleaned[:200])
            return {
                "keywords": [],
                "directions": [],
                "core_problems": [],
                "refined_queries": [],
                "summary": cleaned[:500],
            }
