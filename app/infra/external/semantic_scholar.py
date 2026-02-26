"""Semantic Scholar API client for academic paper retrieval and citation graphs.

Uses the Semantic Scholar Academic Graph API for paper search, details,
citations, and references. No API key required for basic usage (rate-limited).

API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from app.core.logging import logger

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_URL = f"{S2_BASE_URL}/paper/search"

PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,citationCount,"
    "url,externalIds,fieldsOfStudy,publicationDate"
)


class SemanticScholarClient:
    """Async Semantic Scholar API client."""

    def __init__(
        self,
        api_key: str = "",
        timeout_seconds: float = 15.0,
        max_results: int = 20,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        year_range: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for papers by keyword.

        Args:
            query: Search query string.
            max_results: Override default limit.
            year_range: Filter by year range (e.g. "2023-2026").
            fields_of_study: Filter by field (e.g. ["Computer Science"]).

        Returns:
            List of normalized paper dicts.
        """
        if not query or not query.strip():
            return []

        cap = max_results or self.max_results
        params: Dict[str, Any] = {
            "query": query,
            "limit": min(cap, 100),
            "fields": PAPER_FIELDS,
        }
        if year_range:
            params["year"] = year_range
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    S2_PAPER_SEARCH_URL,
                    params=params,
                    headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.warning("s2_search_timeout", query=query[:100], error=str(exc))
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "s2_search_http_error",
                query=query[:100],
                status_code=exc.response.status_code,
                error=str(exc),
            )
            return []
        except httpx.RequestError as exc:
            logger.error("s2_search_request_error", query=query[:100], error=str(exc))
            return []

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("s2_search_invalid_json", query=query[:100], error=str(exc))
            return []

        raw_papers = payload.get("data", [])
        papers = [self._normalize_paper(p) for p in raw_papers if p]

        logger.info("s2_search_complete", query=query[:80], paper_count=len(papers))
        return papers

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID.
        """
        if not paper_id:
            return None

        url = f"{S2_BASE_URL}/paper/{paper_id}"
        params = {"fields": PAPER_FIELDS}

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url, params=params, headers=self._headers()
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "s2_paper_details_error",
                paper_id=paper_id,
                status_code=exc.response.status_code,
            )
            return None
        except httpx.RequestError as exc:
            logger.error("s2_paper_details_request_error", paper_id=paper_id, error=str(exc))
            return None

        try:
            data = response.json()
        except ValueError:
            return None

        return self._normalize_paper(data) if data else None

    async def get_citations(
        self, paper_id: str, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper."""
        return await self._get_related(paper_id, "citations", max_results)

    async def get_references(
        self, paper_id: str, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Get papers referenced by the given paper."""
        return await self._get_related(paper_id, "references", max_results)

    async def _get_related(
        self, paper_id: str, relation: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch citations or references for a paper."""
        if not paper_id:
            return []

        url = f"{S2_BASE_URL}/paper/{paper_id}/{relation}"
        params = {
            "fields": "paperId,title,abstract,year,authors,citationCount,url",
            "limit": min(max_results, 100),
        }

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url, params=params, headers=self._headers()
                )
                response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning(
                "s2_related_papers_error",
                paper_id=paper_id,
                relation=relation,
                error=str(exc),
            )
            return []

        try:
            payload = response.json()
        except ValueError:
            return []

        results = []
        for item in payload.get("data", []):
            cited = item.get("citingPaper" if relation == "citations" else "citedPaper")
            if cited:
                normalized = self._normalize_paper(cited)
                if normalized.get("title"):
                    results.append(normalized)

        return results

    def _normalize_paper(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Semantic Scholar paper response into standard format."""
        authors = []
        for a in raw.get("authors", []) or []:
            name = a.get("name", "")
            if name:
                authors.append(name)

        external_ids = raw.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")

        return {
            "title": (raw.get("title") or "").strip(),
            "authors": authors,
            "abstract": (raw.get("abstract") or "").strip(),
            "year": raw.get("year"),
            "url": raw.get("url", ""),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "citation_count": raw.get("citationCount", 0) or 0,
            "categories": raw.get("fieldsOfStudy", []) or [],
            "s2_paper_id": raw.get("paperId", ""),
            "source": "semantic_scholar",
        }
