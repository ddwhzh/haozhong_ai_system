"""arXiv API client for academic paper retrieval.

Uses the arXiv REST API (Atom feed) for keyword, author, and category searches.
Returns normalized Paper objects for consumption by the research pipeline.

API docs: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from app.core.logging import logger

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_API_URL = "https://export.arxiv.org/api/query"


class ArxivClient:
    """Async arXiv search client."""

    def __init__(
        self,
        timeout_seconds: float = 15.0,
        max_results: int = 20,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        categories: Optional[List[str]] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search query string.
            max_results: Override default max results.
            categories: Optional arXiv category filters (e.g. ["cs.AI", "cs.CL"]).
            sort_by: Sort criterion ("relevance", "lastUpdatedDate", "submittedDate").
            sort_order: "ascending" or "descending".

        Returns:
            List of normalized paper dicts.
        """
        if not query or not query.strip():
            return []

        search_query = self._build_search_query(query, categories)
        cap = max_results or self.max_results

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(cap, 50),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(ARXIV_API_URL, params=params)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.warning("arxiv_search_timeout", query=query[:100], error=str(exc))
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "arxiv_search_http_error",
                query=query[:100],
                status_code=exc.response.status_code,
                error=str(exc),
            )
            return []
        except httpx.RequestError as exc:
            logger.error("arxiv_search_request_error", query=query[:100], error=str(exc))
            return []

        return self._parse_atom_feed(response.text)

    def _build_search_query(
        self, query: str, categories: Optional[List[str]] = None
    ) -> str:
        """Build an arXiv API search query string."""
        parts = [f"all:{query}"]
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            parts.append(f"({cat_query})")
        return " AND ".join(parts)

    def _parse_atom_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse arXiv Atom XML feed into normalized paper dicts."""
        papers: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("arxiv_xml_parse_error", error=str(exc))
            return []

        for entry in root.findall("atom:entry", ARXIV_NS):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        logger.info("arxiv_search_parsed", paper_count=len(papers))
        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single Atom entry into a paper dict."""
        title_el = entry.find("atom:title", ARXIV_NS)
        summary_el = entry.find("atom:summary", ARXIV_NS)
        id_el = entry.find("atom:id", ARXIV_NS)
        published_el = entry.find("atom:published", ARXIV_NS)

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_url = (id_el.text or "").strip() if id_el is not None else ""

        if not title:
            return None

        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        year = None
        if published_el is not None and published_el.text:
            try:
                year = int(published_el.text[:4])
            except (ValueError, IndexError):
                pass

        categories = []
        for cat_el in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall("{http://www.w3.org/2005/Atom}category"):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        pdf_url = ""
        for link_el in entry.findall("atom:link", ARXIV_NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "url": arxiv_url,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
            "categories": categories,
            "source": "arxiv",
        }
