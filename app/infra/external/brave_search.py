"""Brave Search API client (optional retrieval source).

Provides a lightweight async client for Brave Web Search.
If API key is not configured, callers should treat it as disabled.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx

from app.core.logging import logger


class BraveSearchClient:
    """Async Brave Search HTTP client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def web_search(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """Search web with Brave and return normalized results."""
        if not self.enabled:
            return []
        if not query or not query.strip():
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": max(1, min(int(count), 20)),
        }

        timeout = httpx.Timeout(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.base_url, headers=headers, params=params)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.warning("brave_search_timeout", query=query[:100], error=str(exc))
            return []
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            logger.error(
                "brave_search_http_error",
                query=query[:100],
                status_code=status_code,
                error=str(exc),
            )
            return []
        except httpx.RequestError as exc:
            logger.error("brave_search_request_error", query=query[:100], error=str(exc))
            return []

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("brave_search_invalid_json", query=query[:100], error=str(exc))
            return []

        web_results = payload.get("web", {}).get("results", [])
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(web_results):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            description = (item.get("description") or "").strip()
            if not (title or description or url):
                continue
            normalized.append(
                {
                    "rank": idx,
                    "title": title,
                    "url": url,
                    "description": description,
                }
            )
        return normalized
