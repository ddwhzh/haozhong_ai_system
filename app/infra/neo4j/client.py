"""Neo4j async client for knowledge graph operations.

Provides BFS-style multi-hop traversal, node/edge CRUD, and
subgraph extraction for the retrieval agent's depth-first pipeline.
"""

from __future__ import annotations

from time import monotonic
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import (
    AuthError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger


class Neo4jClient:
    """Async Neo4j driver wrapper with connection lifecycle management."""

    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None
        self._reconnect_blocked_until: float = 0.0
        self._last_connection_error: str = ""

    @retry(
        stop=stop_after_attempt(max(1, settings.NEO4J_CONNECT_MAX_RETRIES)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (ServiceUnavailable, SessionExpired, OSError)
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _verify_connectivity_with_retry(self) -> None:
        """Verify Neo4j connectivity with exponential backoff."""
        if self._driver is None:
            raise RuntimeError("neo4j_driver_not_initialized")
        await self._driver.verify_connectivity()

    async def _reset_driver(self) -> None:
        """Close and reset driver instance safely."""
        if self._driver is None:
            return
        driver = self._driver
        self._driver = None
        try:
            await driver.close()
        except (ServiceUnavailable, SessionExpired, OSError, RuntimeError) as e:
            logger.warning("neo4j_driver_close_failed", error=str(e))

    async def connect(self) -> None:
        """Initialise the Neo4j driver."""
        if self._driver is not None:
            return
        now = monotonic()
        if now < self._reconnect_blocked_until:
            cooldown_remaining = round(self._reconnect_blocked_until - now, 2)
            logger.warning(
                "neo4j_reconnect_cooldown_active",
                cooldown_seconds_remaining=cooldown_remaining,
                last_error=self._last_connection_error[:300],
            )
            raise ServiceUnavailable("neo4j reconnect cooldown active")
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_pool_size=settings.NEO4J_POOL_SIZE,
            )
            await self._verify_connectivity_with_retry()
            self._reconnect_blocked_until = 0.0
            self._last_connection_error = ""
            logger.info("neo4j_connected", uri=settings.NEO4J_URI)
        except (
            ServiceUnavailable,
            SessionExpired,
            AuthError,
            Neo4jError,
            OSError,
        ) as e:
            self._last_connection_error = str(e)
            self._reconnect_blocked_until = (
                monotonic() + settings.NEO4J_RETRY_COOLDOWN_SECONDS
            )
            await self._reset_driver()
            logger.exception(
                "neo4j_connection_failed",
                uri=settings.NEO4J_URI,
                cooldown_seconds=settings.NEO4J_RETRY_COOLDOWN_SECONDS,
                error=str(e),
            )
            raise

    async def close(self) -> None:
        """Close the driver."""
        if self._driver is None:
            return
        await self._reset_driver()
        logger.info("neo4j_disconnected")

    async def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            if self._driver is None:
                await self.connect()
            await self._driver.verify_connectivity()
            return True
        except (
            ServiceUnavailable,
            SessionExpired,
            AuthError,
            Neo4jError,
            OSError,
            RuntimeError,
        ):
            return False

    # ── Query execution ──────────────────────────────────────────────────

    async def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a read transaction and return list of record dicts."""
        if self._driver is None:
            await self.connect()
        try:
            async with self._driver.session(
                database=database or settings.NEO4J_DATABASE
            ) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except (ServiceUnavailable, SessionExpired, OSError, Neo4jError):
            await self._reset_driver()
            raise

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a write transaction and return list of record dicts."""
        if self._driver is None:
            await self.connect()
        try:
            async with self._driver.session(
                database=database or settings.NEO4J_DATABASE
            ) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except (ServiceUnavailable, SessionExpired, OSError, Neo4jError):
            await self._reset_driver()
            raise

    # ── BFS multi-hop traversal ──────────────────────────────────────────

    async def bfs_traverse(
        self,
        start_node_ids: List[str],
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None,
        limit_per_hop: int = 20,
    ) -> Dict[str, Any]:
        """BFS traversal from start nodes up to max_hops depth.

        Returns a subgraph dict: {nodes: [...], edges: [...], paths: [...]}.
        This is the core primitive for the retrieval agent's depth pipeline.
        """
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"

        query = f"""
        MATCH path = (start)-[r{rel_filter}*1..{max_hops}]-(connected)
        WHERE start.id IN $start_ids
        WITH path, connected,
             [rel IN relationships(path) | type(rel)] AS rel_types,
             length(path) AS depth
        ORDER BY depth ASC
        WITH collect(DISTINCT connected) AS all_connected,
             collect(DISTINCT path) AS all_paths
        UNWIND all_connected AS node
        WITH node, all_paths
        LIMIT {limit_per_hop * max_hops}
        RETURN
            collect(DISTINCT {{
                id: node.id,
                labels: labels(node),
                properties: properties(node)
            }}) AS nodes,
            all_paths
        """
        try:
            records = await self.execute_read(
                query, {"start_ids": start_node_ids}
            )
            if not records:
                return {"nodes": [], "edges": [], "depth": 0}

            return self._parse_bfs_result(records, max_hops)
        except (ServiceUnavailable, SessionExpired, OSError, Neo4jError) as e:
            logger.warning(
                "bfs_traverse_failed",
                start_nodes=start_node_ids,
                max_hops=max_hops,
                error=str(e),
            )
            return {"nodes": [], "edges": [], "depth": 0}

    async def get_subgraph(
        self,
        node_ids: List[str],
    ) -> Dict[str, Any]:
        """Extract a subgraph containing exactly the given node IDs and
        all edges between them.  Used to split query subgraph -> reasoning
        chain subgraph.
        """
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IN $node_ids AND b.id IN $node_ids
        RETURN
            collect(DISTINCT {
                id: a.id, labels: labels(a), properties: properties(a)
            }) +
            collect(DISTINCT {
                id: b.id, labels: labels(b), properties: properties(b)
            }) AS nodes,
            collect({
                source: a.id,
                target: b.id,
                type: type(r),
                properties: properties(r)
            }) AS edges
        """
        records = await self.execute_read(query, {"node_ids": node_ids})
        if not records:
            return {"nodes": [], "edges": []}
        row = records[0]
        # Deduplicate nodes by id
        seen = set()
        unique_nodes = []
        for n in row.get("nodes", []):
            nid = n.get("id")
            if nid and nid not in seen:
                seen.add(nid)
                unique_nodes.append(n)
        return {"nodes": unique_nodes, "edges": row.get("edges", [])}

    async def find_entities(
        self,
        text: str,
        label: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search for entities whose name matches *text*.

        Supports bidirectional CONTAINS:
        - entity name in query text (e.g. '知识图谱' found in '什么是知识图谱')
        - query text in entity name/description
        """
        label_clause = f":{label}" if label else ""
        query = f"""
        MATCH (n{label_clause})
        WHERE toLower(n.name) CONTAINS toLower($text)
           OR toLower(n.description) CONTAINS toLower($text)
           OR toLower($text) CONTAINS toLower(n.name)
        RETURN {{
            id: n.id,
            name: n.name,
            labels: labels(n),
            properties: properties(n)
        }} AS entity
        LIMIT {limit}
        """
        records = await self.execute_read(query, {"text": text})
        return [r["entity"] for r in records]

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_bfs_result(
        records: List[Dict[str, Any]],
        max_depth: int,
    ) -> Dict[str, Any]:
        """Parse raw Neo4j BFS records into a structured subgraph dict."""
        nodes = []
        edges = []
        if records:
            row = records[0]
            nodes = row.get("nodes", [])
        return {
            "nodes": nodes,
            "edges": edges,
            "depth": max_depth,
        }


# Singleton
neo4j_client = Neo4jClient()
