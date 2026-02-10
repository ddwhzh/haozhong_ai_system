"""Cypher query templates for the knowledge graph.

Centralises all KG query patterns so that agents never build raw Cypher.
Each method returns a (query_string, parameters_dict) tuple.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class KGQueries:
    """Static Cypher query builder for knowledge graph operations."""

    @staticmethod
    def find_shortest_paths(
        source_id: str,
        target_id: str,
        max_hops: int = 5,
    ) -> Tuple[str, Dict[str, Any]]:
        """Find all shortest paths between two entities."""
        query = f"""
        MATCH path = allShortestPaths(
            (a {{id: $source_id}})-[*..{max_hops}]-(b {{id: $target_id}})
        )
        RETURN path,
               length(path) AS hop_count,
               [n IN nodes(path) | n.id] AS node_ids,
               [r IN relationships(path) | type(r)] AS rel_types
        """
        return query, {"source_id": source_id, "target_id": target_id}

    @staticmethod
    def multi_hop_reasoning_chain(
        start_ids: List[str],
        max_hops: int = 3,
        relationship_types: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """BFS-style reasoning chain extraction.

        Returns ordered paths from start nodes, suitable for splitting into
        reasoning chain subgraphs that the retrieval agent can interpret.
        """
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)

        query = f"""
        MATCH path = (start)-[r{rel_filter}*1..{max_hops}]-(end)
        WHERE start.id IN $start_ids
        WITH path,
             length(path) AS depth,
             [n IN nodes(path) | {{
                 id: n.id,
                 name: n.name,
                 labels: labels(n)
             }}] AS chain_nodes,
             [rel IN relationships(path) | {{
                 type: type(rel),
                 properties: properties(rel)
             }}] AS chain_rels
        ORDER BY depth ASC
        RETURN chain_nodes, chain_rels, depth
        LIMIT 50
        """
        return query, {"start_ids": start_ids}

    @staticmethod
    def expand_neighbourhood(
        node_id: str,
        hops: int = 1,
        limit: int = 30,
    ) -> Tuple[str, Dict[str, Any]]:
        """Expand the 1-hop (or n-hop) neighbourhood of a node."""
        query = f"""
        MATCH (center {{id: $node_id}})-[r*1..{hops}]-(neighbour)
        WITH DISTINCT neighbour, r
        RETURN {{
            id: neighbour.id,
            name: neighbour.name,
            labels: labels(neighbour),
            properties: properties(neighbour)
        }} AS node
        LIMIT {limit}
        """
        return query, {"node_id": node_id}

    @staticmethod
    def create_entity(
        entity_id: str,
        label: str,
        properties: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Create or merge an entity node."""
        query = f"""
        MERGE (n:{label} {{id: $entity_id}})
        SET n += $props
        RETURN n
        """
        return query, {"entity_id": entity_id, "props": properties}

    @staticmethod
    def create_relationship(
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create a typed relationship between two nodes."""
        props_clause = "SET r += $props" if properties else ""
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {props_clause}
        RETURN r
        """
        params: Dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if properties:
            params["props"] = properties
        return query, params
