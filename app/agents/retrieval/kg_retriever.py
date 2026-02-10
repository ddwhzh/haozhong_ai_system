"""KG Retriever — Neo4j knowledge graph search for depth expansion.

KG is a pre-existing knowledge base with entities and relationships.
This retriever searches the KG directly (NOT entity-extraction from query),
treating it as a parallel retrieval source alongside RAG/vector search.

Pipeline per sub-query:
1. Full-text search in KG to find matching nodes (like RAG searches vectors).
2. BFS expansion from matched nodes to gather relational context.
3. Split expanded subgraph into readable reasoning chains.
4. Wrap each chain as EvidenceNode with belief + intent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType
from app.infra.neo4j import neo4j_client


class KGRetriever:
    """Knowledge graph retriever — searches existing KG data.

    Unlike vector search which operates on document embeddings,
    KG search leverages the graph structure: nodes, relationships,
    and multi-hop paths provide structured, relational evidence.
    """

    def __init__(
        self,
        max_hops: int = 2,
        limit_per_hop: int = 15,
        search_limit: int = 10,
    ) -> None:
        self.max_hops = max_hops
        self.limit_per_hop = limit_per_hop
        self.search_limit = search_limit

    async def retrieve(
        self,
        sub_query_nodes: List[EvidenceNode],
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """For each sub-query, search KG and expand with BFS.

        This is parallel to vector retrieval:
        - Vector retriever: query -> embedding -> similarity search -> document chunks
        - KG retriever: query -> full-text search -> BFS expansion -> reasoning chains

        Both produce EvidenceNodes that feed into Cascade Fusion.
        """
        all_kg_nodes: List[EvidenceNode] = []

        # Accept both classic "sub_query" and QDMR "tree_step" content types
        _retrieval_types = ("sub_query", "tree_step")
        for sq_node in sub_query_nodes:
            if sq_node.evidence.content_type not in _retrieval_types:
                continue

            query_text = sq_node.evidence.content
            try:
                kg_nodes = await self._search_and_expand(
                    query_text, sq_node, evidence_graph
                )
                all_kg_nodes.extend(kg_nodes)
            except Exception as e:
                logger.error(
                    "kg_retrieval_failed",
                    query=query_text,
                    error=str(e),
                )

        logger.info("kg_retrieval_complete", total_chains=len(all_kg_nodes))
        return all_kg_nodes

    async def _search_and_expand(
        self,
        query: str,
        parent_node: EvidenceNode,
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """Search KG for matching nodes, then BFS expand for context.

        Step 1: Full-text search in KG (name/description CONTAINS query)
        Step 2: BFS from matched nodes to gather relational context
        Step 3: Build reasoning chains from the expanded subgraph
        """
        # Step 1: Search KG for relevant nodes (parallel to vector search)
        matched_nodes = await neo4j_client.find_entities(
            text=query, limit=self.search_limit
        )

        if not matched_nodes:
            logger.debug("kg_no_matches", query=query)
            return []

        # Create evidence nodes for direct KG matches (like document chunks from RAG)
        result_nodes: List[EvidenceNode] = []

        for i, match in enumerate(matched_nodes):
            node_name = match.get("name", match.get("id", "unknown"))
            node_props = match.get("properties", {})
            node_labels = match.get("labels", [])

            # Build a readable description from the KG node
            content_parts = [f"[{':'.join(node_labels)}] {node_name}"]
            if node_props.get("description"):
                content_parts.append(node_props["description"])
            for k, v in node_props.items():
                if k not in ("id", "name", "description") and v:
                    content_parts.append(f"  {k}: {v}")

            content = "\n".join(content_parts)

            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=content,
                    content_type="kg_match",
                    source="knowledge_graph",
                    metadata={
                        "kg_node_id": match.get("id", ""),
                        "kg_labels": node_labels,
                        "rank": i,
                    },
                ),
                belief=Belief(
                    content=f"KG direct match: {node_name}",
                    confidence=max(0.6, 0.9 - i * 0.05),  # Decay by rank
                    source_agent="retrieval_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.KG_EXPLORE,
                    description=f"KG full-text search match (rank={i})",
                    parameters={"kg_node_id": match.get("id", ""), "rank": i},
                    source_agent="retrieval_agent",
                ),
                parent_node_ids=[parent_node.node_id],
                relation=EdgeRelation.DERIVED_FROM,
                tags=["kg_match", f"rank_{i}"],
            )
            result_nodes.append(node)

        # Step 2: BFS expansion for relational context
        matched_ids = [m.get("id") for m in matched_nodes if m.get("id")]
        if matched_ids:
            expansion_nodes = await self._bfs_expand(
                seed_ids=matched_ids,
                parent_node=parent_node,
                evidence_graph=evidence_graph,
            )
            result_nodes.extend(expansion_nodes)

        return result_nodes

    async def _bfs_expand(
        self,
        seed_ids: List[str],
        parent_node: EvidenceNode,
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """BFS expand from seed nodes to gather relational reasoning chains."""
        subgraph = await neo4j_client.bfs_traverse(
            start_node_ids=seed_ids,
            max_hops=self.max_hops,
            limit_per_hop=self.limit_per_hop,
        )

        if not subgraph.get("nodes"):
            return []

        # Build reasoning chains from the BFS subgraph
        chains = self._build_reasoning_chains(subgraph, seed_ids)

        result_nodes: List[EvidenceNode] = []
        for i, chain in enumerate(chains):
            chain_text = self._chain_to_text(chain)
            confidence = self._compute_chain_confidence(chain)

            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=chain_text,
                    content_type="reasoning_chain",
                    source="knowledge_graph",
                    metadata={
                        "chain_index": i,
                        "chain_length": len(chain.get("nodes", [])),
                        "seed_entities": seed_ids[:3],
                        "depth": chain.get("depth", 0),
                    },
                ),
                belief=Belief(
                    content=f"KG reasoning chain (depth={chain.get('depth', 0)})",
                    confidence=confidence,
                    source_agent="retrieval_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.KG_EXPLORE,
                    description=f"BFS reasoning chain, depth={chain.get('depth', 0)}",
                    parameters={"chain_index": i, "max_hops": self.max_hops},
                    source_agent="retrieval_agent",
                ),
                parent_node_ids=[parent_node.node_id],
                relation=EdgeRelation.DERIVED_FROM,
                tags=["kg_chain", f"depth_{chain.get('depth', 0)}"],
            )
            result_nodes.append(node)

        return result_nodes

    def _build_reasoning_chains(
        self,
        subgraph: Dict[str, Any],
        seed_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Build reasoning chains from BFS subgraph."""
        nodes = subgraph.get("nodes", [])
        if not nodes:
            return []

        chains = []
        for seed_id in seed_ids:
            chain_nodes = [n for n in nodes if n.get("id") == seed_id]
            remaining = [n for n in nodes if n.get("id") != seed_id]
            connected = chain_nodes + remaining[:self.limit_per_hop]
            if connected:
                chains.append({
                    "nodes": connected,
                    "seed": seed_id,
                    "depth": min(self.max_hops, len(connected)),
                })

        if not chains and nodes:
            chains.append({
                "nodes": nodes,
                "seed": seed_ids[0] if seed_ids else "unknown",
                "depth": self.max_hops,
            })

        return chains

    @staticmethod
    def _chain_to_text(chain: Dict[str, Any]) -> str:
        """Convert a reasoning chain to human-readable text."""
        parts = []
        for node in chain.get("nodes", []):
            name = node.get("name", node.get("id", "?"))
            labels = node.get("labels", [])
            label_str = ":".join(labels) if labels else ""
            props = node.get("properties", {})
            desc = props.get("description", "")
            part = f"[{label_str}] {name}"
            if desc:
                part += f" ({desc[:80]})"
            parts.append(part)
        return " -> ".join(parts) if parts else "(empty chain)"

    @staticmethod
    def _compute_chain_confidence(chain: Dict[str, Any]) -> float:
        """Heuristic confidence: shorter chains are more reliable."""
        depth = chain.get("depth", 1)
        return round(0.95 ** max(depth, 1), 3)
