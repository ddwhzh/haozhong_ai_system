"""Evidence Graph — the shared data structure for inter-agent communication.

Every agent reads from and writes to an *evidence graph*.  Nodes carry content +
belief + intent; edges encode provenance ("this node was derived from that node").

The graph is *append-only* within a single pipeline run — agents never delete nodes,
they can only add new ones or propose belief updates.

Design notes
------------
- `EvidenceNode`: atomic unit — one retrieval chunk, one generation paragraph, one eval score.
- `EvidenceEdge`: directed provenance link with a typed relation.
- `EvidenceGraph`: container with O(1) node lookup and convenience traversal helpers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.types.belief import Belief
from app.core.types.intent import Intent


# ── Edge relation taxonomy ───────────────────────────────────────────────

class EdgeRelation(str, Enum):
    """Typed relations between evidence nodes."""

    DERIVED_FROM = "derived_from"       # B was produced using A
    SUPPORTS = "supports"               # A supports the claim in B
    CONTRADICTS = "contradicts"         # A contradicts the claim in B
    DECOMPOSES_TO = "decomposes_to"     # A (query) was decomposed into B (sub-query)
    FUSED_INTO = "fused_into"           # A, … were fused into B (cascade)
    EVALUATED_BY = "evaluated_by"       # A was evaluated, result is B
    REFINED_INTO = "refined_into"       # A was refined into B


# ── Evidence primitives ──────────────────────────────────────────────────

class Evidence(BaseModel):
    """Raw evidence payload — the content carried by a node."""

    content: str = Field(..., description="Textual content of the evidence")
    content_type: str = Field(
        default="text",
        description="MIME-like content type: text, json, embedding, score, etc.",
    )
    source: str = Field(default="", description="Origin: document ID, KG node, URL, …")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceNode(BaseModel):
    """A single node in the evidence graph.

    Invariant: every node has exactly one Belief and one Intent at creation time.
    """

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evidence: Evidence = Field(..., description="The actual evidence payload")
    belief: Belief = Field(..., description="Agent's belief about this evidence")
    intent: Intent = Field(..., description="Why this evidence was produced")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list, description="Free-form tags for filtering")


class EvidenceEdge(BaseModel):
    """Directed edge between two evidence nodes."""

    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = Field(..., description="ID of the source EvidenceNode")
    target_node_id: str = Field(..., description="ID of the target EvidenceNode")
    relation: EdgeRelation = Field(..., description="Typed relation")
    weight: float = Field(default=1.0, description="Optional edge weight / strength")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Evidence Graph container ─────────────────────────────────────────────

class EvidenceGraph(BaseModel):
    """Append-only graph of evidence nodes and provenance edges.

    This is the *shared blackboard* that all agents read from and write to.
    """

    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nodes: Dict[str, EvidenceNode] = Field(
        default_factory=dict,
        description="node_id -> EvidenceNode lookup",
    )
    edges: List[EvidenceEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Mutation helpers (append-only) ───────────────────────────────────

    def add_node(self, node: EvidenceNode) -> EvidenceNode:
        """Add a node to the graph. Returns the added node."""
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: EvidenceEdge) -> EvidenceEdge:
        """Add an edge. Both endpoint nodes must already exist."""
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_node_id} not in graph")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_node_id} not in graph")
        self.edges.append(edge)
        return edge

    def add_evidence(
        self,
        evidence: Evidence,
        belief: Belief,
        intent: Intent,
        parent_node_ids: Optional[List[str]] = None,
        relation: EdgeRelation = EdgeRelation.DERIVED_FROM,
        tags: Optional[List[str]] = None,
    ) -> EvidenceNode:
        """Convenience: create node + edges from parents in one call."""
        node = EvidenceNode(
            evidence=evidence,
            belief=belief,
            intent=intent,
            tags=tags or [],
        )
        self.add_node(node)

        for pid in parent_node_ids or []:
            edge = EvidenceEdge(
                source_node_id=pid,
                target_node_id=node.node_id,
                relation=relation,
            )
            self.add_edge(edge)

        return node

    # ── Query helpers ────────────────────────────────────────────────────

    def get_nodes_by_agent(self, agent_name: str) -> List[EvidenceNode]:
        """Return all nodes produced by a specific agent."""
        return [
            n for n in self.nodes.values()
            if n.belief.source_agent == agent_name
        ]

    def get_nodes_by_intent(self, intent_type: str) -> List[EvidenceNode]:
        """Return all nodes with a given intent type."""
        return [
            n for n in self.nodes.values()
            if n.intent.intent_type == intent_type
        ]

    def get_children(self, node_id: str) -> List[EvidenceNode]:
        """Return direct child nodes (targets of edges from node_id)."""
        child_ids = [e.target_node_id for e in self.edges if e.source_node_id == node_id]
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, node_id: str) -> List[EvidenceNode]:
        """Return direct parent nodes (sources of edges into node_id)."""
        parent_ids = [e.source_node_id for e in self.edges if e.target_node_id == node_id]
        return [self.nodes[pid] for pid in parent_ids if pid in self.nodes]

    def get_leaf_nodes(self) -> List[EvidenceNode]:
        """Return nodes that have no outgoing edges (terminal evidence)."""
        sources = {e.source_node_id for e in self.edges}
        return [n for nid, n in self.nodes.items() if nid not in sources]

    def get_high_confidence_nodes(self, threshold: float = 0.7) -> List[EvidenceNode]:
        """Return nodes whose belief confidence >= threshold."""
        return [
            n for n in self.nodes.values()
            if n.belief.confidence >= threshold
        ]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary dict for logging / debugging."""
        agents = set()
        for n in self.nodes.values():
            agents.add(n.belief.source_agent)
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "agents_involved": sorted(agents),
            "avg_confidence": (
                sum(n.belief.confidence for n in self.nodes.values()) / self.node_count
                if self.node_count
                else 0.0
            ),
        }
