"""Retrieval Agent — multi-hop retrieval with belief+intent reasoning graph.

Pipeline: Query Decomposition (breadth) -> KG BFS (depth) -> Reasoning Chain
           Subgraph Split -> Cascade Multi-Round Fusion
"""

from app.agents.retrieval.agent import RetrievalAgent

__all__ = ["RetrievalAgent"]
