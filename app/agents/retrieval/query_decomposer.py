"""Query Decomposer — RAGFlow-style + QDMR query decomposition.

Supports two modes:
1. Classic mode: flat sub-query decomposition for breadth expansion.
2. QDMR mode: type-aware decomposition into ordered steps with
   operator/strategy hints, returning TreeStep objects for tree exploration.

Design: inspired by RAGFlow + QDMR (Question Decomposition Meaning Representation).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode, EdgeRelation
from app.core.types.intent import Intent, IntentType
from app.infra.llm import llm_service
from app.prompts import load_prompt

if TYPE_CHECKING:
    from app.agents.retrieval.query_classifier import QType
    from app.agents.retrieval.tree_explorer import TreeStep


class QueryDecomposer:
    """Decomposes a complex query into sub-queries or QDMR tree steps."""

    def __init__(self, max_sub_queries: int = 5) -> None:
        self.max_sub_queries = max_sub_queries

    async def decompose(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """Decompose query into sub-queries, each wrapped as an EvidenceNode.

        Args:
            query: The original user query.
            evidence_graph: Current graph (for context if needed).

        Returns:
            List of EvidenceNodes representing sub-queries, plus the original
            query node as root.
        """
        # Create root query node
        root_node = EvidenceNode(
            evidence=Evidence(
                content=query,
                content_type="query",
                source="user",
            ),
            belief=Belief(
                content=f"Original user query: {query}",
                confidence=1.0,
                source_agent="retrieval_agent",
            ),
            intent=Intent(
                intent_type=IntentType.RETRIEVE,
                description="Original user query to be decomposed",
                source_agent="retrieval_agent",
            ),
            tags=["query", "root"],
        )
        evidence_graph.add_node(root_node)

        # LLM-based decomposition
        try:
            sub_queries = await self._llm_decompose(query)
        except Exception as e:
            logger.error("query_decomposition_failed", error=str(e))
            # Fallback: use original query as sole sub-query
            sub_queries = [{"query": query, "aspect": "original"}]

        # Wrap each sub-query as an EvidenceNode
        sub_query_nodes: List[EvidenceNode] = []
        for i, sq in enumerate(sub_queries):
            sq_text = sq.get("query", query)
            aspect = sq.get("aspect", f"aspect_{i}")

            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=sq_text,
                    content_type="sub_query",
                    source="query_decomposer",
                    metadata={"aspect": aspect, "index": i},
                ),
                belief=Belief(
                    content=f"Sub-query for aspect '{aspect}': {sq_text}",
                    confidence=0.9,
                    source_agent="retrieval_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.DECOMPOSE_QUERY,
                    description=f"Decomposed sub-query targeting '{aspect}' aspect",
                    parameters={"aspect": aspect, "parent_query": query},
                    source_agent="retrieval_agent",
                ),
                parent_node_ids=[root_node.node_id],
                relation=EdgeRelation.DECOMPOSES_TO,
                tags=["sub_query", aspect],
            )
            sub_query_nodes.append(node)

        logger.info(
            "query_decomposed",
            original=query,
            sub_query_count=len(sub_query_nodes),
        )
        return [root_node] + sub_query_nodes

    async def _llm_decompose(self, query: str) -> List[Dict[str, Any]]:
        """Use LLM to decompose the query."""
        messages = [
            SystemMessage(content=load_prompt(
                "query_decompose_system",
                max_sub_queries=self.max_sub_queries,
            )),
            HumanMessage(content=f"Decompose this query:\n{query}"),
        ]

        response = await llm_service.call(messages, prompt_name="query_decompose_system")
        content = response.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed[:self.max_sub_queries]
        return [{"query": query, "aspect": "original"}]

    # ── QDMR-aware decomposition ──────────────────────────────────────────

    async def qdmr_decompose(
        self,
        query: str,
        q_type: "QType",
        max_steps: int = 5,
    ) -> List["TreeStep"]:
        """Decompose query into QDMR-ordered tree steps based on query type.

        Args:
            query: Original user query.
            q_type: Classified query type (QSelect, QChain, etc.).
            max_steps: Maximum number of decomposition steps.

        Returns:
            List of TreeStep objects for the TreeExplorer.
        """
        from app.agents.retrieval.tree_explorer import TreeStep

        try:
            raw_steps = await self._llm_qdmr_decompose(query, q_type, max_steps)
        except Exception as e:
            logger.error("qdmr_decomposition_failed", error=str(e))
            raw_steps = [
                {
                    "step": 1,
                    "question": query,
                    "operator": "select",
                    "retrieval_strategy": "both",
                    "depends_on": [],
                }
            ]

        steps = []
        for raw in raw_steps:
            steps.append(TreeStep(
                step_idx=int(raw.get("step", len(steps) + 1)),
                question=raw.get("question", query),
                operator=raw.get("operator", "select"),
                retrieval_strategy=raw.get("retrieval_strategy", "both"),
                depends_on=raw.get("depends_on", []),
            ))

        logger.info(
            "qdmr_decomposed",
            q_type=q_type.value,
            step_count=len(steps),
        )
        return steps

    async def _llm_qdmr_decompose(
        self,
        query: str,
        q_type: "QType",
        max_steps: int,
    ) -> List[Dict[str, Any]]:
        """Use LLM for QDMR-aware decomposition."""
        messages = [
            SystemMessage(content=load_prompt(
                "qdmr_decompose_system",
                q_type=q_type.value,
                max_steps=max_steps,
            )),
            HumanMessage(content=f"Decompose this query (type: {q_type.value}):\n{query}"),
        ]

        response = await llm_service.call(
            messages, prompt_name="qdmr_decompose_system"
        )
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed[:max_steps]
        if isinstance(parsed, dict):
            return [parsed]
        return [{"step": 1, "question": query, "operator": "select",
                 "retrieval_strategy": "both", "depends_on": []}]
