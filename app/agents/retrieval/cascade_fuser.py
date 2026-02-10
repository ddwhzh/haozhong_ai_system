"""Cascade Fuser — multi-round result fusion for multi-hop retrieval.

Implements Cascade mode: iteratively fuse retrieval results from both
the breadth pipeline (vector search) and the depth pipeline (KG BFS),
progressively refining the evidence graph across rounds.

Each round:
1. Collect all current evidence nodes.
2. Score and rank by relevance to the original query.
3. Identify information gaps.
4. If gaps remain and rounds budget allows, trigger another retrieval round.
5. Fuse results into consolidated evidence nodes with updated beliefs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType
from app.infra.llm import llm_service
from app.prompts import load_prompt


class CascadeFuser:
    """Multi-round cascade fusion of retrieval results."""

    def __init__(self, max_rounds: int = 3) -> None:
        self.max_rounds = max_rounds

    async def fuse(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
        breadth_nodes: List[EvidenceNode],
        depth_nodes: List[EvidenceNode],
    ) -> List[EvidenceNode]:
        """Run cascade fusion across multiple rounds.

        Args:
            query: Original user query.
            evidence_graph: Shared evidence graph.
            breadth_nodes: Nodes from vector/RAG retrieval (breadth).
            depth_nodes: Nodes from KG BFS retrieval (depth).

        Returns:
            Fused evidence nodes added to the graph.
        """
        all_source_nodes = breadth_nodes + depth_nodes
        fused_nodes: List[EvidenceNode] = []

        for round_idx in range(self.max_rounds):
            logger.info(
                "cascade_round_start",
                round=round_idx + 1,
                source_node_count=len(all_source_nodes),
            )

            # Collect evidence texts for fusion
            evidence_texts = self._collect_evidence_texts(all_source_nodes)
            if not evidence_texts:
                break

            # LLM-based fusion
            try:
                fusion_result = await self._llm_fuse(query, evidence_texts)
            except Exception as e:
                logger.error("cascade_fusion_llm_failed", round=round_idx, error=str(e))
                break

            # Create fused evidence node
            completeness = fusion_result.get("completeness_score", 0.5)
            relevance = fusion_result.get("relevance_score", 0.5)
            combined_confidence = (completeness + relevance) / 2.0

            parent_ids = [n.node_id for n in all_source_nodes]
            fused_node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=fusion_result.get("fused_summary", ""),
                    content_type="fused_evidence",
                    source="cascade_fuser",
                    metadata={
                        "round": round_idx + 1,
                        "relevance": relevance,
                        "completeness": completeness,
                        "gaps": fusion_result.get("gaps", []),
                        "reasoning": fusion_result.get("reasoning", ""),
                    },
                ),
                belief=Belief(
                    content=f"Cascade round {round_idx + 1} fusion (completeness={completeness:.2f})",
                    confidence=combined_confidence,
                    source_agent="retrieval_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.CASCADE_FUSE,
                    description=f"Cascade fusion round {round_idx + 1}",
                    parameters={
                        "round": round_idx + 1,
                        "source_count": len(all_source_nodes),
                        "gaps": fusion_result.get("gaps", []),
                    },
                    source_agent="retrieval_agent",
                ),
                parent_node_ids=parent_ids,
                relation=EdgeRelation.FUSED_INTO,
                tags=["fused", f"cascade_round_{round_idx + 1}"],
            )
            fused_nodes.append(fused_node)

            # Check if we're complete enough to stop
            if completeness >= 0.85:
                logger.info(
                    "cascade_early_stop",
                    round=round_idx + 1,
                    completeness=completeness,
                )
                break

            # If gaps remain, the next round will include fused nodes
            all_source_nodes = [fused_node] + all_source_nodes

        logger.info(
            "cascade_fusion_complete",
            rounds=len(fused_nodes),
            final_confidence=(
                fused_nodes[-1].belief.confidence if fused_nodes else 0.0
            ),
        )
        return fused_nodes

    @staticmethod
    def _collect_evidence_texts(nodes: List[EvidenceNode]) -> List[str]:
        """Extract text content from evidence nodes."""
        texts = []
        for node in nodes:
            content = node.evidence.content
            if content and content.strip():
                conf = node.belief.confidence
                texts.append(f"[confidence={conf:.2f}] {content}")
        return texts

    async def _llm_fuse(
        self,
        query: str,
        evidence_texts: List[str],
    ) -> Dict[str, Any]:
        """Use LLM to fuse multiple evidence fragments."""
        evidence_block = "\n---\n".join(evidence_texts[:20])  # limit context

        messages = [
            SystemMessage(content=load_prompt("cascade_fusion_system")),
            HumanMessage(
                content=f"Query: {query}\n\nEvidence fragments:\n{evidence_block}"
            ),
        ]

        response = await llm_service.call(messages, prompt_name="cascade_fusion_system")
        content = response.content.strip()

        # Parse JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "fused_summary": content,
                "relevance_score": 0.5,
                "completeness_score": 0.5,
                "gaps": [],
                "reasoning": "Failed to parse structured output",
            }
