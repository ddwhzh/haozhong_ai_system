"""Proposal Decomposer - Fast RCNN-inspired evidence graph slot filling.

Stage 1 (RPN analogue): generate N same-structured "proposals".
Each proposal identifies an aspect of the answer and defines slots
to be filled by extracting from the evidence graph.

Key design: Proposals are NOT templates. They are structured slot
definitions pointing to evidence graph nodes for extraction.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import logger
from app.core.nlp import query_keyword_extractor
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


class ProposalDecomposer:
    """Stage-1: Decompose query into N same-structured slot-filling proposals."""

    def __init__(self, max_proposals: int = 5) -> None:
        self.max_proposals = max_proposals

    async def generate_proposals(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
    ) -> List[EvidenceNode]:
        """Produce slot-filling proposals from query + evidence."""
        prefer_zh = self._is_chinese_query(query)
        query_keywords = await query_keyword_extractor.extract_keywords(
            query=query,
            use_llm=settings.KEYWORD_EXTRACTION_USE_LLM,
            max_keywords=settings.KEYWORD_EXTRACTION_MAX_TERMS,
        )

        retrieval_candidates = [
            n
            for n in evidence_graph.get_high_confidence_nodes(threshold=0.3)
            if n.evidence.content_type in (
                "document_chunk",
                "web_result",
                "kg_match",
                "reasoning_chain",
            )
        ]

        if query_keywords:
            filtered_nodes = [
                n for n in retrieval_candidates
                if query_keyword_extractor.match_keywords(query_keywords, n.evidence.content)
            ]
            if filtered_nodes:
                high_conf_nodes = filtered_nodes
            else:
                high_conf_nodes = retrieval_candidates
                if retrieval_candidates:
                    logger.warning(
                        "proposal_keyword_filter_fallback",
                        query=query[:120],
                        keywords=query_keywords[:6],
                        candidates_before=len(retrieval_candidates),
                    )
        else:
            high_conf_nodes = retrieval_candidates

        evidence_summary = self._summarize_evidence(high_conf_nodes)
        evidence_stats = self._count_evidence_types(evidence_graph)

        if not high_conf_nodes:
            logger.warning(
                "proposal_generation_no_relevant_evidence",
                query=query,
                keywords=query_keywords,
            )
            proposals = [self._fallback_proposal(query, prefer_zh)]
        else:
            try:
                proposals = await self._llm_propose(
                    query=query,
                    evidence_summary=evidence_summary,
                    evidence_stats=evidence_stats,
                    prefer_zh=prefer_zh,
                )
            except Exception as e:
                logger.error("proposal_generation_failed", error=str(e))
                proposals = [self._fallback_proposal(query, prefer_zh)]

        proposal_nodes: List[EvidenceNode] = []
        for i, prop in enumerate(proposals):
            aspect = prop.get("aspect", "general")
            question = prop.get("question", "")
            slots = prop.get("slots", [])

            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=json.dumps(prop, ensure_ascii=False),
                    content_type="proposal",
                    source="proposal_decomposer",
                    metadata={
                        "aspect": aspect,
                        "question": question,
                        "slot_count": len(slots),
                        "priority": prop.get("priority", 3),
                        "index": i,
                    },
                ),
                belief=Belief(
                    content="Proposal [%s]: %s" % (aspect, question),
                    confidence=0.85,
                    source_agent="generation_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.PROPOSE_PLAN,
                    description="Slot-filling proposal: %s" % aspect,
                    parameters={
                        "aspect": aspect,
                        "slot_count": len(slots),
                        "priority": prop.get("priority", 3),
                    },
                    source_agent="generation_agent",
                ),
                parent_node_ids=[],
                tags=["proposal", aspect],
            )
            proposal_nodes.append(node)

        logger.info(
            "proposals_generated",
            count=len(proposal_nodes),
            aspects=[p.evidence.metadata.get("aspect") for p in proposal_nodes],
        )
        return proposal_nodes

    @staticmethod
    def _summarize_evidence(nodes: List[EvidenceNode]) -> str:
        if not nodes:
            return "(no evidence available)"
        parts = []
        for node in nodes[:15]:
            conf = node.belief.confidence
            ctype = node.evidence.content_type
            content = node.evidence.content[:150]
            parts.append("[%s conf=%.2f] %s" % (ctype, conf, content))
        return "\n---\n".join(parts)

    @staticmethod
    def _count_evidence_types(evidence_graph: EvidenceGraph) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in evidence_graph.nodes.values():
            ct = n.evidence.content_type
            counts[ct] = counts.get(ct, 0) + 1
        return counts

    async def _llm_propose(
        self,
        query: str,
        evidence_summary: str,
        evidence_stats: Dict[str, int],
        prefer_zh: bool,
    ) -> List[Dict[str, Any]]:
        stats_text = ", ".join("%s=%s" % (k, v) for k, v in evidence_stats.items())
        output_language = "中文" if prefer_zh else "English"
        messages = [
            SystemMessage(content=load_prompt(
                "proposal_decompose_system",
                max_proposals=self.max_proposals,
            )),
            HumanMessage(content=(
                "User Query: %s\n"
                "Output Language: %s\n\n"
                "Evidence stats: %s\n\n"
                "Evidence:\n%s"
                % (query, output_language, stats_text, evidence_summary)
            )),
        ]
        response = await llm_service.call(messages, prompt_name="proposal_decompose_system")
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed[:self.max_proposals]
        return [self._fallback_proposal(query, prefer_zh)]

    @staticmethod
    def _fallback_proposal(query: str, prefer_zh: bool = False) -> Dict[str, Any]:
        if prefer_zh:
            return {
                "aspect": "综合回答",
                "question": query,
                "slots": [
                    {"name": "核心回答", "description": "直接回答问题", "evidence_hint": "fused_evidence"},
                    {"name": "关键依据", "description": "支撑回答的关键证据点", "evidence_hint": "fused_evidence"},
                    {"name": "背景信息", "description": "必要背景信息", "evidence_hint": "document_chunk"},
                ],
                "priority": 1,
            }
        return {
            "aspect": "comprehensive",
            "question": query,
            "slots": [
                {"name": "core_answer", "description": "Direct answer", "evidence_hint": "fused_evidence"},
                {"name": "key_points", "description": "2-3 key points", "evidence_hint": "fused_evidence"},
                {"name": "context", "description": "Background info", "evidence_hint": "document_chunk"},
            ],
            "priority": 1,
        }

    @staticmethod
    def _is_chinese_query(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

