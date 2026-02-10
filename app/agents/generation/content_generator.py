"""Content Generator - evidence graph slot extraction for each proposal.

Stage-2 of the Fast RCNN analogy: given a proposal with slot definitions,
EXTRACT information from the evidence graph to fill each slot.

Key difference from template-based approach:
- Old: LLM fills a text template freely.
- New: LLM extracts evidence from the graph to fill defined slots.
  Each slot maps to specific evidence nodes. The output is structured
  slot-filled JSON, not free-form text.

This ensures:
- Consistent granularity: all proposals same structure = same slot types
- Extraction > Generation: slots are filled from evidence, not invented
- Traceability: each filled slot links back to source evidence nodes
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


class ContentGenerator:
    """Stage-2: Fill proposal slots by extracting from evidence graph."""

    async def generate(
        self,
        proposal_nodes: List[EvidenceNode],
        evidence_graph: EvidenceGraph,
        query: str,
    ) -> List[EvidenceNode]:
        """For each proposal, extract evidence to fill its slots.

        Returns:
            List of generated content EvidenceNodes (one per proposal).
        """
        generated_nodes: List[EvidenceNode] = []

        sorted_proposals = sorted(
            proposal_nodes,
            key=lambda n: n.evidence.metadata.get("priority", 3),
        )

        for prop_node in sorted_proposals:
            try:
                gen_node = await self._extract_for_proposal(
                    prop_node, evidence_graph, query
                )
                if gen_node:
                    generated_nodes.append(gen_node)
            except Exception as e:
                logger.error(
                    "slot_extraction_failed",
                    proposal_aspect=prop_node.evidence.metadata.get("aspect"),
                    error=str(e),
                )

        logger.info("slot_extraction_complete", count=len(generated_nodes))
        return generated_nodes

    async def _extract_for_proposal(
        self,
        proposal_node: EvidenceNode,
        evidence_graph: EvidenceGraph,
        query: str,
    ) -> EvidenceNode | None:
        """Extract evidence to fill slots for a single proposal."""
        try:
            proposal = json.loads(proposal_node.evidence.content)
        except json.JSONDecodeError:
            proposal = {
                "aspect": "general",
                "question": proposal_node.evidence.content,
                "slots": [{"name": "content", "description": "Main content", "evidence_hint": "any"}],
            }

        aspect = proposal.get("aspect", "general")
        question = proposal.get("question", "")
        slots = proposal.get("slots", [])

        # Gather evidence relevant to the slot hints
        evidence_by_hint = await self._gather_evidence_by_hint(
            evidence_graph=evidence_graph,
            slots=slots,
            query=query,
        )
        evidence_text = self._format_evidence_for_extraction(evidence_by_hint)

        # Slots description
        slots_desc = json.dumps(slots, ensure_ascii=False, indent=2)
        output_language = "中文" if self._is_chinese_query(query) else "English"

        messages = [
            SystemMessage(content=load_prompt("slot_extraction_system")),
            HumanMessage(content=(
                "Original Query: %s\n\n"
                "Output Language: %s\n"
                "Answer only with that language.\n\n"
                "Proposal Aspect: %s\n"
                "Proposal Question: %s\n\n"
                "Slots to fill:\n%s\n\n"
                "Available Evidence:\n%s"
            ) % (query, output_language, aspect, question, slots_desc, evidence_text)),
        ]

        response = await llm_service.call(messages, prompt_name="slot_extraction_system")
        raw = response.content.strip()

        # Parse filled slots
        filled_slots = self._parse_filled_slots(raw)

        # Build readable content from filled slots
        readable_content = self._slots_to_readable(
            aspect=aspect,
            question=question,
            filled_slots=filled_slots,
            query=query,
        )

        # Calculate overall confidence from slot confidences
        confidences = [
            v.get("confidence", 0.5) for v in filled_slots.values()
            if isinstance(v, dict) and v.get("value") is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.3

        node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=readable_content,
                content_type="generated_content",
                source="content_generator",
                metadata={
                    "aspect": aspect,
                    "question": question,
                    "filled_slots": filled_slots,
                    "slot_count": len(slots),
                    "filled_count": sum(
                        1 for v in filled_slots.values()
                        if isinstance(v, dict) and v.get("value") is not None
                    ),
                    "evidence_count": sum(len(v) for v in evidence_by_hint.values()),
                },
            ),
            belief=Belief(
                content="Generated content for [%s]: %s" % (aspect, question),
                confidence=min(avg_confidence, 1.0),
                source_agent="generation_agent",
            ),
            intent=Intent(
                intent_type=IntentType.GENERATE,
                description="Slot-filled content for: %s" % aspect,
                parameters={"aspect": aspect},
                source_agent="generation_agent",
            ),
            parent_node_ids=[proposal_node.node_id],
            relation=EdgeRelation.DERIVED_FROM,
            tags=["generated", aspect, "slot_filled"],
        )
        return node

    @staticmethod
    async def _gather_evidence_by_hint(
        evidence_graph: EvidenceGraph,
        slots: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, List[EvidenceNode]]:
        """Gather evidence nodes organized by slot evidence_hint."""
        all_nodes = list(evidence_graph.nodes.values())
        result: Dict[str, List[EvidenceNode]] = {}
        query_keywords = await query_keyword_extractor.extract_keywords(
            query=query,
            use_llm=settings.KEYWORD_EXTRACTION_USE_LLM,
            max_keywords=settings.KEYWORD_EXTRACTION_MAX_TERMS,
        )

        # Only retrieval evidence is allowed for slot filling.
        # Avoid using internal orchestration/proposal/classification nodes.
        hint_to_types = {
            # For strict grounding, avoid using fused_evidence directly because
            # it is an LLM-produced summary and can echo query wording.
            # Use raw retrieval evidence instead.
            "fused_evidence": {"document_chunk", "web_result", "kg_match", "reasoning_chain"},
            "web_result": {"web_result"},
            "kg_match": {"kg_match", "reasoning_chain"},
            "document_chunk": {"document_chunk"},
        }

        for slot in slots:
            hint = slot.get("evidence_hint", "any")
            if hint not in result:
                matching_types = hint_to_types.get(hint, set())
                if matching_types:
                    candidates = [
                        n for n in all_nodes
                        if n.evidence.content_type in matching_types
                        and n.belief.confidence >= 0.25
                    ]
                else:
                    candidates = [
                        n for n in all_nodes
                        if n.evidence.content_type in (
                            "fused_evidence",
                            "document_chunk",
                            "web_result",
                            "kg_match",
                            "reasoning_chain",
                        )
                        and n.belief.confidence >= 0.25
                    ]

                # Query-keyword relevance gate: avoid off-topic evidence.
                if query_keywords:
                    filtered_candidates = [
                        n for n in candidates
                        if query_keyword_extractor.match_keywords(query_keywords, n.evidence.content)
                    ]
                    if filtered_candidates:
                        candidates = filtered_candidates
                    elif candidates:
                        # Fallback when keyword extraction is too strict.
                        logger.warning(
                            "slot_extraction_keyword_filter_fallback",
                            query=query[:120],
                            hint=hint,
                            keywords=query_keywords[:6],
                            candidates_before=len(candidates),
                        )

                result[hint] = candidates

        return result

    @staticmethod
    def _format_evidence_for_extraction(
        evidence_by_hint: Dict[str, List[EvidenceNode]],
    ) -> str:
        parts = []
        for hint, nodes in evidence_by_hint.items():
            parts.append("=== Evidence type: %s ===" % hint)
            for n in nodes[:8]:
                parts.append(
                    "[id=%s type=%s conf=%.2f] %s"
                    % (
                        n.node_id,
                        n.evidence.content_type,
                        n.belief.confidence,
                        n.evidence.content[:300],
                    )
                )
            if not nodes:
                parts.append("(no evidence of this type)")
        return "\n".join(parts)

    @staticmethod
    def _parse_filled_slots(raw: str) -> Dict[str, Any]:
        """Parse LLM output into filled slots dict."""
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_output": {"value": raw, "confidence": 0.5, "source_evidence_ids": []}}

    @staticmethod
    def _slots_to_readable(
        aspect: str,
        question: str,
        filled_slots: Dict[str, Any],
        query: str,
    ) -> str:
        """Convert filled slots to readable text with query-language alignment."""
        is_zh = ContentGenerator._is_chinese_query(query)
        aspect_label = ContentGenerator._localize_label(aspect, is_zh)
        q_label = "问题" if is_zh else "Q"
        conf_label = "置信度" if is_zh else "conf"
        source_label = "证据ID" if is_zh else "source_evidence_ids"
        insufficient_text = "证据不足" if is_zh else "insufficient evidence"

        lines = ["## %s" % aspect_label, "", "**%s:** %s" % (q_label, question), ""]
        all_insufficient = True

        for slot_name, slot_data in filled_slots.items():
            # Some models may return list[dict] for a slot; normalize to best item.
            if isinstance(slot_data, list):
                dict_items = [x for x in slot_data if isinstance(x, dict)]
                if dict_items:
                    slot_data = max(
                        dict_items, key=lambda x: float(x.get("confidence", 0.0))
                    )
                else:
                    slot_data = {"value": None, "confidence": 0.0, "source_evidence_ids": []}

            slot_label = ContentGenerator._localize_label(slot_name, is_zh)
            if isinstance(slot_data, dict):
                value = slot_data.get("value")
                conf = slot_data.get("confidence", 0)
                source_ids = slot_data.get("source_evidence_ids", [])
                has_value = value not in (None, "", []) and bool(source_ids)
                if has_value:
                    all_insufficient = False
                    lines.append("**%s** (%s=%.2f):" % (slot_label, conf_label, conf))
                    lines.append(str(value).strip())
                    lines.append("(%s: %s)" % (source_label, ", ".join(source_ids)))
                    lines.append("")
                else:
                    lines.append("**%s**: [%s]" % (slot_label, insufficient_text))
                    lines.append("")
            else:
                lines.append("**%s**: %s" % (slot_label, str(slot_data)))
                lines.append("")

        if all_insufficient:
            if is_zh:
                lines.append(
                    "结论: 当前检索证据不足以直接回答该子问题, 请补充该领域资料后重试。"
                )
            else:
                lines.append(
                    "Conclusion: Current retrieval evidence is insufficient to answer this sub-question."
                )

        return "\n".join(lines)

    @staticmethod
    def _is_chinese_query(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _localize_label(label: str, prefer_zh: bool) -> str:
        if not label:
            return ""
        raw = str(label).strip()
        if not prefer_zh:
            return raw

        # Common slot/aspect aliases seen in this project.
        zh_map = {
            "definition": "定义",
            "components": "组成",
            "data": "数据",
            "relationship": "关系",
            "applications": "应用",
            "comprehensive": "综合回答",
            "core_answer": "核心回答",
            "key_points": "关键要点",
            "context": "背景信息",
            "answer": "答案",
            "details": "细节",
            "reasoning": "推理依据",
            "conclusion": "结论",
        }
        if raw in zh_map:
            return zh_map[raw]
        if re.search(r"[\u4e00-\u9fff]", raw):
            return raw
        return raw.replace("_", " ")
