"""Auto Prompt Optimizer — automated prompt refinement based on evaluation feedback.

Uses evaluation results (Hungarian matching scores, section-level feedback)
to automatically propose prompt improvements for the generation agent.

Strategy:
1. Identify weak sections: low match scores, unfaithful content, gaps.
2. Diagnose issues: missing context, wrong granularity, hallucination.
3. Generate prompt patches: specific modifications to system prompts / templates.
4. Validate patches: simulate improvement before applying.

This module provides the foundation for continuous self-improvement of the
agent pipeline without manual prompt engineering.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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


class AutoPromptOptimizer:
    """Automated prompt optimization based on evaluation feedback."""

    async def optimize(
        self,
        evaluation_results: Dict[str, Any],
        evidence_graph: EvidenceGraph,
        current_prompts: Optional[Dict[str, str]] = None,
    ) -> List[EvidenceNode]:
        """Analyse evaluation results and propose prompt improvements.

        Args:
            evaluation_results: Output from HungarianMatcher evaluation.
            evidence_graph: Current evidence graph for context.
            current_prompts: Current prompt templates (optional).

        Returns:
            List of EvidenceNodes representing prompt optimization proposals.
        """
        # Identify weak sections
        weak_sections = self._identify_weak_sections(evaluation_results)

        if not weak_sections:
            logger.info("auto_prompt_no_weak_sections")
            return []

        # Generate optimization proposals via LLM
        try:
            proposals = await self._generate_optimizations(
                weak_sections, current_prompts or {}
            )
        except Exception as e:
            logger.error("auto_prompt_generation_failed", error=str(e))
            return []

        # Wrap as evidence nodes
        optimization_nodes: List[EvidenceNode] = []
        for i, prop in enumerate(proposals):
            node = evidence_graph.add_evidence(
                evidence=Evidence(
                    content=json.dumps(prop, ensure_ascii=False),
                    content_type="prompt_optimization",
                    source="auto_prompt_optimizer",
                    metadata={
                        "diagnosis": prop.get("diagnosis", ""),
                        "target": prop.get("target", ""),
                        "expected_improvement": prop.get("expected_improvement", 0),
                    },
                ),
                belief=Belief(
                    content=f"Prompt optimization: {prop.get('diagnosis', 'unknown')}",
                    confidence=prop.get("expected_improvement", 0.5),
                    source_agent="evaluation_agent",
                ),
                intent=Intent(
                    intent_type=IntentType.OPTIMIZE_PROMPT,
                    description=f"Auto prompt optimization targeting {prop.get('target', 'unknown')}",
                    parameters={
                        "target": prop.get("target", ""),
                        "diagnosis": prop.get("diagnosis", ""),
                    },
                    source_agent="evaluation_agent",
                ),
                tags=["prompt_optimization", prop.get("target", "general")],
            )
            optimization_nodes.append(node)

        logger.info(
            "auto_prompt_proposals_generated",
            count=len(optimization_nodes),
        )
        return optimization_nodes

    @staticmethod
    def _identify_weak_sections(
        evaluation_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract weak sections from evaluation results."""
        weak = []
        section_scores = evaluation_results.get("section_scores", [])

        for section in section_scores:
            if not section.get("is_faithful", True) or section.get("score", 1.0) < 0.5:
                weak.append({
                    "section_idx": section.get("generated_idx"),
                    "score": section.get("score", 0.0),
                    "is_faithful": section.get("is_faithful", False),
                })

        # Also flag unmatched sections as weak
        for idx in evaluation_results.get("unmatched_sections", []):
            weak.append({
                "section_idx": idx,
                "score": 0.0,
                "is_faithful": False,
                "issue": "unmatched",
            })

        return weak

    async def _generate_optimizations(
        self,
        weak_sections: List[Dict[str, Any]],
        current_prompts: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Use LLM to generate prompt optimization proposals."""
        weak_summary = json.dumps(weak_sections, indent=2)
        prompt_summary = json.dumps(
            {k: v[:200] + "..." for k, v in current_prompts.items()},
            indent=2,
        ) if current_prompts else "(no current prompts provided)"

        messages = [
            SystemMessage(content=load_prompt("auto_prompt_system")),
            HumanMessage(content=(
                f"Weak sections from evaluation:\n{weak_summary}\n\n"
                f"Current prompts (truncated):\n{prompt_summary}"
            )),
        ]

        response = await llm_service.call(messages, prompt_name="auto_prompt_system")
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [{
                "diagnosis": "parse_error",
                "prompt_patch": content,
                "expected_improvement": 0.3,
                "target": "general",
            }]
