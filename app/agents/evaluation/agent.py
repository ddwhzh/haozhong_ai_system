"""Evaluation Agent — automated quality assessment with Hungarian matching.

Pipeline:
1. Scoring: Redesigned score function (self-supervised denoising + sparse matrix)
2. Hungarian Matching: DETR-inspired bipartite alignment of generated ↔ evidence
3. Auto Prompt: Identify weak sections and propose prompt optimizations

Addresses:
- Vector retrieval's poor fine-grained similarity discrimination
- Cosine similarity's symmetry problem
- Manual prompt engineering bottleneck
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

from app.agents.base import BaseAgent
from app.agents.evaluation.auto_prompt import AutoPromptOptimizer
from app.agents.evaluation.hungarian_matcher import HungarianMatcher
from app.agents.evaluation.scoring import ScoringFunction
from app.core.config import settings
from app.core.logging import logger
from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus
from app.core.types.belief import Belief, BeliefUpdate
from app.core.types.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType
from app.infra.llm import llm_service


class EvaluationAgent(BaseAgent):
    """Evaluation pipeline: score -> match -> auto-prompt optimize."""

    agent_name = "evaluation_agent"

    def __init__(self) -> None:
        self.scoring_fn = ScoringFunction(
            sparse_weight=settings.EVALUATION_SPARSE_WEIGHT,
        )
        self.matcher = HungarianMatcher(scoring_fn=self.scoring_fn)
        self.auto_prompt = AutoPromptOptimizer()
        embed_kwargs = {
            "model": settings.EMBEDDING_MODEL,
            "api_key": settings.OPENAI_API_KEY,
        }
        if settings.OPENAI_API_BASE:
            embed_kwargs["base_url"] = settings.OPENAI_API_BASE
        self._embeddings = OpenAIEmbeddings(**embed_kwargs)

    async def _execute(self, agent_input: AgentInput) -> AgentOutput:
        """Run the evaluation pipeline.

        1) Collect generated content nodes and evidence source nodes
        2) Embed them for scoring
        3) Run Hungarian matching (faithfulness evaluation)
        4) If quality is below threshold, run auto-prompt optimization
        5) Emit evaluation result nodes + belief updates
        """
        evidence_graph = agent_input.evidence_graph
        query = agent_input.query

        logger.info("evaluation_pipeline_start", query=query)

        # ── Collect generated and evidence nodes ─────────────────────────
        generated_nodes = [
            n for n in evidence_graph.nodes.values()
            if n.evidence.content_type == "generated_content"
        ]
        evidence_nodes = [
            n for n in evidence_graph.nodes.values()
            if n.evidence.content_type in (
                "fused_evidence",
                "document_chunk",
                "web_result",
                "kg_match",
                "reasoning_chain",
            )
        ]

        if not generated_nodes:
            return AgentOutput(
                agent_name=self.agent_name,
                status=AgentStatus.NEEDS_MORE_INFO,
                error_message="No generated content to evaluate",
                summary="Evaluation skipped: no generated content found.",
            )

        # ── Embed for scoring ────────────────────────────────────────────
        gen_dicts = await self._prepare_for_matching(generated_nodes)
        evi_dicts = await self._prepare_for_matching(evidence_nodes)

        # ── Hungarian Matching ───────────────────────────────────────────
        eval_result = self.matcher.evaluate_faithfulness(
            generated_sections=gen_dicts,
            evidence_sources=evi_dicts,
            threshold=settings.EVALUATION_CONFIDENCE_THRESHOLD,
        )

        # Create evaluation result node
        eval_node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=json.dumps(eval_result, ensure_ascii=False, default=str),
                content_type="evaluation_result",
                source="evaluation_agent",
                metadata={
                    "faithfulness_rate": eval_result.get("faithfulness_rate", 0),
                    "overall_score": eval_result.get("overall_score", 0),
                    "sections_evaluated": len(generated_nodes),
                },
            ),
            belief=Belief(
                content=f"Evaluation: faithfulness={eval_result.get('faithfulness_rate', 0):.2f}",
                confidence=eval_result.get("overall_score", 0.5),
                source_agent=self.agent_name,
            ),
            intent=Intent(
                intent_type=IntentType.EVALUATE,
                description="Hungarian matching faithfulness evaluation",
                parameters=eval_result,
                source_agent=self.agent_name,
            ),
            parent_node_ids=[n.node_id for n in generated_nodes],
            relation=EdgeRelation.EVALUATED_BY,
            tags=["evaluation"],
        )

        new_nodes = [eval_node]
        belief_updates: List[BeliefUpdate] = []

        # Update beliefs on generated nodes based on evaluation
        for section in eval_result.get("section_scores", []):
            gen_idx = section.get("generated_idx")
            if gen_idx is not None and gen_idx < len(generated_nodes):
                gen_node = generated_nodes[gen_idx]
                belief_updates.append(BeliefUpdate(
                    target_belief_id=gen_node.belief.belief_id,
                    new_confidence=section.get("score", 0.5),
                    reason=f"Evaluation score: {section.get('score', 0):.3f}, faithful={section.get('is_faithful')}",
                    proposed_by=self.agent_name,
                ))

        # ── Auto Prompt Optimization (if quality below threshold) ────────
        overall_score = eval_result.get("overall_score", 0.0)
        if overall_score < settings.EVALUATION_CONFIDENCE_THRESHOLD:
            logger.info(
                "auto_prompt_triggered",
                overall_score=overall_score,
                threshold=settings.EVALUATION_CONFIDENCE_THRESHOLD,
            )
            prompt_nodes = await self.auto_prompt.optimize(
                evaluation_results=eval_result,
                evidence_graph=evidence_graph,
            )
            new_nodes.extend(prompt_nodes)

        status = (
            AgentStatus.SUCCESS
            if overall_score >= settings.EVALUATION_CONFIDENCE_THRESHOLD
            else AgentStatus.PARTIAL
        )

        logger.info(
            "evaluation_pipeline_complete",
            overall_score=overall_score,
            faithfulness=eval_result.get("faithfulness_rate", 0),
            auto_prompt_triggered=overall_score < settings.EVALUATION_CONFIDENCE_THRESHOLD,
            status=status.value,
        )

        return AgentOutput(
            agent_name=self.agent_name,
            status=status,
            new_nodes=new_nodes,
            belief_updates=belief_updates,
            summary=(
                f"Evaluation complete: overall_score={overall_score:.3f}, "
                f"faithfulness={eval_result.get('faithfulness_rate', 0):.3f}, "
                f"{len(belief_updates)} belief updates proposed."
            ),
            metrics={
                "overall_score": overall_score,
                "faithfulness_rate": eval_result.get("faithfulness_rate", 0),
                "sections_evaluated": len(generated_nodes),
                "belief_updates": len(belief_updates),
                "prompt_optimizations": len(new_nodes) - 1,
            },
        )

    async def _prepare_for_matching(
        self,
        nodes: List[EvidenceNode],
    ) -> List[Dict[str, Any]]:
        """Prepare evidence nodes for Hungarian matching by adding embeddings."""
        dicts = []
        for node in nodes:
            content = node.evidence.content
            tokens = content.lower().split()

            try:
                embedding = await self._embeddings.aembed_query(content[:8000])
            except Exception:
                embedding = []

            dicts.append({
                "id": node.node_id,
                "content": content,
                "tokens": tokens,
                "embedding": embedding,
            })
        return dicts
