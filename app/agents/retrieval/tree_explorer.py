"""Tree Explorer - tree-shaped retrieval with branch-level backtracking.

Given QDMR decomposition steps, explores each step via its primary
retrieval strategy. If branch quality is below threshold, backtracks
to the fallback strategy.

Design:
    Root (classified query)
      -> Step 1 (QDMR decomposition step)
           -> Branch A (primary strategy, e.g. RAG)
           -> Branch B (fallback strategy, e.g. KG)   [only if A backtracks]
      -> Step 2
           -> Branch A
           -> Branch B
      -> ...
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agents.retrieval.kg_retriever import KGRetriever
from app.agents.retrieval.query_classifier import ClassificationResult, QType
from app.agents.retrieval.vector_retriever import VectorRetriever
from app.agents.retrieval.web_retriever import BraveWebRetriever
from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceNode,
    EdgeRelation,
)
from app.core.types.intent import Intent, IntentType


@dataclass
class TreeStep:
    """A single step in the QDMR decomposition tree."""

    step_idx: int
    question: str
    operator: str
    retrieval_strategy: str     # "rag", "kg", "both"
    depends_on: List[int]       # indices of steps this depends on
    results: List[EvidenceNode] = field(default_factory=list)
    quality: float = 0.0
    explored: bool = False
    backtracked: bool = False


class TreeExplorer:
    """Explores QDMR decomposition tree with branch-level backtracking.

    For each decomposition step:
    1. Execute primary retrieval strategy (determined by query type)
    2. Evaluate branch quality (mean confidence of results)
    3. If quality < threshold, backtrack and try fallback strategy
    4. Keep the best results
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        kg_retriever: KGRetriever,
        web_retriever: Optional[BraveWebRetriever] = None,
        backtrack_threshold: float = 0.4,
        max_branch_retries: int = 1,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.kg_retriever = kg_retriever
        self.web_retriever = web_retriever
        self.backtrack_threshold = backtrack_threshold
        self.max_branch_retries = max_branch_retries

    async def explore(
        self,
        steps: List[TreeStep],
        classification: ClassificationResult,
        evidence_graph: EvidenceGraph,
        parent_node: EvidenceNode,
    ) -> List[EvidenceNode]:
        """Explore all QDMR steps with branch-level backtracking.

        Independent steps (no unresolved dependencies) are executed in
        parallel via asyncio.gather for reduced latency.

        Args:
            steps: Ordered QDMR decomposition steps.
            classification: Query classification result.
            evidence_graph: Shared evidence graph.
            parent_node: Parent node (classification node) to link from.

        Returns:
            All evidence nodes collected across steps.
        """
        all_nodes: List[EvidenceNode] = []
        step_results: Dict[int, List[EvidenceNode]] = {}
        completed_indices: set[int] = set()

        # Group steps into waves: each wave contains steps whose
        # dependencies are already satisfied by previous waves.
        waves = self._build_execution_waves(steps)

        for wave_idx, wave in enumerate(waves):
            if len(wave) == 1:
                # Single step — run directly (no gather overhead)
                nodes = await self._explore_single_step(
                    wave[0], classification, evidence_graph,
                    parent_node, step_results,
                )
                all_nodes.extend(nodes)
                step_results[wave[0].step_idx] = wave[0].results
                completed_indices.add(wave[0].step_idx)
            else:
                # Multiple independent steps — run in parallel
                logger.info(
                    "tree_parallel_wave",
                    wave=wave_idx + 1,
                    step_count=len(wave),
                    step_indices=[s.step_idx for s in wave],
                )
                coros = [
                    self._explore_single_step(
                        step, classification, evidence_graph,
                        parent_node, step_results,
                    )
                    for step in wave
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)

                for step, result in zip(wave, results):
                    if isinstance(result, BaseException):
                        logger.exception(
                            "tree_parallel_step_failed",
                            step=step.step_idx,
                            error=str(result),
                        )
                        step.explored = True
                        step.quality = 0.0
                        continue
                    all_nodes.extend(result)
                    step_results[step.step_idx] = step.results
                    completed_indices.add(step.step_idx)

        return all_nodes

    async def _explore_single_step(
        self,
        step: TreeStep,
        classification: ClassificationResult,
        evidence_graph: EvidenceGraph,
        parent_node: EvidenceNode,
        step_results: Dict[int, List[EvidenceNode]],
    ) -> List[EvidenceNode]:
        """Explore a single QDMR step with backtracking.

        Returns:
            All nodes produced by this step (step_node + retrieval results).
        """
        nodes: List[EvidenceNode] = []

        logger.info(
            "tree_step_start",
            step=step.step_idx,
            question=step.question[:80],
            strategy=step.retrieval_strategy,
        )

        # Create a sub-query evidence node for this step
        step_node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=step.question,
                content_type="tree_step",
                source="tree_explorer",
                metadata={
                    "step_idx": step.step_idx,
                    "operator": step.operator,
                    "retrieval_strategy": step.retrieval_strategy,
                    "depends_on": step.depends_on,
                    "q_type": classification.q_type.value,
                },
            ),
            belief=Belief(
                content=f"QDMR step {step.step_idx}: {step.operator} - {step.question}",
                confidence=0.9,
                source_agent="retrieval_agent",
            ),
            intent=Intent(
                intent_type=IntentType.DECOMPOSE_QUERY,
                description=(
                    f"QDMR step {step.step_idx}: {step.operator} - {step.question}"
                ),
                source_agent="retrieval_agent",
            ),
            parent_node_ids=[parent_node.node_id],
            relation=EdgeRelation.DECOMPOSES_TO,
            tags=["tree_step", step.operator, classification.q_type.value.lower()],
        )
        nodes.append(step_node)

        # Determine strategies to try
        primary = self._resolve_strategy(
            step.retrieval_strategy, classification.strategy.primary_strategy
        )
        fallback = self._resolve_strategy(
            step.retrieval_strategy, classification.strategy.fallback_strategy
        )

        # Try primary branch
        branch_nodes = await self._execute_branch(
            step, step_node, evidence_graph, primary
        )
        quality = self._evaluate_quality(branch_nodes)

        logger.info(
            "tree_branch_result",
            step=step.step_idx,
            strategy=primary,
            node_count=len(branch_nodes),
            quality=round(quality, 3),
        )

        # Backtrack if quality is low
        if quality < self.backtrack_threshold and fallback != primary:
            logger.info(
                "tree_backtrack",
                step=step.step_idx,
                from_strategy=primary,
                to_strategy=fallback,
                quality=round(quality, 3),
                threshold=self.backtrack_threshold,
            )
            step.backtracked = True

            fallback_nodes = await self._execute_branch(
                step, step_node, evidence_graph, fallback
            )
            fallback_quality = self._evaluate_quality(fallback_nodes)

            logger.info(
                "tree_fallback_result",
                step=step.step_idx,
                strategy=fallback,
                node_count=len(fallback_nodes),
                quality=round(fallback_quality, 3),
            )

            # Keep the best branch
            if fallback_quality > quality:
                branch_nodes = fallback_nodes
                quality = fallback_quality
            else:
                # Keep both (original had some results too)
                branch_nodes = branch_nodes + fallback_nodes

        step.results = branch_nodes
        step.quality = quality
        step.explored = True
        nodes.extend(branch_nodes)

        logger.info(
            "tree_step_complete",
            step=step.step_idx,
            total_nodes=len(branch_nodes),
            quality=round(quality, 3),
            backtracked=step.backtracked,
        )

        return nodes

    @staticmethod
    def _build_execution_waves(steps: List[TreeStep]) -> List[List[TreeStep]]:
        """Group steps into execution waves based on dependencies.

        Steps with no dependencies (or all dependencies resolved in prior
        waves) are grouped into the same wave for parallel execution.

        Returns:
            Ordered list of waves; each wave is a list of independent steps.
        """
        if not steps:
            return []

        resolved: set[int] = set()
        remaining = list(steps)
        waves: List[List[TreeStep]] = []

        while remaining:
            wave: List[TreeStep] = []
            still_remaining: List[TreeStep] = []

            for step in remaining:
                deps = set(step.depends_on)
                if deps.issubset(resolved):
                    wave.append(step)
                else:
                    still_remaining.append(step)

            if not wave:
                # Circular dependency or unresolvable — force remaining into one wave
                logger.warning(
                    "tree_unresolvable_deps",
                    remaining_steps=[s.step_idx for s in still_remaining],
                )
                waves.append(still_remaining)
                break

            waves.append(wave)
            resolved.update(s.step_idx for s in wave)
            remaining = still_remaining

        return waves

    async def _execute_branch(
        self,
        step: TreeStep,
        step_node: EvidenceNode,
        evidence_graph: EvidenceGraph,
        strategy: str,
    ) -> List[EvidenceNode]:
        """Execute a single retrieval branch for a step.

        Args:
            step: The QDMR step to retrieve for.
            step_node: The step's evidence node (used as parent).
            evidence_graph: Shared evidence graph.
            strategy: "rag", "kg", or "both".

        Returns:
            List of evidence nodes from this branch.
        """
        nodes: List[EvidenceNode] = []

        if strategy in ("rag", "both"):
            rag_nodes = await self.vector_retriever.retrieve(
                [step_node], evidence_graph
            )
            nodes.extend(rag_nodes)

            # Web search augmentation for RAG-like branches.
            if self.web_retriever and self.web_retriever.enabled:
                web_nodes = await self.web_retriever.retrieve(
                    [step_node], evidence_graph
                )
                nodes.extend(web_nodes)

        if strategy in ("kg", "both"):
            kg_nodes = await self.kg_retriever.retrieve(
                [step_node], evidence_graph
            )
            nodes.extend(kg_nodes)

            # If KG-only branch returns empty, optionally try web as fallback evidence source.
            if (
                strategy == "kg"
                and not kg_nodes
                and self.web_retriever
                and self.web_retriever.enabled
            ):
                web_nodes = await self.web_retriever.retrieve(
                    [step_node], evidence_graph
                )
                nodes.extend(web_nodes)

        return nodes

    @staticmethod
    def _evaluate_quality(nodes: List[EvidenceNode]) -> float:
        """Evaluate branch quality as mean confidence of results."""
        if not nodes:
            return 0.0
        total_conf = sum(n.belief.confidence for n in nodes)
        return total_conf / len(nodes)

    @staticmethod
    def _resolve_strategy(step_strategy: str, type_strategy: str) -> str:
        """Resolve which strategy to use.

        If the step has a specific strategy, use it.
        Otherwise fall back to the type-level strategy.
        """
        if step_strategy in ("rag", "kg"):
            return step_strategy
        # "both" or unknown -> use type-level
        return type_strategy
