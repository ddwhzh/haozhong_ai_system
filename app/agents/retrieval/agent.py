"""Retrieval Agent — QDMR-based tree exploration retrieval pipeline.

Pipeline stages:
1. Query Classification (QDMR)     — classify query into 8 QDMR types
2. QDMR Decomposition              — decompose into ordered tree steps
3. Tree Exploration (with backtrack)— explore each step via primary/fallback strategy
4. Cascade Fusion (multi-round)     — fuse all tree results

The tree explorer implements branch-level backtracking:
if a branch's quality < threshold, it backtracks to try the fallback strategy.

Output: Evidence graph enriched with retrieval results carrying belief + intent,
        enabling downstream agents to consume structured, trustable evidence.
"""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.retrieval.cascade_fuser import CascadeFuser
from app.agents.retrieval.kg_retriever import KGRetriever
from app.agents.retrieval.query_classifier import QueryClassifier
from app.agents.retrieval.query_decomposer import QueryDecomposer
from app.agents.retrieval.query_router import QueryComplexity, route_query
from app.agents.retrieval.tree_explorer import TreeExplorer, TreeStep
from app.agents.retrieval.vector_retriever import VectorRetriever
from app.agents.retrieval.web_retriever import BraveWebRetriever
from app.core.config import settings
from app.core.logging import logger
from app.core.types.agent_io import AgentInput, AgentOutput, AgentStatus


class RetrievalAgent(BaseAgent):
    """QDMR-based retrieval: classify -> decompose -> tree explore -> fuse."""

    agent_name = "retrieval_agent"

    def __init__(self) -> None:
        self.query_classifier = QueryClassifier(
            max_depth=settings.RETRIEVAL_MAX_TREE_DEPTH,
            backtrack_threshold=settings.RETRIEVAL_BACKTRACK_THRESHOLD,
        )
        self.query_decomposer = QueryDecomposer(
            max_sub_queries=settings.RETRIEVAL_MAX_SUB_QUERIES,
        )
        self.vector_retriever = VectorRetriever(
            top_k=settings.RETRIEVAL_TOP_K,
        )
        self.kg_retriever = KGRetriever(
            max_hops=settings.RETRIEVAL_KG_MAX_HOPS,
        )
        self.web_retriever = BraveWebRetriever(
            top_k=settings.BRAVE_SEARCH_TOP_K,
        )
        self.tree_explorer = TreeExplorer(
            vector_retriever=self.vector_retriever,
            kg_retriever=self.kg_retriever,
            web_retriever=self.web_retriever,
            backtrack_threshold=settings.RETRIEVAL_BACKTRACK_THRESHOLD,
            max_branch_retries=settings.RETRIEVAL_MAX_BRANCH_RETRIES,
        )
        self.cascade_fuser = CascadeFuser(
            max_rounds=settings.RETRIEVAL_CASCADE_ROUNDS,
        )

    async def _execute(self, agent_input: AgentInput) -> AgentOutput:
        """Run the retrieval pipeline with complexity-based routing.

        Routes:
            simple   → direct retrieval (skip QDMR classification + decomposition)
            standard → full QDMR pipeline
            complex  → full QDMR pipeline (same, but enables future differentiation)
        """
        query = agent_input.query
        evidence_graph = agent_input.evidence_graph

        logger.info("retrieval_pipeline_start", query=query)

        # -- Stage 0: Query Complexity Routing (heuristic, no LLM) ---------
        routing = route_query(query)

        if routing.complexity == QueryComplexity.SIMPLE:
            return await self._execute_simple(query, evidence_graph, routing)

        # -- Stage 1: QDMR Query Classification ----------------------------
        classification, class_node = await self.query_classifier.classify(
            query, evidence_graph
        )
        classification_nodes = [class_node]

        # -- Stage 2: QDMR Decomposition into Tree Steps -------------------
        tree_steps = await self.query_decomposer.qdmr_decompose(
            query=query,
            q_type=classification.q_type,
            max_steps=settings.RETRIEVAL_MAX_SUB_QUERIES,
        )

        # -- Stage 3: Tree Exploration with Backtracking --------------------
        explored_nodes = await self.tree_explorer.explore(
            steps=tree_steps,
            classification=classification,
            evidence_graph=evidence_graph,
            parent_node=class_node,
        )

        # Separate tree step nodes from retrieval result nodes
        tree_step_nodes = [
            n for n in explored_nodes
            if n.evidence.content_type == "tree_step"
        ]
        retrieval_nodes = [
            n for n in explored_nodes
            if n.evidence.content_type != "tree_step"
        ]

        # Separate breadth (RAG) vs depth (KG) for cascade fuser
        breadth_nodes = [
            n for n in retrieval_nodes
            if n.evidence.content_type in ("document_chunk", "web_result")
        ]
        depth_nodes = [
            n for n in retrieval_nodes
            if n.evidence.content_type in ("kg_match", "reasoning_chain")
        ]
        web_nodes = [
            n for n in retrieval_nodes
            if n.evidence.content_type == "web_result"
        ]

        # -- Stage 4: Cascade Multi-Round Fusion ----------------------------
        fused_nodes = await self.cascade_fuser.fuse(
            query=query,
            evidence_graph=evidence_graph,
            breadth_nodes=breadth_nodes,
            depth_nodes=depth_nodes,
        )

        # Collect all new nodes for output
        all_new_nodes = (
            classification_nodes
            + tree_step_nodes
            + retrieval_nodes
            + fused_nodes
        )

        # Determine status based on results
        if fused_nodes:
            final_confidence = fused_nodes[-1].belief.confidence
            status = AgentStatus.SUCCESS if final_confidence >= 0.5 else AgentStatus.PARTIAL
        elif retrieval_nodes:
            status = AgentStatus.PARTIAL
        else:
            status = AgentStatus.NEEDS_MORE_INFO

        # Count backtracks
        backtrack_count = sum(1 for s in tree_steps if s.backtracked)

        logger.info(
            "retrieval_pipeline_complete",
            query=query,
            q_type=classification.q_type.value,
            tree_steps=len(tree_steps),
            breadth_results=len(breadth_nodes),
            depth_chains=len(depth_nodes),
            web_results=len(web_nodes),
            fused_results=len(fused_nodes),
            backtracks=backtrack_count,
            status=status.value,
        )

        return AgentOutput(
            agent_name=self.agent_name,
            status=status,
            new_nodes=all_new_nodes,
            summary=(
                f"[{classification.q_type.value}] "
                f"Explored {len(tree_steps)} QDMR steps "
                f"({backtrack_count} backtracks), "
                f"retrieved {len(breadth_nodes)} RAG/Web + {len(depth_nodes)} KG "
                f"(web={len(web_nodes)}), "
                f"fused into {len(fused_nodes)} evidence nodes."
            ),
            metrics={
                "q_type": classification.q_type.value,
                "classification_confidence": classification.confidence,
                "tree_step_count": len(tree_steps),
                "breadth_result_count": len(breadth_nodes),
                "depth_chain_count": len(depth_nodes),
                "web_result_count": len(web_nodes),
                "fused_count": len(fused_nodes),
                "backtrack_count": backtrack_count,
            },
        )

    # ── Simple query fast path ───────────────────────────────────────────

    async def _execute_simple(
        self,
        query: str,
        evidence_graph: "EvidenceGraph",
        routing: "RoutingResult",
    ) -> AgentOutput:
        """Fast path for simple queries: direct retrieval without QDMR.

        Skips classification + decomposition LLM calls. Creates a single
        TreeStep with the original query and runs vector + KG retrieval
        directly, then fuses.
        """
        from app.agents.retrieval.query_router import RoutingResult
        from app.core.types.evidence import EvidenceGraph

        logger.info(
            "retrieval_simple_fast_path",
            query=query,
            reason=routing.reason,
        )

        # Create a single synthetic tree step for the original query
        single_step = TreeStep(
            step_idx=1,
            question=query,
            operator="select",
            retrieval_strategy="both",
            depends_on=[],
        )

        # Build a minimal classification result for tree explorer compatibility
        from app.agents.retrieval.query_classifier import (
            ClassificationResult,
            QType,
            StrategyConfig,
        )

        synthetic_classification = ClassificationResult(
            q_type=QType.Q_SELECT,
            confidence=0.9,
            reasoning=f"simple_route: {routing.reason}",
            decomposition_hint=[],
            strategy=StrategyConfig.for_type(
                QType.Q_SELECT, self.tree_explorer.backtrack_threshold
            ),
        )

        # Create a lightweight classification node (no LLM call)
        from app.core.types.belief import Belief
        from app.core.types.evidence import Evidence
        from app.core.types.intent import Intent, IntentType

        class_node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=f'{{"q_type": "QSelect", "confidence": 0.9, '
                f'"reasoning": "simple_route: {routing.reason}", '
                f'"primary_strategy": "both", "fallback_strategy": "both"}}',
                content_type="query_classification",
                source="query_router",
                metadata={
                    "q_type": "QSelect",
                    "confidence": 0.9,
                    "routing_complexity": routing.complexity.value,
                    "routing_reason": routing.reason,
                },
            ),
            belief=Belief(
                content=f"Query routed as simple (QSelect): {routing.reason}",
                confidence=0.9,
                source_agent="retrieval_agent",
            ),
            intent=Intent(
                intent_type=IntentType.DECOMPOSE_QUERY,
                description=f"Simple query routed directly: {routing.reason}",
                source_agent="retrieval_agent",
            ),
            tags=["query_classification", "simple_route", "qselect"],
        )

        # Explore the single step (reuse tree explorer for consistency)
        explored_nodes = await self.tree_explorer.explore(
            steps=[single_step],
            classification=synthetic_classification,
            evidence_graph=evidence_graph,
            parent_node=class_node,
        )

        # Separate nodes
        tree_step_nodes = [
            n for n in explored_nodes if n.evidence.content_type == "tree_step"
        ]
        retrieval_nodes = [
            n for n in explored_nodes if n.evidence.content_type != "tree_step"
        ]
        breadth_nodes = [
            n for n in retrieval_nodes
            if n.evidence.content_type in ("document_chunk", "web_result")
        ]
        depth_nodes = [
            n for n in retrieval_nodes
            if n.evidence.content_type in ("kg_match", "reasoning_chain")
        ]

        # Cascade fuse (single round is usually enough for simple queries)
        fused_nodes = await self.cascade_fuser.fuse(
            query=query,
            evidence_graph=evidence_graph,
            breadth_nodes=breadth_nodes,
            depth_nodes=depth_nodes,
        )

        all_new_nodes = [class_node] + tree_step_nodes + retrieval_nodes + fused_nodes

        if fused_nodes:
            final_confidence = fused_nodes[-1].belief.confidence
            status = AgentStatus.SUCCESS if final_confidence >= 0.5 else AgentStatus.PARTIAL
        elif retrieval_nodes:
            status = AgentStatus.PARTIAL
        else:
            status = AgentStatus.NEEDS_MORE_INFO

        logger.info(
            "retrieval_simple_complete",
            query=query,
            breadth_results=len(breadth_nodes),
            depth_chains=len(depth_nodes),
            fused_results=len(fused_nodes),
            status=status.value,
        )

        return AgentOutput(
            agent_name=self.agent_name,
            status=status,
            new_nodes=all_new_nodes,
            summary=(
                f"[simple_route] Direct retrieval: "
                f"{len(breadth_nodes)} RAG + {len(depth_nodes)} KG, "
                f"fused into {len(fused_nodes)} evidence nodes."
            ),
            metrics={
                "q_type": "QSelect",
                "routing_complexity": routing.complexity.value,
                "tree_step_count": 1,
                "breadth_result_count": len(breadth_nodes),
                "depth_chain_count": len(depth_nodes),
                "fused_count": len(fused_nodes),
                "backtrack_count": 0,
            },
        )
