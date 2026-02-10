"""QDMR Query Classifier - classifies queries into 8 QDMR-based categories.

Based on Question Decomposition Meaning Representation (QDMR),
queries are classified to determine optimal retrieval strategy and
decomposition approach.

Categories:
    QSelect, QFilter, QChain, QComposition, QComparison,
    QAggregation, QBoolean, QUnion
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.belief import Belief
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode
from app.core.types.intent import Intent, IntentType
from app.infra.llm import llm_service
from app.prompts import load_prompt


class QType(str, Enum):
    """QDMR-based query classification types."""

    Q_SELECT = "QSelect"
    Q_FILTER = "QFilter"
    Q_CHAIN = "QChain"
    Q_COMPOSITION = "QComposition"
    Q_COMPARISON = "QComparison"
    Q_AGGREGATION = "QAggregation"
    Q_BOOLEAN = "QBoolean"
    Q_UNION = "QUnion"


@dataclass
class StrategyConfig:
    """Retrieval strategy configuration determined by query type."""

    primary_strategy: str       # "rag", "kg", or "both"
    fallback_strategy: str      # "kg", "rag", or "both"
    max_decomposition_depth: int
    backtrack_threshold: float

    @staticmethod
    def for_type(q_type: QType, backtrack_threshold: float = 0.4) -> "StrategyConfig":
        """Get default strategy config for a query type."""
        configs = {
            QType.Q_SELECT: StrategyConfig("rag", "kg", 2, backtrack_threshold),
            QType.Q_FILTER: StrategyConfig("rag", "kg", 2, backtrack_threshold),
            QType.Q_CHAIN: StrategyConfig("kg", "rag", 3, backtrack_threshold),
            QType.Q_COMPOSITION: StrategyConfig("both", "both", 3, backtrack_threshold),
            QType.Q_COMPARISON: StrategyConfig("both", "both", 3, backtrack_threshold),
            QType.Q_AGGREGATION: StrategyConfig("kg", "rag", 2, backtrack_threshold),
            QType.Q_BOOLEAN: StrategyConfig("rag", "kg", 1, backtrack_threshold),
            QType.Q_UNION: StrategyConfig("both", "both", 3, backtrack_threshold),
        }
        return configs.get(q_type, StrategyConfig("both", "both", 2, backtrack_threshold))


@dataclass
class ClassificationResult:
    """Result of query classification."""

    q_type: QType
    confidence: float
    reasoning: str
    decomposition_hint: List[str]
    strategy: StrategyConfig


class QueryClassifier:
    """Classifies queries into QDMR-based categories using LLM."""

    def __init__(self, max_depth: int = 3, backtrack_threshold: float = 0.4) -> None:
        self.max_depth = max_depth
        self.backtrack_threshold = backtrack_threshold

    async def classify(
        self,
        query: str,
        evidence_graph: EvidenceGraph,
    ) -> tuple[ClassificationResult, EvidenceNode]:
        """Classify query and return classification result + evidence node.

        Returns:
            Tuple of (ClassificationResult, EvidenceNode with classification info)
        """
        result = await self._llm_classify(query)

        # Create evidence node for classification
        node = evidence_graph.add_evidence(
            evidence=Evidence(
                content=json.dumps({
                    "q_type": result.q_type.value,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "primary_strategy": result.strategy.primary_strategy,
                    "fallback_strategy": result.strategy.fallback_strategy,
                }, ensure_ascii=False),
                content_type="query_classification",
                source="query_classifier",
                metadata={
                    "q_type": result.q_type.value,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "primary_strategy": result.strategy.primary_strategy,
                    "fallback_strategy": result.strategy.fallback_strategy,
                    "max_depth": result.strategy.max_decomposition_depth,
                },
            ),
            belief=Belief(
                content=f"Query classified as {result.q_type.value}: {result.reasoning}",
                confidence=result.confidence,
                source_agent="retrieval_agent",
            ),
            intent=Intent(
                intent_type=IntentType.DECOMPOSE_QUERY,
                description=f"Query classified as {result.q_type.value}: {result.reasoning}",
                source_agent="retrieval_agent",
            ),
            tags=["query_classification", result.q_type.value.lower()],
        )

        logger.info(
            "query_classified",
            q_type=result.q_type.value,
            confidence=result.confidence,
            primary_strategy=result.strategy.primary_strategy,
        )

        return result, node

    async def _llm_classify(self, query: str) -> ClassificationResult:
        """Use LLM to classify the query."""
        try:
            messages = [
                SystemMessage(content=load_prompt(
                    "query_classify_system",
                    max_depth=self.max_depth,
                )),
                HumanMessage(content=f"Classify this query:\n{query}"),
            ]

            response = await llm_service.call(
                messages, prompt_name="query_classify_system"
            )
            content = response.content.strip()

            # Parse JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            q_type_str = data.get("q_type", "QSelect")

            # Map string to enum
            try:
                q_type = QType(q_type_str)
            except ValueError:
                logger.warning("unknown_q_type", raw=q_type_str)
                q_type = QType.Q_SELECT

            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.7))))
            reasoning = data.get("reasoning", "")
            hints = data.get("decomposition_hint", [])
            if isinstance(hints, str):
                hints = [hints]

            strategy = StrategyConfig.for_type(q_type, self.backtrack_threshold)

            return ClassificationResult(
                q_type=q_type,
                confidence=confidence,
                reasoning=reasoning,
                decomposition_hint=hints,
                strategy=strategy,
            )

        except Exception as e:
            logger.error("query_classification_failed", error=str(e))
            # Fallback: default to QComposition (most general)
            q_type = QType.Q_COMPOSITION
            strategy = StrategyConfig.for_type(q_type, self.backtrack_threshold)
            return ClassificationResult(
                q_type=q_type,
                confidence=0.5,
                reasoning=f"Classification failed ({e}), defaulting to QComposition",
                decomposition_hint=[],
                strategy=strategy,
            )
