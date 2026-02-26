"""Query Complexity Router — lightweight heuristic-based query routing.

Routes queries into three complexity levels to avoid over-decomposition
of simple definitional queries and reduce unnecessary LLM calls.

Levels:
    simple   — single-fact / definitional queries → skip QDMR, direct retrieval
    standard — moderate complexity → standard QDMR pipeline
    complex  — multi-hop / comparison / multi-aspect → full tree pipeline

Design: heuristic regex + character-count rules; no LLM call needed.
Follows Adaptive RAG pattern (LangChain, 2025).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from app.core.logging import logger


class QueryComplexity(str, Enum):
    """Query complexity levels for routing decisions."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class RoutingResult:
    """Result of the query complexity router."""

    complexity: QueryComplexity
    reason: str
    skip_qdmr: bool
    skip_decomposition: bool


# ── Heuristic patterns ──────────────────────────────────────────────────

# Patterns that strongly indicate a simple definitional query
_SIMPLE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^什么是.{1,20}[?？]?$"),          # "什么是X?"
    re.compile(r"^.{1,20}是什么[?？]?$"),           # "X是什么?"
    re.compile(r"^.{1,15}的定义[?？]?$"),           # "X的定义?"
    re.compile(r"^.{1,15}的概念[?？]?$"),           # "X的概念?"
    re.compile(r"^.{1,20}使用什么.{1,10}[?？]?$"),  # "X使用什么Y?"
    re.compile(r"^.{1,20}用什么.{1,10}[?？]?$"),    # "X用什么Y?"
    re.compile(r"^.{1,20}借助什么.{1,10}[?？]?$"),  # "X借助什么Y?"
    re.compile(r"^.{1,20}的核心思想[?？]?$"),       # "X的核心思想?"
    re.compile(r"^.{1,20}的全称[?？]?$"),           # "X的全称?"
]

# Patterns that indicate multi-hop / complex queries
_COMPLEX_MARKERS: List[re.Pattern[str]] = [
    re.compile(r"分别"),                  # "分别" indicates multi-part
    re.compile(r"比较|对比|异同"),         # comparison markers
    re.compile(r"和.+各自"),              # "A和B各自"
    re.compile(r"有哪些.+以及"),           # multi-aspect enumeration
    re.compile(r"如何.+并且.+"),           # multi-step how-to
    re.compile(r"优缺点|优劣"),            # pros & cons = comparison
]


def route_query(
    query: str,
    max_simple_length: int = 25,
    min_complex_length: int = 30,
) -> RoutingResult:
    """Route a query to a complexity level using heuristics.

    Args:
        query: The original user query.
        max_simple_length: Queries shorter than this with simple patterns
                           are classified as simple.
        min_complex_length: Queries longer than this with complex markers
                            are classified as complex.

    Returns:
        RoutingResult with complexity level and routing flags.
    """
    stripped = query.strip().rstrip("?？。.")
    query_len = len(stripped)

    # ── Check for simple patterns ────────────────────────────────────
    if query_len <= max_simple_length:
        for pattern in _SIMPLE_PATTERNS:
            if pattern.match(query.strip()):
                result = RoutingResult(
                    complexity=QueryComplexity.SIMPLE,
                    reason=f"matched_simple_pattern: {pattern.pattern}",
                    skip_qdmr=True,
                    skip_decomposition=True,
                )
                logger.info(
                    "query_routed",
                    complexity=result.complexity.value,
                    reason=result.reason,
                    query_length=query_len,
                )
                return result

    # ── Check for complex markers ────────────────────────────────────
    complex_marker_count = sum(
        1 for pattern in _COMPLEX_MARKERS if pattern.search(query)
    )

    question_marks = query.count("?") + query.count("？")
    has_conjunction = bool(re.search(r"[,，;；]", query))

    if complex_marker_count >= 1 or (query_len >= min_complex_length and has_conjunction):
        result = RoutingResult(
            complexity=QueryComplexity.COMPLEX,
            reason=f"complex_markers={complex_marker_count}, len={query_len}, conjunction={has_conjunction}",
            skip_qdmr=False,
            skip_decomposition=False,
        )
        logger.info(
            "query_routed",
            complexity=result.complexity.value,
            reason=result.reason,
            query_length=query_len,
        )
        return result

    # ── Default: standard ────────────────────────────────────────────
    result = RoutingResult(
        complexity=QueryComplexity.STANDARD,
        reason=f"default_standard: len={query_len}, no_special_pattern",
        skip_qdmr=False,
        skip_decomposition=False,
    )
    logger.info(
        "query_routed",
        complexity=result.complexity.value,
        reason=result.reason,
        query_length=query_len,
    )
    return result
