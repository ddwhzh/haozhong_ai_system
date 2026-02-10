"""Unit tests for generation keyword filtering robustness."""

import pytest

from app.agents.generation.content_generator import ContentGenerator
from app.core.nlp import query_keyword_extractor
from app.core.types.belief import Belief
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode
from app.core.types.intent import Intent, IntentType


def _retrieval_node(content_type: str, content: str, confidence: float = 0.8) -> EvidenceNode:
    """Build a retrieval-style evidence node."""
    return EvidenceNode(
        evidence=Evidence(
            content=content,
            content_type=content_type,
            source="test_source",
            metadata={},
        ),
        belief=Belief(
            content="test belief",
            confidence=confidence,
            source_agent="retrieval_agent",
        ),
        intent=Intent(
            intent_type=IntentType.RETRIEVE,
            description="test retrieve",
            source_agent="retrieval_agent",
        ),
        tags=[content_type],
    )


class TestKeywordExtraction:
    """Verify Chinese comparison queries are split into entity keywords."""

    def test_extractor_extracts_entity_level_keywords(self):
        query = "鹿紫云能不能打过五条悟"
        keywords = query_keyword_extractor.extract_keywords_sync(query)
        assert "鹿紫云" in keywords
        assert "五条悟" in keywords
        assert "鹿紫云能不能打过五条悟" not in keywords

    def test_keyword_match_allows_alias_one_char_mismatch(self):
        assert query_keyword_extractor.match_keywords(
            ["鹿紫云一"], "鹿紫云是咒术师"
        )


class TestKeywordFilterFallback:
    """Verify empty keyword filtering does not drop all evidence."""

    @pytest.mark.asyncio
    async def test_content_generator_fallback_keeps_candidates_when_filtered_empty(self):
        graph = EvidenceGraph()
        graph.add_node(_retrieval_node("web_result", "五条悟是现代最强咒术师。", 0.9))

        slots = [{"name": "核心回答", "description": "直接回答", "evidence_hint": "web_result"}]
        # English gibberish query creates strict keyword that won't match Chinese evidence.
        gathered = await ContentGenerator._gather_evidence_by_hint(
            evidence_graph=graph,
            slots=slots,
            query="abcxyzfoo",
        )
        assert "web_result" in gathered
        assert len(gathered["web_result"]) == 1
