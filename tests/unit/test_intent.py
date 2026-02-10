"""Unit tests for Intent model."""

import pytest

from app.core.types.intent import Intent, IntentType


class TestIntentType:
    """Tests for IntentType enumeration."""

    def test_retrieval_intents_exist(self):
        """检索相关intent类型存在."""
        assert IntentType.RETRIEVE == "retrieve"
        assert IntentType.DECOMPOSE_QUERY == "decompose_query"
        assert IntentType.KG_EXPLORE == "kg_explore"
        assert IntentType.CASCADE_FUSE == "cascade_fuse"

    def test_generation_intents_exist(self):
        """生成相关intent类型存在."""
        assert IntentType.GENERATE == "generate"
        assert IntentType.PROPOSE_PLAN == "propose_plan"
        assert IntentType.REFINE == "refine"

    def test_evaluation_intents_exist(self):
        """评估相关intent类型存在."""
        assert IntentType.EVALUATE == "evaluate"
        assert IntentType.MATCH == "match"
        assert IntentType.OPTIMIZE_PROMPT == "optimize_prompt"

    def test_orchestration_intents_exist(self):
        """编排相关intent类型存在."""
        assert IntentType.ROUTE == "route"
        assert IntentType.DELEGATE == "delegate"
        assert IntentType.TERMINATE == "terminate"


class TestIntent:
    """Tests for Intent model."""

    def test_create_intent_basic(self):
        """基本创建."""
        intent = Intent(
            intent_type=IntentType.RETRIEVE,
            source_agent="retrieval_agent",
        )
        assert intent.intent_type == IntentType.RETRIEVE
        assert intent.source_agent == "retrieval_agent"
        assert intent.intent_id  # UUID

    def test_intent_with_parameters(self):
        """携带结构化参数."""
        intent = Intent(
            intent_type=IntentType.KG_EXPLORE,
            description="BFS遍历知识图谱",
            parameters={"max_hops": 3, "seed_entities": ["e1"]},
            source_agent="retrieval_agent",
            target_agent="generation_agent",
        )
        assert intent.parameters["max_hops"] == 3
        assert intent.target_agent == "generation_agent"

    def test_intent_target_default_none(self):
        """target_agent默认为None(广播)."""
        intent = Intent(
            intent_type=IntentType.GENERATE,
            source_agent="gen",
        )
        assert intent.target_agent is None
