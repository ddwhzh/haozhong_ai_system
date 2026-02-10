"""Unit tests for the Hungarian Matcher."""

import pytest

from app.agents.evaluation.hungarian_matcher import HungarianMatcher
from app.agents.evaluation.scoring import ScoringFunction


class TestHungarianMatcher:
    """Tests for DETR-inspired Hungarian matching."""

    def setup_method(self):
        self.sf = ScoringFunction(sparse_weight=0.0, denoising_strength=0.0)
        self.matcher = HungarianMatcher(scoring_fn=self.sf)

    def test_empty_predictions(self):
        """空predictions应返回全部unmatched_targets."""
        result = self.matcher.match(
            predictions=[],
            targets=[{"embedding": [1, 0], "tokens": ["a"], "content": "t", "id": "1"}],
        )
        assert result["matches"] == []
        assert result["unmatched_targets"] == [0]
        assert result["overall_score"] == 0.0

    def test_empty_targets(self):
        """空targets应返回全部unmatched_preds."""
        result = self.matcher.match(
            predictions=[{"embedding": [1, 0], "tokens": ["a"], "content": "t", "id": "1"}],
            targets=[],
        )
        assert result["matches"] == []
        assert result["unmatched_preds"] == [0]

    def test_perfect_match(self):
        """完全匹配: prediction和target相同."""
        items = [
            {"embedding": [1, 0, 0], "tokens": ["a"], "content": "a", "id": "1"},
            {"embedding": [0, 1, 0], "tokens": ["b"], "content": "b", "id": "2"},
        ]
        result = self.matcher.match(predictions=items, targets=items)
        assert len(result["matches"]) == 2
        assert result["overall_score"] > 0.9
        assert result["unmatched_preds"] == []
        assert result["unmatched_targets"] == []

    def test_partial_match(self):
        """部分匹配: 数量不等时有unmatched."""
        preds = [
            {"embedding": [1, 0, 0], "tokens": ["a"], "content": "a", "id": "1"},
            {"embedding": [0, 1, 0], "tokens": ["b"], "content": "b", "id": "2"},
            {"embedding": [0, 0, 1], "tokens": ["c"], "content": "c", "id": "3"},
        ]
        targets = [
            {"embedding": [1, 0, 0], "tokens": ["a"], "content": "a", "id": "t1"},
            {"embedding": [0, 1, 0], "tokens": ["b"], "content": "b", "id": "t2"},
        ]
        result = self.matcher.match(predictions=preds, targets=targets)
        assert len(result["matches"]) == 2
        assert len(result["unmatched_preds"]) == 1

    def test_match_returns_correct_structure(self):
        """匹配结果包含所有必要字段."""
        items = [
            {"embedding": [1, 0], "tokens": ["x"], "content": "x", "id": "1"},
        ]
        result = self.matcher.match(predictions=items, targets=items)
        assert "matches" in result
        assert "unmatched_preds" in result
        assert "unmatched_targets" in result
        assert "cost_matrix" in result
        assert "score_matrix" in result
        assert "overall_score" in result

    def test_match_tuple_format(self):
        """每个match是(pred_idx, target_idx, score)三元组."""
        items = [
            {"embedding": [1, 0], "tokens": ["x"], "content": "x", "id": "1"},
        ]
        result = self.matcher.match(predictions=items, targets=items)
        for m in result["matches"]:
            assert len(m) == 3
            assert isinstance(m[0], int)
            assert isinstance(m[1], int)
            assert isinstance(m[2], float)


class TestFaithfulnessEvaluation:
    """Tests for faithfulness evaluation using Hungarian matching."""

    def setup_method(self):
        self.matcher = HungarianMatcher()

    def test_evaluate_faithfulness_all_faithful(self):
        """所有section都忠实时, faithfulness_rate=1.0."""
        items = [
            {"embedding": [1, 0, 0], "tokens": ["a"], "content": "a", "id": "1"},
        ]
        result = self.matcher.evaluate_faithfulness(
            generated_sections=items,
            evidence_sources=items,
            threshold=0.5,
        )
        assert result["faithfulness_rate"] >= 0.5

    def test_evaluate_faithfulness_has_structure(self):
        """evaluate_faithfulness返回正确结构."""
        items = [
            {"embedding": [1, 0], "tokens": ["x"], "content": "x", "id": "1"},
        ]
        result = self.matcher.evaluate_faithfulness(
            generated_sections=items,
            evidence_sources=items,
        )
        assert "faithfulness_rate" in result
        assert "section_scores" in result
        assert "unmatched_sections" in result
        assert "overall_score" in result
