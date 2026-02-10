"""Unit tests for the redesigned scoring function.

Tests cover:
- Self-supervised denoising
- Sparse (asymmetric) scoring
- Combined scoring
- Symmetry-breaking verification
"""

import math

import numpy as np
import pytest

from app.agents.evaluation.scoring import ScoringFunction


class TestScoringFunctionDense:
    """Tests for the denoised dense scoring component."""

    def setup_method(self):
        self.sf = ScoringFunction(sparse_weight=0.0, denoising_strength=0.1)

    def test_identical_vectors_score_high(self):
        """完全相同的向量应得分接近1."""
        vec = [1.0, 0.0, 0.5, -0.3] * 10
        result = self.sf.score(vec, vec, [], [])
        assert result["dense_score"] > 0.95

    def test_orthogonal_vectors_score_low(self):
        """正交向量应得分接近0."""
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        result = self.sf.score(a, b, [], [])
        assert abs(result["dense_score"]) < 0.15

    def test_opposite_vectors_score_negative(self):
        """反向向量应得分为负."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = self.sf.score(a, b, [], [])
        assert result["dense_score"] < 0

    def test_empty_vectors_return_zero(self):
        """空向量应返回0."""
        result = self.sf.score([], [], [], [])
        assert result["dense_score"] == 0.0

    def test_denoising_changes_score(self):
        """开启去噪 vs 关闭去噪应产生不同分数."""
        a = np.random.RandomState(42).randn(100).tolist()
        b = np.random.RandomState(43).randn(100).tolist()

        sf_denoise = ScoringFunction(sparse_weight=0.0, denoising_strength=0.3)
        sf_no_denoise = ScoringFunction(sparse_weight=0.0, denoising_strength=0.0)

        s1 = sf_denoise.score(a, b, [], [])["dense_score"]
        s2 = sf_no_denoise.score(a, b, [], [])["dense_score"]
        # 去噪应该改变分数(除非恰好相等, 概率极低)
        assert s1 != s2


class TestScoringFunctionSparse:
    """Tests for the asymmetric sparse scoring component."""

    def setup_method(self):
        self.sf = ScoringFunction(sparse_weight=1.0, denoising_strength=0.0)

    def test_perfect_overlap_scores_high(self):
        """完全重叠的tokens应得高分."""
        tokens = ["知识", "图谱", "语义"]
        result = self.sf.score([], [], tokens, tokens)
        assert result["sparse_score"] > 0.3

    def test_no_overlap_scores_zero(self):
        """无重叠tokens应得0分."""
        q = ["知识", "图谱"]
        d = ["天气", "预报"]
        result = self.sf.score([], [], q, d)
        assert result["sparse_score"] == 0.0

    def test_asymmetry(self):
        """score(A,B) != score(B,A) — 打破对称性."""
        q = ["知识", "图谱", "深度", "学习"]
        d = ["知识", "图谱", "图谱", "图谱", "应用", "应用"]

        # sparse_score(q, d) 和 sparse_score(d, q) 应该不同
        result_qd = self.sf.score([], [], q, d)
        result_dq = self.sf.score([], [], d, q)

        assert result_qd["sparse_score"] != result_dq["sparse_score"]

    def test_asymmetry_factor_nonzero(self):
        """asymmetry_factor应大于0(当两方向分数不同时)."""
        q = ["A", "B", "C"]
        d = ["A", "A", "A", "D", "E"]
        result = self.sf.score([], [], q, d)
        assert result["asymmetry_factor"] > 0

    def test_empty_tokens_return_zero(self):
        """空tokens列表应返回0."""
        result = self.sf.score([], [], [], [])
        assert result["sparse_score"] == 0.0


class TestScoringFunctionCombined:
    """Tests for the combined dense + sparse score."""

    def test_combined_is_weighted_sum(self):
        """combined = dense_weight * dense + sparse_weight * sparse."""
        sf = ScoringFunction(sparse_weight=0.3, denoising_strength=0.0)
        a = [1.0, 0.5, 0.3]
        b = [0.9, 0.4, 0.2]
        tokens = ["hello"]

        result = sf.score(a, b, tokens, tokens)
        expected = 0.7 * result["dense_score"] + 0.3 * result["sparse_score"]
        assert abs(result["combined_score"] - expected) < 0.001

    def test_sparse_weight_zero_equals_pure_dense(self):
        """sparse_weight=0时等于纯dense."""
        sf = ScoringFunction(sparse_weight=0.0)
        a = [1.0, 0.0]
        b = [0.8, 0.2]
        result = sf.score(a, b, ["a"], ["a"])
        assert abs(result["combined_score"] - result["dense_score"]) < 0.001


class TestBatchScore:
    """Tests for batch_score matrix computation."""

    def test_batch_score_shape(self):
        """batch_score返回正确形状的矩阵."""
        sf = ScoringFunction()
        queries = [
            {"embedding": [1, 0, 0], "tokens": ["a"]},
            {"embedding": [0, 1, 0], "tokens": ["b"]},
        ]
        docs = [
            {"embedding": [1, 0, 0], "tokens": ["a"]},
            {"embedding": [0, 1, 0], "tokens": ["b"]},
            {"embedding": [0, 0, 1], "tokens": ["c"]},
        ]
        matrix = sf.batch_score(queries, docs)
        assert matrix.shape == (2, 3)

    def test_batch_score_diagonal_high_for_identical(self):
        """相同的query-doc对应位置应有较高分数."""
        sf = ScoringFunction(sparse_weight=0.0, denoising_strength=0.0)
        items = [
            {"embedding": [1, 0, 0], "tokens": ["a"]},
            {"embedding": [0, 1, 0], "tokens": ["b"]},
        ]
        matrix = sf.batch_score(items, items)
        # 对角线应该最高
        assert matrix[0, 0] > matrix[0, 1]
        assert matrix[1, 1] > matrix[1, 0]
