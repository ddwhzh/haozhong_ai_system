"""Unit tests for Belief model."""

import pytest

from app.core.types.belief import Belief, BeliefStatus, BeliefUpdate


class TestBelief:
    """Tests for Belief creation and validation."""

    def test_create_belief_basic(self):
        """基本创建: 必填字段."""
        belief = Belief(
            content="测试信念",
            confidence=0.8,
            source_agent="test_agent",
        )
        assert belief.content == "测试信念"
        assert belief.confidence == 0.8
        assert belief.source_agent == "test_agent"
        assert belief.status == BeliefStatus.ACTIVE
        assert belief.belief_id  # UUID自动生成

    def test_confidence_out_of_range_rejected(self):
        """超出[0, 1]范围的confidence应被Pydantic拒绝."""
        with pytest.raises(Exception):
            Belief(content="high", confidence=1.5, source_agent="test")

        with pytest.raises(Exception):
            Belief(content="low", confidence=-0.5, source_agent="test")

    def test_confidence_boundary_values(self):
        """边界值: 0和1应有效."""
        b0 = Belief(content="zero", confidence=0.0, source_agent="test")
        assert b0.confidence == 0.0

        b1 = Belief(content="one", confidence=1.0, source_agent="test")
        assert b1.confidence == 1.0

    def test_supporting_evidence_ids_default_empty(self):
        """supporting_evidence_ids默认为空列表."""
        belief = Belief(content="t", confidence=0.5, source_agent="test")
        assert belief.supporting_evidence_ids == []

    def test_belief_with_evidence_ids(self):
        """可以附带证据ID列表."""
        belief = Belief(
            content="t",
            confidence=0.9,
            source_agent="test",
            supporting_evidence_ids=["ev1", "ev2"],
        )
        assert len(belief.supporting_evidence_ids) == 2

    def test_belief_status_transitions(self):
        """状态枚举值有效."""
        assert BeliefStatus.ACTIVE == "active"
        assert BeliefStatus.REVISED == "revised"
        assert BeliefStatus.RETRACTED == "retracted"


class TestBeliefUpdate:
    """Tests for BeliefUpdate model."""

    def test_create_belief_update(self):
        """基本创建."""
        update = BeliefUpdate(
            target_belief_id="belief-123",
            new_confidence=0.6,
            reason="新证据降低了置信度",
            proposed_by="evaluation_agent",
        )
        assert update.target_belief_id == "belief-123"
        assert update.new_confidence == 0.6
        assert update.proposed_by == "evaluation_agent"
        assert update.update_id  # UUID自动生成

    def test_update_with_status_change(self):
        """可以附带状态变更."""
        update = BeliefUpdate(
            target_belief_id="b-1",
            new_confidence=0.0,
            new_status=BeliefStatus.RETRACTED,
            reason="证据被证伪",
            proposed_by="eval",
        )
        assert update.new_status == BeliefStatus.RETRACTED
