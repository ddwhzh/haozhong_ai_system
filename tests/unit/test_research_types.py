"""Unit tests for research domain types."""

import pytest

from app.core.types.research_types import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisStatus,
    IterationRecord,
    JournalEntry,
    Paper,
    PaperSource,
    ResearchDecision,
)


class TestPaper:
    """Tests for Paper model."""

    def test_default_paper(self):
        """默认值正确初始化."""
        p = Paper(title="Test Paper")
        assert p.title == "Test Paper"
        assert p.authors == []
        assert p.source == PaperSource.WEB
        assert p.relevance_score == 0.0

    def test_dedup_key_doi(self):
        """DOI优先作为去重key."""
        p = Paper(title="Test", doi="10.1234/test")
        assert p.dedup_key() == "doi:10.1234/test"

    def test_dedup_key_arxiv(self):
        """arXiv ID作为备选去重key."""
        p = Paper(title="Test", arxiv_id="2401.12345")
        assert p.dedup_key() == "arxiv:2401.12345"

    def test_dedup_key_title(self):
        """标题作为最后备选去重key."""
        p = Paper(title="Test Paper Title")
        assert p.dedup_key() == "title:test paper title"

    def test_dedup_key_priority(self):
        """DOI > arXiv ID > title 的优先级."""
        p = Paper(title="Test", doi="10.1234/test", arxiv_id="2401.12345")
        assert p.dedup_key() == "doi:10.1234/test"

    def test_paper_with_metadata(self):
        """完整元数据创建."""
        p = Paper(
            title="RAG Survey",
            authors=["Alice", "Bob"],
            abstract="A survey of RAG systems",
            year=2025,
            url="https://arxiv.org/abs/2401.12345",
            arxiv_id="2401.12345",
            citation_count=42,
            source=PaperSource.ARXIV,
            categories=["cs.AI", "cs.CL"],
            relevance_score=0.85,
        )
        assert p.year == 2025
        assert p.citation_count == 42
        assert len(p.categories) == 2


class TestHypothesis:
    """Tests for Hypothesis model."""

    def test_default_hypothesis(self):
        """默认状态为proposed."""
        h = Hypothesis(statement="Test hypothesis")
        assert h.status == HypothesisStatus.PROPOSED
        assert h.confidence == 0.5

    def test_hypothesis_lifecycle(self):
        """假设状态可以更新."""
        h = Hypothesis(statement="Test")
        h.status = HypothesisStatus.TESTING
        assert h.status == HypothesisStatus.TESTING
        h.status = HypothesisStatus.CONFIRMED
        assert h.status == HypothesisStatus.CONFIRMED

    def test_hypothesis_test_results(self):
        """测试结果可以追加."""
        h = Hypothesis(statement="Test")
        h.test_results.append("exp-1")
        h.test_results.append("exp-2")
        assert len(h.test_results) == 2


class TestExperiment:
    """Tests for Experiment model."""

    def test_default_experiment(self):
        """默认状态为designed."""
        e = Experiment(name="test_exp")
        assert e.status == ExperimentStatus.DESIGNED
        assert e.code == ""
        assert e.result is None

    def test_experiment_with_result(self):
        """实验结果可以填充."""
        r = ExperimentResult(
            stdout='{"accuracy": 0.95}',
            stderr="",
            exit_code=0,
            metrics={"accuracy": 0.95},
            elapsed_seconds=12.5,
        )
        e = Experiment(name="test", result=r)
        assert e.result.exit_code == 0
        assert e.result.metrics["accuracy"] == 0.95


class TestJournalAndIteration:
    """Tests for JournalEntry and IterationRecord."""

    def test_journal_entry(self):
        """日志条目正确创建."""
        j = JournalEntry(phase="scout", summary="Explored domain")
        assert j.phase == "scout"
        assert j.summary == "Explored domain"
        assert j.entry_id  # has uuid

    def test_iteration_record(self):
        """迭代记录正确创建."""
        r = IterationRecord(
            iteration=1,
            decision=ResearchDecision.ITERATE,
            direction="focus on graph RAG",
            key_findings=["finding 1", "finding 2"],
            hypotheses_tested=3,
            hypotheses_confirmed=1,
        )
        assert r.iteration == 1
        assert r.decision == ResearchDecision.ITERATE
        assert r.direction == "focus on graph RAG"
        assert len(r.key_findings) == 2
