"""Unit tests for ResearchState and reducers."""

import pytest

from app.core.types.research_state import (
    ResearchPhase,
    ResearchState,
    _append_journal,
    _merge_experiments,
    _merge_hypotheses,
    _merge_papers,
)
from app.core.types.research_types import (
    Experiment,
    Hypothesis,
    HypothesisStatus,
    JournalEntry,
    Paper,
    PaperSource,
)


class TestResearchState:
    """Tests for ResearchState model."""

    def test_default_state(self):
        """默认状态正确初始化."""
        state = ResearchState()
        assert state.research_topic == ""
        assert state.current_phase == "scout"
        assert state.iteration_count == 1
        assert state.max_iterations == 5
        assert state.literature == []
        assert state.hypotheses == []
        assert state.experiments == []
        assert state.research_journal == []
        assert state.final_report == ""

    def test_state_with_topic(self):
        """携带课题创建."""
        state = ResearchState(research_topic="RAG improvement")
        assert state.research_topic == "RAG improvement"

    def test_all_phases_defined(self):
        """所有阶段都有定义."""
        phases = [p.value for p in ResearchPhase]
        assert "scout" in phases
        assert "survey" in phases
        assert "synthesize" in phases
        assert "design" in phases
        assert "experiment" in phases
        assert "analyze" in phases
        assert "decide" in phases
        assert "conclude" in phases


class TestPaperReducer:
    """Tests for paper deduplication reducer."""

    def test_merge_empty(self):
        """空列表合并."""
        assert _merge_papers([], []) == []

    def test_merge_new_papers(self):
        """新论文正确追加."""
        p1 = Paper(title="Paper A")
        p2 = Paper(title="Paper B")
        merged = _merge_papers([p1], [p2])
        assert len(merged) == 2

    def test_merge_duplicate_by_title(self):
        """标题相同的论文去重."""
        p1 = Paper(title="Same Title")
        p2 = Paper(title="Same Title")
        merged = _merge_papers([p1], [p2])
        assert len(merged) == 1

    def test_merge_duplicate_by_doi(self):
        """DOI相同的论文去重."""
        p1 = Paper(title="Paper A", doi="10.1234/test")
        p2 = Paper(title="Paper A (copy)", doi="10.1234/test")
        merged = _merge_papers([p1], [p2])
        assert len(merged) == 1

    def test_merge_duplicate_by_arxiv_id(self):
        """arXiv ID相同的论文去重."""
        p1 = Paper(title="Paper A", arxiv_id="2401.12345")
        p2 = Paper(title="Paper A v2", arxiv_id="2401.12345")
        merged = _merge_papers([p1], [p2])
        assert len(merged) == 1

    def test_merge_different_papers(self):
        """不同论文不去重."""
        p1 = Paper(title="Paper A", doi="10.1234/a")
        p2 = Paper(title="Paper B", doi="10.1234/b")
        merged = _merge_papers([p1], [p2])
        assert len(merged) == 2


class TestHypothesisReducer:
    """Tests for hypothesis merge reducer."""

    def test_merge_new_hypothesis(self):
        """新假设追加."""
        h1 = Hypothesis(statement="H1")
        h2 = Hypothesis(statement="H2")
        merged = _merge_hypotheses([h1], [h2])
        assert len(merged) == 2

    def test_merge_update_same_id(self):
        """相同ID的假设被更新."""
        h1 = Hypothesis(statement="H1", confidence=0.5)
        h2 = Hypothesis(
            hypothesis_id=h1.hypothesis_id,
            statement="H1 updated",
            confidence=0.8,
            status=HypothesisStatus.CONFIRMED,
        )
        merged = _merge_hypotheses([h1], [h2])
        assert len(merged) == 1
        assert merged[0].statement == "H1 updated"
        assert merged[0].confidence == 0.8


class TestExperimentReducer:
    """Tests for experiment merge reducer."""

    def test_merge_new_experiment(self):
        """新实验追加."""
        e1 = Experiment(name="exp1")
        e2 = Experiment(name="exp2")
        merged = _merge_experiments([e1], [e2])
        assert len(merged) == 2

    def test_merge_update_same_id(self):
        """相同ID的实验被更新."""
        from app.core.types.research_types import ExperimentResult, ExperimentStatus

        e1 = Experiment(name="exp1")
        e2 = Experiment(
            experiment_id=e1.experiment_id,
            name="exp1",
            status=ExperimentStatus.COMPLETED,
            result=ExperimentResult(exit_code=0, stdout="ok"),
        )
        merged = _merge_experiments([e1], [e2])
        assert len(merged) == 1
        assert merged[0].status == ExperimentStatus.COMPLETED


class TestJournalReducer:
    """Tests for journal append reducer."""

    def test_append_journal(self):
        """日志追加不去重."""
        j1 = JournalEntry(phase="scout", summary="S1")
        j2 = JournalEntry(phase="survey", summary="S2")
        merged = _append_journal([j1], [j2])
        assert len(merged) == 2
        assert merged[0].phase == "scout"
        assert merged[1].phase == "survey"
