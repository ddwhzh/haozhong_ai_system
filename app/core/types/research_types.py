"""Research domain types — Paper, Hypothesis, Experiment, JournalEntry.

These types model the artifacts flowing through the research pipeline:
- Paper: academic paper metadata from arXiv / Semantic Scholar
- Hypothesis: testable claim derived from literature synthesis
- Experiment: experiment definition + code + results
- JournalEntry: timestamped log of research activity per phase
- IterationRecord: summary of one full research iteration cycle
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    """Origin of a paper record."""

    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    WEB = "web"
    MANUAL = "manual"


class Paper(BaseModel):
    """Academic paper metadata."""

    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list)
    abstract: str = Field(default="", description="Paper abstract")
    year: Optional[int] = None
    url: str = Field(default="", description="Link to paper (arXiv, DOI, etc.)")
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    citation_count: int = Field(default=0)
    source: PaperSource = Field(default=PaperSource.WEB)
    categories: List[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def dedup_key(self) -> str:
        """Key for deduplication: prefer DOI, then arXiv ID, then title."""
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        return f"title:{self.title.lower().strip()}"


class HypothesisStatus(str, Enum):
    """Lifecycle of a research hypothesis."""

    PROPOSED = "proposed"
    TESTING = "testing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class Hypothesis(BaseModel):
    """A testable research hypothesis."""

    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    statement: str = Field(..., description="The hypothesis statement")
    rationale: str = Field(default="", description="Why this hypothesis was proposed")
    supporting_papers: List[str] = Field(
        default_factory=list, description="Paper IDs that support this hypothesis"
    )
    status: HypothesisStatus = Field(default=HypothesisStatus.PROPOSED)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    test_results: List[str] = Field(
        default_factory=list, description="Experiment IDs that tested this"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentStatus(str, Enum):
    """Lifecycle of an experiment."""

    DESIGNED = "designed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ExperimentResult(BaseModel):
    """Captured output from experiment execution."""

    stdout: str = Field(default="")
    stderr: str = Field(default="")
    exit_code: int = Field(default=-1)
    output_files: Dict[str, str] = Field(
        default_factory=dict, description="filename -> content for small text outputs"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed numeric/structured metrics"
    )
    elapsed_seconds: float = Field(default=0.0)


class Experiment(BaseModel):
    """An experiment definition with optional results."""

    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Short experiment name")
    description: str = Field(default="", description="What this experiment tests")
    hypothesis_id: Optional[str] = Field(
        default=None, description="Hypothesis being tested"
    )
    code: str = Field(default="", description="Python code to execute")
    dependencies: List[str] = Field(
        default_factory=list, description="pip packages required"
    )
    expected_metrics: List[str] = Field(
        default_factory=list, description="Metric names to track"
    )
    success_criteria: str = Field(
        default="", description="How to judge success/failure"
    )
    status: ExperimentStatus = Field(default=ExperimentStatus.DESIGNED)
    result: Optional[ExperimentResult] = None
    iteration: int = Field(default=1, description="Which research iteration")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchDecision(str, Enum):
    """Decision from the iteration planner."""

    ITERATE = "iterate"
    PIVOT = "pivot"
    CONCLUDE = "conclude"


class JournalEntry(BaseModel):
    """A single entry in the research journal."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = Field(..., description="Research phase that produced this entry")
    summary: str = Field(..., description="What happened in this phase")
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IterationRecord(BaseModel):
    """Summary of one full research iteration cycle."""

    iteration: int = Field(..., description="1-indexed iteration number")
    decision: ResearchDecision = Field(
        default=ResearchDecision.ITERATE,
        description="What the planner decided after this iteration",
    )
    direction: str = Field(
        default="", description="Next research direction (for iterate/pivot)"
    )
    key_findings: List[str] = Field(default_factory=list)
    hypotheses_tested: int = Field(default=0)
    hypotheses_confirmed: int = Field(default=0)
    experiments_run: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
