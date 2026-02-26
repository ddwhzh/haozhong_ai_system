"""ResearchState — top-level LangGraph state for the research pipeline.

State machine phases for the automated research loop:

    [*] --> Scout        (domain exploration via web search + LLM)
    Scout --> Survey     (literature retrieval: arXiv + Semantic Scholar)
    Survey --> Synthesize(knowledge synthesis, gap analysis, hypothesis generation)
    Synthesize --> Design(experiment design + code generation)
    Design --> Experiment(sandbox code execution)
    Experiment --> Analyze(result analysis, hypothesis validation)
    Analyze --> Decide   (iterate / pivot / conclude)
    Decide --> Survey    (iterate: refine approach)
    Decide --> Scout     (pivot: change direction)
    Decide --> Conclude  (conclude: generate final report)

Design rule: Worker nodes read from / write to this state.
Only the ResearchGateway is allowed to set routing (Command.goto).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.types.research_types import (
    Experiment,
    Hypothesis,
    IterationRecord,
    JournalEntry,
    Paper,
)


class ResearchPhase(str, Enum):
    """State machine phases for the research pipeline."""

    SCOUT = "scout"
    SURVEY = "survey"
    SYNTHESIZE = "synthesize"
    DESIGN = "design"
    EXPERIMENT = "experiment"
    ANALYZE = "analyze"
    DECIDE = "decide"
    CONCLUDE = "conclude"


def _merge_papers(existing: List[Paper], new: List[Paper]) -> List[Paper]:
    """Reducer: deduplicate papers by dedup_key."""
    seen = {p.dedup_key() for p in existing}
    merged = list(existing)
    for p in new:
        key = p.dedup_key()
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged


def _merge_hypotheses(
    existing: List[Hypothesis], new: List[Hypothesis]
) -> List[Hypothesis]:
    """Reducer: merge hypotheses, update status if same ID."""
    by_id = {h.hypothesis_id: h for h in existing}
    for h in new:
        by_id[h.hypothesis_id] = h
    return list(by_id.values())


def _merge_experiments(
    existing: List[Experiment], new: List[Experiment]
) -> List[Experiment]:
    """Reducer: merge experiments, update if same ID."""
    by_id = {e.experiment_id: e for e in existing}
    for e in new:
        by_id[e.experiment_id] = e
    return list(by_id.values())


def _append_journal(
    existing: List[JournalEntry], new: List[JournalEntry]
) -> List[JournalEntry]:
    """Reducer: append journal entries."""
    return existing + new


def _append_iterations(
    existing: List[IterationRecord], new: List[IterationRecord]
) -> List[IterationRecord]:
    """Reducer: append iteration records."""
    return existing + new


class ResearchState(BaseModel):
    """Global state flowing through the research LangGraph.

    Fields
    ------
    research_topic : str
        The high-level research question or topic.
    domain_context : dict
        Accumulated domain knowledge (keywords, directions, key concepts).
    literature : list[Paper]
        Collected papers with deduplication (reducer).
    hypotheses : list[Hypothesis]
        Research hypotheses with lifecycle tracking (reducer).
    experiments : list[Experiment]
        Experiment definitions and results (reducer).
    research_journal : list[JournalEntry]
        Chronological log of all research activity (append-only reducer).
    iteration_history : list[IterationRecord]
        Summary per iteration cycle (append-only reducer).
    current_phase : str
        Which research phase we are in (see ResearchPhase enum).
    iteration_count : int
        Current iteration number (1-indexed).
    max_iterations : int
        Hard ceiling on research iterations.
    synthesis_report : str
        Latest knowledge synthesis output.
    analysis_report : str
        Latest experiment analysis output.
    final_report : str
        Final research report (populated at conclude).
    metadata : dict
        Arbitrary metadata (user_id, session_id, trace_id, ...).
    """

    research_topic: str = Field(default="", description="Research question or topic")
    domain_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated domain knowledge: keywords, directions, concepts",
    )
    literature: Annotated[List[Paper], _merge_papers] = Field(
        default_factory=list,
        description="Collected papers (deduplicated by reducer)",
    )
    hypotheses: Annotated[List[Hypothesis], _merge_hypotheses] = Field(
        default_factory=list,
        description="Research hypotheses with lifecycle tracking",
    )
    experiments: Annotated[List[Experiment], _merge_experiments] = Field(
        default_factory=list,
        description="Experiment definitions and results",
    )
    research_journal: Annotated[List[JournalEntry], _append_journal] = Field(
        default_factory=list,
        description="Chronological research activity log",
    )
    iteration_history: Annotated[List[IterationRecord], _append_iterations] = Field(
        default_factory=list,
        description="Summary per iteration cycle",
    )
    current_phase: str = Field(
        default=ResearchPhase.SCOUT.value,
        description="Current research phase",
    )
    iteration_count: int = Field(default=1, description="Current iteration (1-indexed)")
    max_iterations: int = Field(default=5, description="Hard ceiling on iterations")
    synthesis_report: str = Field(default="", description="Latest synthesis output")
    analysis_report: str = Field(default="", description="Latest analysis output")
    final_report: str = Field(default="", description="Final research report")
    metadata: Dict[str, Any] = Field(default_factory=dict)
