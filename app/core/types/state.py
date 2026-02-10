"""PipelineState — top-level LangGraph state for the orchestration layer.

State machine phases (based on disciplined agent workflow):

    [*] --> Explore       (retrieve: RAG + KG parallel search, cascade fusion)
    Explore --> Freeze    (write context_manifest + evidence snapshot)
    Freeze --> Plan       (proposal decomposition, write todo DAG)
    Plan --> Execute      (generate content per proposal, bounded steps)
    Execute --> Validate  (evaluation: scoring + Hungarian matching)
    Validate --> Done     (pass: assemble output)
    Validate --> Rollback (fail: expand/repair manifest, retry)
    Rollback --> Explore  (re-enter retrieval with feedback)

Design rule: Worker nodes read from / write to this state via AgentInput/AgentOutput.
Only the Gateway node is allowed to set `next_agent` (routing).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.core.types.agent_io import AgentOutput
from app.core.types.evidence import EvidenceGraph


class PipelinePhase(str, Enum):
    """State machine phases for the orchestration pipeline."""

    EXPLORE = "explore"       # Retrieval: RAG + KG parallel, cascade fusion
    FREEZE = "freeze"         # Snapshot context_manifest + evidence
    PLAN = "plan"             # Proposal decomposition (generation planning)
    EXECUTE = "execute"       # Content generation per proposal
    VALIDATE = "validate"     # Evaluation: scoring + Hungarian matching
    DONE = "done"             # Final output assembly
    ROLLBACK = "rollback"     # Validation failed, repair and retry
    HUMAN_REVIEW = "human_review"  # HITL checkpoint


def _merge_agent_outputs(
    existing: List[AgentOutput],
    new: List[AgentOutput],
) -> List[AgentOutput]:
    """Reducer: append new agent outputs to the list."""
    return existing + new


class PipelineState(BaseModel):
    """Global state flowing through the LangGraph orchestration graph.

    Fields
    ------
    messages : list
        LangChain-style message list (auto-merged by `add_messages`).
    query : str
        Current user query being processed.
    evidence_graph : EvidenceGraph
        The shared evidence graph that all agents contribute to.
    agent_outputs : list[AgentOutput]
        Ordered log of every agent execution result.
    current_phase : str
        Which pipeline phase we're in (see PipelinePhase enum).
    next_agent : str | None
        Set by the Gateway to route to the next agent.
    iteration_count : int
        Global iteration counter (safety limit).
    max_iterations : int
        Hard ceiling on total iterations.
    rollback_count : int
        How many times we've rolled back (to prevent infinite loops).
    max_rollbacks : int
        Hard ceiling on rollbacks.
    context_manifest : dict
        Frozen context after Explore phase — evidence summary + gaps.
    plan_dag : list
        Ordered list of proposal steps from Plan phase.
    validation_feedback : str
        Feedback from Validate phase when rolling back.
    metadata : dict
        Arbitrary metadata (user_id, session_id, trace_id, …).
    """

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation messages (LangChain format, auto-merged)",
    )
    query: str = Field(default="", description="Current user query")
    evidence_graph: EvidenceGraph = Field(
        default_factory=EvidenceGraph,
        description="Shared evidence graph across all agents",
    )
    agent_outputs: Annotated[List[AgentOutput], _merge_agent_outputs] = Field(
        default_factory=list,
        description="Ordered log of agent execution results",
    )
    current_phase: str = Field(
        default=PipelinePhase.EXPLORE.value,
        description="Current pipeline phase (see PipelinePhase enum)",
    )
    next_agent: Optional[str] = Field(
        default=None,
        description="Next agent to route to (set ONLY by Gateway)",
    )
    iteration_count: int = Field(default=0, description="Global iteration counter")
    max_iterations: int = Field(default=10, description="Hard ceiling on iterations")
    rollback_count: int = Field(default=0, description="Rollback counter")
    max_rollbacks: int = Field(default=1, description="Hard ceiling on rollbacks")
    context_manifest: Dict[str, Any] = Field(
        default_factory=dict,
        description="Frozen context after Explore: evidence summary, gaps, retrieval stats",
    )
    plan_dag: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered proposal steps from Plan phase",
    )
    validation_feedback: str = Field(
        default="",
        description="Feedback from Validate when rolling back",
    )
    final_output: str = Field(default="", description="Final generated response text")
    metadata: Dict[str, Any] = Field(default_factory=dict)
