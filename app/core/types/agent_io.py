"""Agent I/O contracts — typed input/output for every worker node.

Every worker agent in the business layer receives an `AgentInput` and returns
an `AgentOutput`.  This ensures:
1. Worker nodes NEVER need to know about routing (no `Command.goto`).
2. The orchestration layer can inspect outputs uniformly.
3. Evidence graph mutations are explicit and auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.types.belief import Belief, BeliefUpdate
from app.core.types.evidence import EvidenceGraph, EvidenceNode


class AgentStatus(str, Enum):
    """Exit status of an agent execution."""

    SUCCESS = "success"
    PARTIAL = "partial"           # Produced results but with caveats
    NEEDS_MORE_INFO = "needs_more_info"
    ERROR = "error"


class AgentInput(BaseModel):
    """Standard input envelope for every worker agent.

    The orchestration layer constructs this from the global PipelineState
    before invoking a worker node.
    """

    query: str = Field(..., description="Current user query or sub-task description")
    evidence_graph: EvidenceGraph = Field(
        default_factory=EvidenceGraph,
        description="Shared evidence graph (read-only snapshot for the worker)",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific context / parameters from the gateway",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Recent conversation messages for context",
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum iterations this agent is allowed to run",
    )


class AgentOutput(BaseModel):
    """Standard output envelope from every worker agent.

    Workers return this; the orchestration layer merges the new evidence
    nodes and belief updates into the global PipelineState.
    """

    agent_name: str = Field(..., description="Name of the agent that produced this output")
    status: AgentStatus = Field(default=AgentStatus.SUCCESS)
    new_nodes: List[EvidenceNode] = Field(
        default_factory=list,
        description="New evidence nodes to add to the graph",
    )
    belief_updates: List[BeliefUpdate] = Field(
        default_factory=list,
        description="Proposed revisions to existing beliefs",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of what the agent did",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific metrics (latency, token count, categorical labels, etc.)",
    )
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
