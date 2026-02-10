"""Response schemas for the presentation layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntermediateNode(BaseModel):
    """Serialized evidence node for frontend display."""

    node_id: str
    content: str
    content_type: str
    source: str = ""
    source_agent: str
    created_at: str = ""
    confidence: float
    belief_content: str = ""
    intent: str
    intent_type: str = ""
    tags: List[str] = Field(default_factory=list)
    parent_node_ids: List[str] = Field(default_factory=list)
    child_node_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntermediateStep(BaseModel):
    """A group of intermediate artifacts from one pipeline stage."""

    agent: str
    label: str
    nodes: List[IntermediateNode] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    session_id: str = Field(..., description="Session ID")
    response: str = Field(..., description="Generated response text")
    evidence_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of the evidence graph used",
    )
    pipeline_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline execution metrics",
    )
    intermediate_steps: List[IntermediateStep] = Field(
        default_factory=list,
        description="Intermediate artifacts from each pipeline stage",
    )
    phase_completed: str = Field(default="done", description="Final pipeline phase")


class EvidenceGraphResponse(BaseModel):
    """Response body for evidence graph inspection."""

    graph_id: str
    node_count: int
    edge_count: int
    nodes: List[Dict[str, Any]]
    agents_involved: List[str]
    avg_confidence: float


class PipelineStatusResponse(BaseModel):
    """Response body for pipeline status / HITL state."""

    session_id: str
    current_phase: str
    iteration_count: int
    is_waiting_for_human: bool
    evidence_summary: Dict[str, Any]
    agent_outputs: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    version: str
    environment: str
    components: Dict[str, str]
    timestamp: str
