"""Belief model — represents an agent's confidence in a piece of evidence or conclusion.

Belief is the epistemic primitive: every evidence node, retrieval result, and generation
output carries a Belief that quantifies *how much the producing agent trusts it*.

Design notes
------------
- `confidence` ∈ [0, 1]: calibrated probability-like score.
- `source_agent`: which agent produced this belief (provenance).
- `supporting_evidence_ids`: traceability back to evidence nodes.
- `BeliefUpdate`: immutable delta — when an agent revises a belief it emits an update,
  the orchestration layer decides whether to accept.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class BeliefStatus(str, Enum):
    """Lifecycle status of a belief."""

    ACTIVE = "active"
    REVISED = "revised"
    RETRACTED = "retracted"


class Belief(BaseModel):
    """A single belief held by an agent about a piece of information."""

    belief_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Natural-language statement this belief is about")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated confidence score in [0, 1]"
    )
    source_agent: str = Field(..., description="Name of the agent that produced this belief")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of EvidenceNode objects that support this belief",
    )
    status: BeliefStatus = Field(default=BeliefStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class BeliefUpdate(BaseModel):
    """Immutable record of a belief revision.

    When an agent wants to change an existing belief it creates a BeliefUpdate;
    the orchestration gateway decides whether to apply it.
    """

    update_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_belief_id: str = Field(..., description="ID of the belief being revised")
    new_confidence: float = Field(..., ge=0.0, le=1.0)
    new_status: Optional[BeliefStatus] = None
    reason: str = Field(default="", description="Why this revision is proposed")
    proposed_by: str = Field(..., description="Agent proposing the update")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
