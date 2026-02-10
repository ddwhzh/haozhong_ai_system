"""Request schemas for the presentation layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    query: str = Field(..., min_length=1, description="User query")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Max pipeline iterations")


class HumanReviewRequest(BaseModel):
    """Request body for HITL human review decision."""

    session_id: str = Field(..., description="Session ID of the paused pipeline")
    action: str = Field(
        ...,
        description="Review action: 'approve', 'reject', or 'edit'",
        pattern="^(approve|reject|edit)$",
    )
    feedback: str = Field(default="", description="Human feedback text")
    edits: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Direct edits to evidence graph (for action='edit')",
    )
