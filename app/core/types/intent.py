"""Intent model — represents what an agent *wants to achieve* by emitting evidence.

Intent is the pragmatic primitive: it tells downstream agents *why* a piece of
evidence was produced, so they can decide how to consume it.

Design notes
------------
- `IntentType` enumerates the fixed set of inter-agent communication purposes.
- `Intent` carries the type plus optional structured parameters and a natural-language
  description for LLM-based agents that need textual context.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Fixed taxonomy of inter-agent communication intents."""

    # Retrieval agent intents
    RETRIEVE = "retrieve"               # "I want to find relevant information"
    DECOMPOSE_QUERY = "decompose_query" # "I'm breaking a query into sub-queries"
    CLASSIFY_QUERY = "classify_query"   # "I'm classifying the query type (QDMR)"
    TREE_EXPLORE = "tree_explore"       # "I'm exploring a QDMR decomposition tree"
    KG_EXPLORE = "kg_explore"           # "I'm traversing the knowledge graph"
    CASCADE_FUSE = "cascade_fuse"       # "I'm fusing multi-hop retrieval results"

    # Generation agent intents
    GENERATE = "generate"               # "I want to produce content"
    PROPOSE_PLAN = "propose_plan"       # "I'm proposing a generation plan (proposals)"
    REFINE = "refine"                   # "I'm refining a previous generation"

    # Evaluation agent intents
    EVALUATE = "evaluate"               # "I want to judge quality"
    MATCH = "match"                     # "I'm performing Hungarian matching"
    OPTIMIZE_PROMPT = "optimize_prompt" # "I'm optimizing a prompt via auto-prompt"

    # Research agent intents
    RESEARCH_SCOUT = "research_scout"           # "I'm exploring a research domain"
    RESEARCH_SURVEY = "research_survey"         # "I'm retrieving academic literature"
    RESEARCH_SYNTHESIZE = "research_synthesize" # "I'm synthesizing knowledge"
    RESEARCH_DESIGN = "research_design"         # "I'm designing an experiment"
    RESEARCH_EXECUTE = "research_execute"       # "I'm executing experiment code"
    RESEARCH_ANALYZE = "research_analyze"       # "I'm analyzing experiment results"
    RESEARCH_DECIDE = "research_decide"         # "I'm deciding next research step"

    # Orchestration-level intents
    ROUTE = "route"                     # Gateway routing decision
    DELEGATE = "delegate"               # Delegate sub-task to another agent
    TERMINATE = "terminate"             # Signal pipeline completion


class Intent(BaseModel):
    """A structured intent attached to evidence or agent output."""

    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent_type: IntentType = Field(..., description="Categorical intent type")
    description: str = Field(
        default="",
        description="Free-text explanation of the intent (consumed by LLM agents)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters specific to this intent type",
    )
    source_agent: str = Field(..., description="Agent that declared this intent")
    target_agent: Optional[str] = Field(
        default=None,
        description="Intended recipient agent (None = broadcast)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
