"""Core type system for the multi-agent architecture.

Exports the fundamental types used across all layers:
- Belief / Intent / Evidence / EvidenceGraph  — agent interaction primitives
- AgentInput / AgentOutput                    — worker I/O contracts
- PipelineState                               — top-level orchestration state
"""

from app.core.types.belief import Belief, BeliefUpdate
from app.core.types.intent import Intent, IntentType
from app.core.types.evidence import Evidence, EvidenceGraph, EvidenceNode, EvidenceEdge
from app.core.types.agent_io import AgentInput, AgentOutput
from app.core.types.state import PipelineState

__all__ = [
    "Belief",
    "BeliefUpdate",
    "Intent",
    "IntentType",
    "Evidence",
    "EvidenceGraph",
    "EvidenceNode",
    "EvidenceEdge",
    "AgentInput",
    "AgentOutput",
    "PipelineState",
]
