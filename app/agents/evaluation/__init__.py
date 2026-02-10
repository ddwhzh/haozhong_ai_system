"""Evaluation Agent — automated quality assessment and prompt optimization.

- Self-supervised denoising + sparse matrix scoring function redesign
- DETR-inspired Hungarian matching for generated-vs-expected alignment
- Auto prompt optimization based on evaluation feedback
"""

from app.agents.evaluation.agent import EvaluationAgent

__all__ = ["EvaluationAgent"]
