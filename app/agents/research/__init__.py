"""Research Agent — automated research loop for any domain.

Pipeline phases:
1. Domain Scout     — explore the research landscape
2. Literature Survey — retrieve academic papers
3. Knowledge Synthesis — synthesize findings, generate hypotheses
4. Experiment Design — design experiments + generate code
5. Experiment Runner — execute code in sandbox
6. Result Analyzer  — analyze results against hypotheses
7. Iteration Planner — decide iterate / pivot / conclude
"""

from app.agents.research.agent import ResearchAgent

__all__ = ["ResearchAgent"]
