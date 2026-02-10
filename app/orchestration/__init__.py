"""Orchestration layer — LangGraph graph definition, gateway routing, HITL.

Architecture rule: ONLY the Gateway node is allowed to use Command.goto.
All worker nodes return plain dicts.
"""
