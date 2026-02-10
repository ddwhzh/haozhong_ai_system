"""Orchestration helper nodes — non-agent nodes in the LangGraph graph.

These handle phase transitions, context freezing, planning, rollback,
output assembly, and HITL interrupts.

Like worker agents, they return plain dicts (no Command.goto).

State machine:
    explore -> freeze -> plan -> execute -> validate -> done
    validate -> rollback -> explore (retry)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.core.config import settings
from app.core.logging import logger
from app.core.nlp import query_keyword_extractor
from app.core.types.state import PipelinePhase, PipelineState
from app.infra.llm import llm_service
from app.prompts import load_prompt


# ── Phase Transition ─────────────────────────────────────────────────────

async def phase_transition(state: PipelineState) -> Dict[str, Any]:
    """Advance the pipeline phase after a node completes.

    State machine transitions:
      explore   -> freeze
      freeze    -> plan
      plan      -> execute
      execute   -> validate
      validate  -> done | rollback (decided by post_validate logic)
      rollback  -> explore
    """
    current = state.current_phase
    transitions = {
        PipelinePhase.EXPLORE.value: PipelinePhase.FREEZE.value,
        PipelinePhase.FREEZE.value: PipelinePhase.PLAN.value,
        PipelinePhase.PLAN.value: PipelinePhase.EXECUTE.value,
        PipelinePhase.EXECUTE.value: PipelinePhase.VALIDATE.value,
    }
    next_phase = transitions.get(current, PipelinePhase.DONE.value)

    logger.info("phase_transition", from_phase=current, to_phase=next_phase)

    return {"current_phase": next_phase}


# ── Post-Validate Transition ─────────────────────────────────────────────

async def post_validate_transition(state: PipelineState) -> Dict[str, Any]:
    """After evaluation agent finishes, decide: Done or Rollback.

    Checks evaluation score against threshold.
    """
    eval_outputs = [
        o for o in state.agent_outputs
        if o.agent_name == "evaluation_agent"
    ]

    if not eval_outputs:
        return {"current_phase": PipelinePhase.DONE.value}

    latest_eval = eval_outputs[-1]
    overall_score = latest_eval.metrics.get("overall_score", 0.0)
    threshold = settings.EVALUATION_CONFIDENCE_THRESHOLD

    if overall_score >= threshold:
        # Quality passes
        if settings.HITL_ENABLED:
            logger.info("post_validate_hitl", score=overall_score)
            return {"current_phase": PipelinePhase.HUMAN_REVIEW.value}
        logger.info("post_validate_pass", score=overall_score)
        return {"current_phase": PipelinePhase.DONE.value}

    # Quality fails — can we rollback?
    if state.rollback_count < state.max_rollbacks:
        logger.info(
            "post_validate_rollback",
            score=overall_score,
            threshold=threshold,
            rollback_count=state.rollback_count,
        )
        feedback = (
            f"Evaluation score {overall_score:.2f} below threshold {threshold}. "
            f"Weak areas: {json.dumps(latest_eval.metrics.get('section_scores', []), default=str)[:200]}"
        )
        return {
            "current_phase": PipelinePhase.ROLLBACK.value,
            "validation_feedback": feedback,
        }

    # Exhausted rollbacks
    logger.warning("post_validate_exhausted", score=overall_score, rollback_count=state.rollback_count)
    return {"current_phase": PipelinePhase.DONE.value}


# ── Freeze Context ───────────────────────────────────────────────────────

async def freeze_context(state: PipelineState) -> Dict[str, Any]:
    """Freeze phase: snapshot the evidence graph into a context_manifest.

    The context_manifest captures:
    - Evidence summary (node count, types, confidence distribution)
    - Retrieval stats (how many from RAG vs KG)
    - Identified information gaps
    - Decision: is the evidence sufficient to proceed to planning?
    """
    evidence_graph = state.evidence_graph
    summary = evidence_graph.summary()

    # Classify nodes by source
    rag_nodes = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "document_chunk"
    ]
    web_nodes = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "web_result"
    ]
    kg_nodes = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "reasoning_chain"
    ]
    fused_nodes = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "fused_evidence"
    ]
    sub_queries = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "sub_query"
    ]

    context_manifest = {
        "query": state.query,
        "evidence_summary": summary,
        "rag_count": len(rag_nodes),
        "web_count": len(web_nodes),
        "kg_count": len(kg_nodes),
        "fused_count": len(fused_nodes),
        "sub_query_count": len(sub_queries),
        "avg_confidence": summary.get("avg_confidence", 0),
        "high_confidence_nodes": len(evidence_graph.get_high_confidence_nodes(0.7)),
        "sufficient": len(fused_nodes) > 0 or len(rag_nodes) + len(kg_nodes) > 0,
    }

    logger.info(
        "context_frozen",
        rag=len(rag_nodes),
        web=len(web_nodes),
        kg=len(kg_nodes),
        fused=len(fused_nodes),
        sufficient=context_manifest["sufficient"],
    )

    return {
        "context_manifest": context_manifest,
        "current_phase": PipelinePhase.FREEZE.value,
    }


# ── Plan Phase ───────────────────────────────────────────────────────────

async def plan_phase(state: PipelineState) -> Dict[str, Any]:
    """Plan phase: create a slot-filling generation plan (DAG) from evidence.

    Produces an ordered list of same-structured proposal items. Each item
    defines an aspect of the answer and slots to extract from the evidence
    graph. The Execute phase (Generation Agent) will fill these slots.

    This is the Fast RCNN analogy: the plan generates N region proposals,
    each with the SAME slot structure, to be filled by evidence extraction.
    """
    manifest = state.context_manifest
    query = state.query

    # Collect evidence preview for planning context
    evidence_texts = []
    for n in state.evidence_graph.nodes.values():
        if n.evidence.content_type in (
            "fused_evidence",
            "document_chunk",
            "web_result",
            "reasoning_chain",
            "kg_match",
        ):
            evidence_texts.append(n.evidence.content[:200])

    evidence_preview = "\n".join(evidence_texts[:10]) if evidence_texts else "(no evidence retrieved)"

    # Include validation feedback from rollback if present
    feedback_section = ""
    if state.validation_feedback:
        feedback_section = f"\n\nPrevious attempt feedback:\n{state.validation_feedback}"

    plan_prompt = (
        "Based on the query and available evidence, create a slot-filling generation plan.\n\n"
        "Query: %s\n\n"
        "Evidence preview:\n%s\n%s\n\n"
        "Evidence stats: RAG=%d, WEB=%d, KG=%d, Fused=%d\n\n"
        "Create 3-5 proposal items. ALL items share the SAME structure:\n"
        "- aspect: what aspect of the answer to address\n"
        "- question: specific question for this aspect\n"
        "- slots: list of info slots to EXTRACT from evidence\n"
        "  each slot: {name, description, evidence_hint}\n"
        "- priority: 1 (highest) to 5 (lowest)\n\n"
        "Output JSON array. Proposals are slot-filling tasks, NOT templates."
    ) % (
        query,
        evidence_preview,
        feedback_section,
        manifest.get("rag_count", 0),
        manifest.get("web_count", 0),
        manifest.get("kg_count", 0),
        manifest.get("fused_count", 0),
    )

    try:
        response = await llm_service.call(
            [
                SystemMessage(content=load_prompt("plan_phase_system")),
                HumanMessage(content=plan_prompt),
            ],
            prompt_name="plan_phase_system",
        )
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        plan_dag = json.loads(content)
        if not isinstance(plan_dag, list):
            plan_dag = [plan_dag]
    except Exception as e:
        logger.error("plan_phase_llm_failed", error=str(e))
        # Fallback: same-structured default plan
        plan_dag = [
            {
                "aspect": "core_answer",
                "question": query,
                "slots": [
                    {"name": "answer", "description": "Direct answer", "evidence_hint": "fused_evidence"},
                    {"name": "key_points", "description": "Key supporting points", "evidence_hint": "fused_evidence"},
                ],
                "priority": 1,
            },
            {
                "aspect": "supporting_details",
                "question": "What additional context supports the answer?",
                "slots": [
                    {"name": "details", "description": "Supporting evidence details", "evidence_hint": "document_chunk"},
                    {"name": "reasoning", "description": "Reasoning chain if available", "evidence_hint": "kg_match"},
                ],
                "priority": 2,
            },
            {
                "aspect": "summary",
                "question": "Brief conclusion?",
                "slots": [
                    {"name": "conclusion", "description": "Concise conclusion", "evidence_hint": "fused_evidence"},
                ],
                "priority": 3,
            },
        ]

    logger.info("plan_created", steps=len(plan_dag))

    return {
        "plan_dag": plan_dag,
        "current_phase": PipelinePhase.PLAN.value,
    }


# ── Rollback Phase ───────────────────────────────────────────────────────

async def rollback_phase(state: PipelineState) -> Dict[str, Any]:
    """Rollback phase: prepare state for re-exploration.

    Increments rollback counter, preserves feedback, and resets phase to Explore.
    The retrieval agent will see the validation_feedback and can adjust its strategy.
    """
    logger.info(
        "rollback_triggered",
        rollback_count=state.rollback_count + 1,
        feedback=state.validation_feedback[:100],
    )

    return {
        "current_phase": PipelinePhase.EXPLORE.value,
        "rollback_count": state.rollback_count + 1,
    }


# ── Assemble Output ──────────────────────────────────────────────────────

async def assemble_output(state: PipelineState) -> Dict[str, Any]:
    """Assemble the final output from all generated content nodes.

    Collects generated_content nodes, orders by proposal priority,
    and creates the final response message.
    """
    evidence_graph = state.evidence_graph

    # Collect generated content nodes
    generated = [
        n for n in evidence_graph.nodes.values()
        if n.evidence.content_type == "generated_content"
    ]

    if not generated:
        # Fallback: use fused evidence
        generated = [
            n for n in evidence_graph.nodes.values()
            if n.evidence.content_type == "fused_evidence"
        ]

    # Sort by priority (from metadata) then by confidence
    generated.sort(
        key=lambda n: (
            n.evidence.metadata.get("priority", 99),
            -n.belief.confidence,
        )
    )

    # If every generated section has no filled slots, return concise evidence-gap answer.
    if generated:
        all_empty = all(
            int(n.evidence.metadata.get("filled_count", 0)) <= 0
            for n in generated
        )
    else:
        all_empty = False

    if all_empty:
        is_zh = bool(re.search(r"[\u4e00-\u9fff]", state.query or ""))
        keywords = _extract_query_keywords(state.query or "")
        if is_zh:
            hint_topics = "、".join(keywords[:4]) if keywords else "当前问题主题"
            final_text = (
                "当前知识库中缺少与你问题直接相关的证据, 暂时无法给出可靠结论。\n\n"
                f"建议: 补充“{hint_topics}”相关资料后重试。"
            )
        else:
            hint_topics = ", ".join(keywords[:4]) if keywords else "the current topic"
            final_text = (
                "The current knowledge base lacks directly relevant evidence for your query, "
                "so a reliable answer cannot be produced yet.\n\n"
                f"Suggestion: add more evidence about {hint_topics} and try again."
            )
        sections = [final_text]
    else:
        # Assemble text from slot-filled sections
        sections = []
        for node in generated:
            content = node.evidence.content
            sections.append(content)
        final_text = "\n\n".join(sections) if sections else "No output generated."

    # Create response message
    response_message = AIMessage(content=final_text)

    logger.info(
        "output_assembled",
        sections=len(sections),
        total_length=len(final_text),
    )

    return {
        "messages": [response_message],
        "final_output": final_text,
        "current_phase": PipelinePhase.DONE.value,
    }


def _extract_query_keywords(query: str) -> List[str]:
    """Extract lightweight keywords from query for user-facing gap hint."""
    return query_keyword_extractor.extract_keywords_sync(
        query=query,
        max_keywords=settings.KEYWORD_EXTRACTION_MAX_TERMS,
    )


# ── Human Review ─────────────────────────────────────────────────────────

async def human_review(state: PipelineState) -> Dict[str, Any]:
    """Human-In-The-Loop review checkpoint.

    This node acts as a breakpoint where the pipeline pauses for human review.
    """
    logger.info(
        "human_review_checkpoint",
        iteration=state.iteration_count,
        evidence_nodes=state.evidence_graph.node_count,
    )

    return {
        "current_phase": PipelinePhase.DONE.value,
    }
