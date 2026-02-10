"""Agent pipeline API endpoints.

Presentation layer: handles HTTP requests, delegates to orchestration layer,
formats responses.  No business logic here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import logger
from app.infra.external.langfuse import get_langfuse_callback
from app.orchestration.graph import pipeline_graph
from app.presentation.schemas.request import ChatRequest, HumanReviewRequest
from app.presentation.schemas.response import (
    ChatResponse,
    EvidenceGraphResponse,
    IntermediateNode,
    IntermediateStep,
    PipelineStatusResponse,
)

router = APIRouter()
_presentation_pool = ThreadPoolExecutor(
    max_workers=max(1, settings.API_PRESENTATION_MAX_WORKERS),
    thread_name_prefix="presentation-api",
)


async def _run_in_presentation_pool(func, *args):
    """Run sync presentation helpers in a bounded thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_presentation_pool, partial(func, *args))


def _build_pipeline_metrics(agent_outputs) -> dict:
    """Build response metrics map from agent outputs."""
    metrics = {}
    for output in agent_outputs or []:
        metrics[output.agent_name] = output.metrics
    return metrics


def _build_evidence_summary(evidence_graph) -> dict:
    """Build evidence summary payload for response."""
    return evidence_graph.summary() if evidence_graph else {}


def _build_status_outputs(agent_outputs) -> list[dict]:
    """Serialize agent outputs for status endpoint."""
    return [
        {
            "agent": o.agent_name,
            "status": o.status.value,
            "summary": o.summary,
            "metrics": o.metrics,
        }
        for o in (agent_outputs or [])
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user query through the full multi-agent pipeline.

    Stages: Retrieval -> Generation -> Evaluation -> (HITL) -> Output
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    logger.info(
        "chat_request_received",
        query=request.query[:100],
        session_id=session_id,
        user_id=user_id,
    )

    try:
        graph = await pipeline_graph.create()

        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [get_langfuse_callback(user_id=user_id, session_id=session_id)],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
            },
        }

        # Invoke the pipeline (starts at Explore phase)
        result = await graph.ainvoke(
            input={
                "query": request.query,
                "messages": [{"role": "user", "content": request.query}],
                "current_phase": "explore",
                "max_iterations": request.max_iterations,
                "metadata": {
                    "user_id": user_id,
                    "session_id": session_id,
                    **request.metadata,
                },
            },
            config=config,
        )

        # Extract response
        final_output = result.get("final_output", "")
        evidence_graph = result.get("evidence_graph")
        agent_outputs = result.get("agent_outputs", [])

        # Presentation-layer threadpool: parallel response shaping.
        metrics, intermediate_steps, evidence_summary = await asyncio.gather(
            _run_in_presentation_pool(_build_pipeline_metrics, agent_outputs),
            _run_in_presentation_pool(_build_intermediate_steps, evidence_graph),
            _run_in_presentation_pool(_build_evidence_summary, evidence_graph),
        )

        return ChatResponse(
            session_id=session_id,
            response=final_output,
            evidence_summary=evidence_summary,
            pipeline_metrics=metrics,
            intermediate_steps=intermediate_steps,
            phase_completed=result.get("current_phase", "done"),
        )

    except Exception as e:
        logger.error("chat_request_failed", error=str(e), session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}",
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream the pipeline output as Server-Sent Events."""
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            graph = await pipeline_graph.create()

            config = {
                "configurable": {"thread_id": session_id},
                "callbacks": [get_langfuse_callback(user_id=user_id, session_id=session_id)],
                "metadata": {
                    "user_id": user_id,
                    "session_id": session_id,
                },
            }

            async for event in graph.astream_events(
                input={
                    "query": request.query,
                    "messages": [{"role": "user", "content": request.query}],
                    "max_iterations": request.max_iterations,
                    "metadata": {"user_id": user_id, "session_id": session_id},
                },
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chain_start":
                    yield f"data: {{\"event\": \"phase_start\", \"name\": \"{name}\"}}\n\n"
                elif kind == "on_chain_end":
                    yield f"data: {{\"event\": \"phase_end\", \"name\": \"{name}\"}}\n\n"
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", "")
                    if hasattr(chunk, "content") and chunk.content:
                        import json
                        yield f"data: {json.dumps({'event': 'token', 'content': chunk.content})}\n\n"

            yield "data: {\"event\": \"done\"}\n\n"

        except Exception as e:
            import json
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/review", response_model=ChatResponse)
async def human_review(request: HumanReviewRequest) -> ChatResponse:
    """Submit human review decision for HITL-paused pipeline."""
    session_id = request.session_id

    logger.info(
        "human_review_submitted",
        session_id=session_id,
        action=request.action,
    )

    try:
        graph = await pipeline_graph.create()

        config = {"configurable": {"thread_id": session_id}}

        # Resume the interrupted graph with human input
        if request.action == "approve":
            result = await graph.ainvoke(None, config=config)
        elif request.action == "reject":
            # Resume with feedback that triggers re-generation
            result = await graph.ainvoke(
                {"current_phase": "generation", "metadata": {"human_feedback": request.feedback}},
                config=config,
            )
        else:
            # Edit: apply edits and resume
            result = await graph.ainvoke(
                {"current_phase": "done"},
                config=config,
            )

        final_output = result.get("final_output", "")
        evidence_graph = result.get("evidence_graph")
        evidence_summary = await _run_in_presentation_pool(_build_evidence_summary, evidence_graph)

        return ChatResponse(
            session_id=session_id,
            response=final_output,
            evidence_summary=evidence_summary,
            pipeline_metrics={},
            phase_completed=result.get("current_phase", "done"),
        )

    except Exception as e:
        logger.error("human_review_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


def _build_intermediate_steps(evidence_graph) -> list:
    """Extract and group evidence graph nodes into display-friendly intermediate steps.

    Covers the full state machine pipeline:
      Explore: sub_query, document_chunk, web_result, kg_match, reasoning_chain, fused_evidence
      Plan/Execute: proposal (slot-filling), generated_content (slot-filled)
      Validate: evaluation_result, prompt_optimization
    """
    if not evidence_graph:
        return []

    def _normalize_metadata(value, limit: int = 2000):
        """Serialize metadata values for safe frontend display."""
        if isinstance(value, dict):
            normalized = {}
            for k, v in value.items():
                normalized[k] = _normalize_metadata(v, limit=limit)
            return normalized
        if isinstance(value, list):
            return [_normalize_metadata(v, limit=limit) for v in value[:100]]
        if isinstance(value, str):
            return value[:limit]
        return value

    # Build lightweight provenance index for node inspection.
    parent_ids_map: dict[str, list[str]] = {}
    child_ids_map: dict[str, list[str]] = {}
    for edge in evidence_graph.edges:
        parent_ids_map.setdefault(edge.target_node_id, []).append(edge.source_node_id)
        child_ids_map.setdefault(edge.source_node_id, []).append(edge.target_node_id)

    # Define preferred content type groupings with display labels.
    # Ordered by pipeline phase: Input -> Classify -> Explore -> Plan/Execute -> Validate
    groups = [
        ("retrieval_agent", "query", "0. Input: 原始查询"),
        ("retrieval_agent", "query_classification", "1. Classify: QDMR查询分类"),
        ("retrieval_agent", "tree_step", "2. Explore: QDMR分解树步骤"),
        ("retrieval_agent", "sub_query", "3. Explore: 查询分解(子查询)"),
        ("retrieval_agent", "document_chunk", "4. Explore: RAG向量检索结果"),
        ("retrieval_agent", "web_result", "5. Explore: Brave Web检索结果"),
        ("retrieval_agent", "kg_match", "6. Explore: KG知识图谱匹配"),
        ("retrieval_agent", "reasoning_chain", "7. Explore: KG推理链"),
        ("retrieval_agent", "fused_evidence", "8. Explore: Cascade融合结果"),
        # -- Plan/Execute phase (Generation Agent) --
        ("generation_agent", "proposal", "9. Plan: Proposal槽位定义"),
        ("generation_agent", "generated_content", "10. Execute: 证据图填空结果"),
        # -- Validate phase (Evaluation Agent) --
        ("evaluation_agent", "evaluation_result", "11. Validate: 匈牙利匹配评估"),
        ("evaluation_agent", "prompt_optimization", "12. Validate: Prompt优化建议"),
    ]

    steps = []
    grouped_ids: set[str] = set()

    def _serialize_node(n) -> IntermediateNode:
        # Filter out huge metadata payloads while keeping most inspection details.
        safe_metadata = {}
        for k, v in n.evidence.metadata.items():
            if "embedding" in k.lower():
                continue
            safe_metadata[k] = _normalize_metadata(v)

        return IntermediateNode(
            node_id=n.node_id,
            content=n.evidence.content[:1600],
            content_type=n.evidence.content_type,
            source=n.evidence.source,
            source_agent=n.belief.source_agent,
            created_at=n.created_at.isoformat(),
            confidence=n.belief.confidence,
            belief_content=n.belief.content[:500],
            intent=n.intent.description,
            intent_type=n.intent.intent_type.value,
            tags=n.tags,
            parent_node_ids=parent_ids_map.get(n.node_id, []),
            child_node_ids=child_ids_map.get(n.node_id, []),
            metadata=safe_metadata,
        )

    for agent, content_type, label in groups:
        nodes_data = []
        for n in evidence_graph.nodes.values():
            if n.belief.source_agent == agent and n.evidence.content_type == content_type:
                nodes_data.append(_serialize_node(n))
                grouped_ids.add(n.node_id)
        if nodes_data:
            steps.append(IntermediateStep(
                agent=agent,
                label=label,
                nodes=nodes_data,
            ))

    # Add ungrouped artifacts so users can inspect everything.
    # Group by (agent, content_type) for readability.
    other_groups: dict[tuple[str, str], list] = {}
    for n in evidence_graph.nodes.values():
        if n.node_id in grouped_ids:
            continue
        key = (n.belief.source_agent, n.evidence.content_type)
        other_groups.setdefault(key, []).append(n)

    if other_groups:
        for idx, ((agent, content_type), nodes) in enumerate(
            sorted(other_groups.items(), key=lambda x: (x[0][0], x[0][1])), start=1
        ):
            steps.append(IntermediateStep(
                agent=agent,
                label=f"12+{idx}. Other: {agent}/{content_type}",
                nodes=[_serialize_node(n) for n in nodes],
            ))

    return steps


@router.get("/status/{session_id}", response_model=PipelineStatusResponse)
async def pipeline_status(session_id: str) -> PipelineStatusResponse:
    """Get the current status of a pipeline session."""
    try:
        graph = await pipeline_graph.create()
        config = {"configurable": {"thread_id": session_id}}

        state = await graph.aget_state(config)

        if not state.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        values = state.values
        evidence_graph = values.get("evidence_graph")
        agent_outputs = values.get("agent_outputs", [])
        evidence_summary, status_outputs = await asyncio.gather(
            _run_in_presentation_pool(_build_evidence_summary, evidence_graph),
            _run_in_presentation_pool(_build_status_outputs, agent_outputs),
        )

        return PipelineStatusResponse(
            session_id=session_id,
            current_phase=values.get("current_phase", "unknown"),
            iteration_count=values.get("iteration_count", 0),
            is_waiting_for_human=values.get("current_phase") == "human_review",
            evidence_summary=evidence_summary,
            agent_outputs=status_outputs,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pipeline_status_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
