"""Research pipeline API endpoints.

Presentation layer for the automated research loop.
No business logic here — delegates to the research LangGraph.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_state import ResearchPhase, ResearchState
from app.infra.external.langfuse import get_langfuse_callback
from app.orchestration.research_graph import research_graph

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────

class ResearchStartRequest(BaseModel):
    """Request body to start a research session."""

    topic: str = Field(..., min_length=3, description="Research topic or question")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Max research iterations")
    session_id: Optional[str] = Field(default=None, description="Optional session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchFeedbackRequest(BaseModel):
    """Request body for injecting human feedback."""

    feedback: str = Field(..., min_length=1, description="Human feedback text")
    action: str = Field(
        default="adjust",
        description="Feedback action: 'adjust' direction, 'stop' research",
        pattern="^(adjust|stop)$",
    )


class ResearchStatusResponse(BaseModel):
    """Response for research status queries."""

    session_id: str
    research_topic: str
    current_phase: str
    iteration: int
    max_iterations: int
    paper_count: int
    hypothesis_count: int
    experiment_count: int
    journal_entries: int
    final_report: str = ""


class ResearchJournalResponse(BaseModel):
    """Response for research journal export."""

    session_id: str
    research_topic: str
    entries: List[Dict[str, Any]]


class ResearchResponse(BaseModel):
    """Response for completed research."""

    session_id: str
    research_topic: str
    final_report: str
    iterations_completed: int
    papers_reviewed: int
    hypotheses: List[Dict[str, Any]]
    experiments: List[Dict[str, Any]]


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/start", response_model=ResearchResponse)
async def start_research(request: ResearchStartRequest) -> ResearchResponse:
    """Start a research session and run until completion.

    Runs the full research loop synchronously (may take several minutes).
    For long-running research, use the /stream endpoint instead.
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "research_api_start",
        topic=request.topic[:100],
        session_id=session_id,
        max_iterations=request.max_iterations,
    )

    graph = await research_graph.create()

    initial_state = {
        "research_topic": request.topic,
        "max_iterations": request.max_iterations,
        "current_phase": ResearchPhase.SCOUT.value,
        "iteration_count": 1,
        "metadata": {
            "session_id": session_id,
            **request.metadata,
        },
    }

    config = {"configurable": {"thread_id": session_id}}

    callback = get_langfuse_callback(
        session_id=session_id,
        user_id=request.metadata.get("user_id", "anonymous"),
    )
    if callback:
        config["callbacks"] = [callback]

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        logger.exception(
            "research_api_pipeline_failed",
            session_id=session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research pipeline failed: {exc}",
        )

    return _build_response(session_id, final_state)


@router.post("/stream")
async def stream_research(request: ResearchStartRequest) -> StreamingResponse:
    """Start a research session with SSE streaming of progress.

    Streams phase transitions and intermediate results as Server-Sent Events.
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "research_api_stream_start",
        topic=request.topic[:100],
        session_id=session_id,
    )

    graph = await research_graph.create()

    initial_state = {
        "research_topic": request.topic,
        "max_iterations": request.max_iterations,
        "current_phase": ResearchPhase.SCOUT.value,
        "iteration_count": 1,
        "metadata": {
            "session_id": session_id,
            **request.metadata,
        },
    }

    config = {"configurable": {"thread_id": session_id}}

    callback = get_langfuse_callback(
        session_id=session_id,
        user_id=request.metadata.get("user_id", "anonymous"),
    )
    if callback:
        config["callbacks"] = [callback]

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    phase = node_output.get("current_phase", "")
                    journal = node_output.get("research_journal", [])

                    event_data = {
                        "node": node_name,
                        "phase": phase,
                        "iteration": node_output.get("iteration_count", 0),
                    }

                    if journal:
                        latest = journal[-1] if isinstance(journal, list) else journal
                        if hasattr(latest, "summary"):
                            event_data["summary"] = latest.summary

                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'event': 'complete', 'session_id': session_id})}\n\n"

        except Exception as exc:
            logger.exception(
                "research_api_stream_error",
                session_id=session_id,
                error=str(exc),
            )
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Research-Session-ID": session_id,
        },
    )


@router.get("/{session_id}/status", response_model=ResearchStatusResponse)
async def get_research_status(session_id: str) -> ResearchStatusResponse:
    """Get the current status of a research session."""
    graph = await research_graph.create()

    config = {"configurable": {"thread_id": session_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as exc:
        logger.error(
            "research_api_status_failed",
            session_id=session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    if state is None or state.values is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    sv = state.values
    return ResearchStatusResponse(
        session_id=session_id,
        research_topic=sv.get("research_topic", ""),
        current_phase=sv.get("current_phase", ""),
        iteration=sv.get("iteration_count", 0),
        max_iterations=sv.get("max_iterations", 5),
        paper_count=len(sv.get("literature", [])),
        hypothesis_count=len(sv.get("hypotheses", [])),
        experiment_count=len(sv.get("experiments", [])),
        journal_entries=len(sv.get("research_journal", [])),
        final_report=sv.get("final_report", ""),
    )


@router.get("/{session_id}/journal", response_model=ResearchJournalResponse)
async def get_research_journal(session_id: str) -> ResearchJournalResponse:
    """Export the research journal for a session."""
    graph = await research_graph.create()

    config = {"configurable": {"thread_id": session_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    if state is None or state.values is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    sv = state.values
    entries = []
    for je in sv.get("research_journal", []):
        if hasattr(je, "model_dump"):
            entries.append(je.model_dump(mode="json"))
        elif isinstance(je, dict):
            entries.append(je)

    return ResearchJournalResponse(
        session_id=session_id,
        research_topic=sv.get("research_topic", ""),
        entries=entries,
    )


@router.post("/{session_id}/feedback")
async def submit_feedback(
    session_id: str, request: ResearchFeedbackRequest
) -> Dict[str, Any]:
    """Inject human feedback into a running research session."""
    logger.info(
        "research_api_feedback",
        session_id=session_id,
        action=request.action,
        feedback=request.feedback[:100],
    )

    graph = await research_graph.create()
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = await graph.aget_state(config)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    if state is None or state.values is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session not found: {session_id}",
        )

    if request.action == "stop":
        await graph.aupdate_state(
            config,
            {"current_phase": ResearchPhase.CONCLUDE.value},
        )
        return {"status": "stopping", "session_id": session_id}

    sv = state.values
    ctx = dict(sv.get("domain_context", {}))
    existing = ctx.get("refined_queries", [])
    ctx["refined_queries"] = [request.feedback] + existing[:4]

    await graph.aupdate_state(config, {"domain_context": ctx})

    return {"status": "feedback_applied", "session_id": session_id}


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_response(session_id: str, state: Any) -> ResearchResponse:
    """Build the final response from completed research state."""
    if isinstance(state, dict):
        sv = state
    elif hasattr(state, "values"):
        sv = state.values
    else:
        sv = state

    hypotheses = []
    for h in sv.get("hypotheses", []):
        if hasattr(h, "model_dump"):
            hypotheses.append(h.model_dump(mode="json"))
        elif isinstance(h, dict):
            hypotheses.append(h)

    experiments = []
    for e in sv.get("experiments", []):
        if hasattr(e, "model_dump"):
            experiments.append(e.model_dump(mode="json"))
        elif isinstance(e, dict):
            experiments.append(e)

    return ResearchResponse(
        session_id=session_id,
        research_topic=sv.get("research_topic", ""),
        final_report=sv.get("final_report", ""),
        iterations_completed=sv.get("iteration_count", 0),
        papers_reviewed=len(sv.get("literature", [])),
        hypotheses=hypotheses,
        experiments=experiments,
    )
