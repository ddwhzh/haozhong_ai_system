"""API v1 router aggregation."""

from fastapi import APIRouter

from app.presentation.api.v1.agent import router as agent_router
from app.presentation.api.v1.research import router as research_router

api_router = APIRouter()
api_router.include_router(agent_router, prefix="/agent", tags=["Agent Pipeline"])
api_router.include_router(research_router, prefix="/research", tags=["Research Pipeline"])
