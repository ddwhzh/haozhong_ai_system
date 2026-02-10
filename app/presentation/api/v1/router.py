"""API v1 router aggregation."""

from fastapi import APIRouter

from app.presentation.api.v1.agent import router as agent_router

api_router = APIRouter()
api_router.include_router(agent_router, prefix="/agent", tags=["Agent Pipeline"])
