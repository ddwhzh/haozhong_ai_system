"""Langfuse observability integration.

Provides a shared Langfuse client and a LangChain callback handler factory.
If Langfuse keys are not configured, returns a no-op callback handler.
"""

from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import settings

logger = logging.getLogger(__name__)


class _NoOpCallbackHandler(BaseCallbackHandler):
    """No-op handler when Langfuse is not configured."""
    pass


def get_langfuse_client():
    """Return a configured Langfuse client, or None if not configured."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        logger.warning("Failed to initialize Langfuse client: %s", e)
        return None


def get_langfuse_callback(
    user_id: str | None = None,
    session_id: str | None = None,
) -> BaseCallbackHandler:
    """Create a LangChain CallbackHandler wired to Langfuse.

    Falls back to a no-op handler if Langfuse is not configured.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return _NoOpCallbackHandler()

    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler(
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        logger.warning("Failed to create Langfuse callback: %s", e)
        return _NoOpCallbackHandler()
