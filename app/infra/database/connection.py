"""PostgreSQL connection pool management.

Centralises connection pool creation so that both the LangGraph checkpointer
and application queries share the same pool.
"""

from __future__ import annotations

import traceback
from typing import Optional
from urllib.parse import quote_plus

from psycopg_pool import AsyncConnectionPool

from app.core.config import Environment, settings
from app.core.logging import logger


class ConnectionManager:
    """Manages a shared async PostgreSQL connection pool."""

    def __init__(self) -> None:
        self._pool: Optional[AsyncConnectionPool] = None

    @property
    def pool(self) -> Optional[AsyncConnectionPool]:
        return self._pool

    async def get_pool(self) -> AsyncConnectionPool:
        """Return the pool, creating it lazily if needed."""
        if self._pool is None:
            try:
                url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )
                self._pool = AsyncConnectionPool(
                    url,
                    open=False,
                    max_size=settings.POSTGRES_POOL_SIZE,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._pool.open()
                logger.info(
                    "pg_pool_created",
                    max_size=settings.POSTGRES_POOL_SIZE,
                    env=settings.ENVIRONMENT.value,
                )
            except Exception as e:
                logger.error(
                    "pg_pool_creation_failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_pg_pool")
                    return None  # type: ignore[return-value]
                raise
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("pg_pool_closed")


# Singleton
connection_manager = ConnectionManager()
