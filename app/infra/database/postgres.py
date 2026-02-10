"""High-level PostgreSQL database service.

Wraps connection_manager and exposes application-level helpers
(health check, query execution, etc.).
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.infra.database.connection import connection_manager


class DatabaseService:
    """Application-level database operations."""

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            pool = await connection_manager.get_pool()
            if pool is None:
                return False
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("db_health_check_failed", error=str(e))
            return False

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> None:
        """Execute a single statement."""
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            await conn.execute(query, params)

    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all rows as dicts."""
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = await cur.fetchall()
                return [dict(zip(columns, row)) for row in rows]

    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dict."""
        pool = await connection_manager.get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                row = await cur.fetchone()
                if row is None:
                    return None
                return dict(zip(columns, row))


# Singleton
database_service = DatabaseService()
