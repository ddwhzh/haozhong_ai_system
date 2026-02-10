"""haozhong Multi-Agent System — Application entry point.

Managed Pyramid Architecture:
  Presentation Layer -> Orchestration Layer -> Business Worker Layer -> Infra Layer
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.core.metrics import setup_metrics
from app.core.middleware import LoggingContextMiddleware, MetricsMiddleware
from app.infra.database import database_service
from app.infra.neo4j import neo4j_client
from app.presentation.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info(
        "application_startup",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT.value,
    )
    # Initialize Neo4j connection
    try:
        await neo4j_client.connect()
    except Exception as e:
        logger.warning("neo4j_startup_connection_failed", error=str(e))

    yield

    # Cleanup
    await neo4j_client.close()
    from app.infra.database.connection import connection_manager
    await connection_manager.close()
    logger.info("application_shutdown")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Prometheus metrics
setup_metrics(app)

# Middleware
app.add_middleware(LoggingContextMiddleware)
app.add_middleware(MetricsMiddleware)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.error(
        "validation_error",
        path=request.url.path,
        errors=str(exc.errors()),
    )
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join(
            [str(p) for p in error["loc"] if p != "body"]
        )
        formatted_errors.append({"field": loc, "message": error["msg"]})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": formatted_errors},
    )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Chat UI — serve static HTML
_static_dir = Path(__file__).parent / "presentation" / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/chat")
async def chat_ui():
    """Serve the chat frontend."""
    from fastapi.responses import FileResponse
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/")
async def root(request: Request):
    """Root endpoint."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "healthy",
        "environment": settings.ENVIRONMENT.value,
        "architecture": "Managed Pyramid (Presentation -> Orchestration -> Worker -> Infra)",
        "agents": ["retrieval_agent", "generation_agent", "evaluation_agent"],
        "docs": "/docs",
    }


@app.get("/health")
async def health_check(request: Request):
    """Health check with component status."""
    db_healthy = await database_service.health_check()
    neo4j_healthy = await neo4j_client.health_check()

    all_healthy = db_healthy and neo4j_healthy
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if all_healthy else "degraded",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT.value,
            "components": {
                "api": "healthy",
                "database": "healthy" if db_healthy else "unhealthy",
                "neo4j": "healthy" if neo4j_healthy else "unhealthy",
            },
            "timestamp": datetime.now().isoformat(),
        },
    )
