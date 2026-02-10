FROM python:3.13.2-slim

WORKDIR /app

ARG APP_ENV=development

ENV APP_ENV=${APP_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && pip install --upgrade pip \
    && pip install uv \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user BEFORE copying files (avoids expensive chown -R)
RUN useradd -m -d /home/appuser appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app

# Create venv and install deps as root (for build-essential access)
RUN uv venv /app/.venv

COPY pyproject.toml .
RUN uv pip install --python /app/.venv/bin/python .

# Copy application source
COPY --chown=appuser:appuser . .

# Ensure entrypoint is executable
RUN chmod +x /app/scripts/docker-entrypoint.sh

# Switch to non-root
USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
