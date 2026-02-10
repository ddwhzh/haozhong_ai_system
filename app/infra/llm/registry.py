"""LLM model registry — pre-initialised model instances with metadata.

Centralises model configuration so that agents and services simply
request a model by name or role.

Supports OpenAI-compatible APIs (e.g. 智谱GLM) via OPENAI_API_BASE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Environment, settings
from app.core.logging import logger


def _build_llm(model: str, temperature: float | None = None, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI instance with optional custom base_url."""
    params: Dict[str, Any] = {
        "model": model,
        "temperature": temperature or settings.DEFAULT_LLM_TEMPERATURE,
        "api_key": settings.OPENAI_API_KEY,
        "max_tokens": settings.MAX_TOKENS,
    }
    # Support OpenAI-compatible APIs (智谱GLM, etc.)
    if settings.OPENAI_API_BASE:
        params["base_url"] = settings.OPENAI_API_BASE

    params.update(kwargs)
    return ChatOpenAI(**params)


class LLMRegistry:
    """Registry of available LLM models with pre-initialised instances."""

    LLMS: List[Dict[str, Any]] = [
        {
            "name": settings.DEFAULT_LLM_MODEL,
            "role": "default",
            "llm": _build_llm(
                settings.DEFAULT_LLM_MODEL,
                top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
            ),
        },
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:
        """Get an LLM by name with optional argument overrides."""
        for entry in cls.LLMS:
            if entry["name"] == model_name:
                if kwargs:
                    return _build_llm(model_name, **kwargs)
                return entry["llm"]

        # Model not pre-registered: build on-the-fly
        logger.info("llm_build_on_the_fly", model=model_name)
        return _build_llm(model_name, **kwargs)

    @classmethod
    def get_by_role(cls, role: str) -> BaseChatModel:
        """Get the first model matching a given role (default, fast, reasoning)."""
        for entry in cls.LLMS:
            if entry.get("role") == role:
                return entry["llm"]
        raise ValueError(f"No model with role '{role}' found")

    @classmethod
    def get_all_names(cls) -> List[str]:
        return [e["name"] for e in cls.LLMS]
