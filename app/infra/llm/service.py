"""LLM service with retry, circular fallback, and tool binding.

Adapted from the template's LLMService, restructured into infra layer.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import logger
from app.infra.llm.registry import LLMRegistry


class LLMService:
    """Manages LLM calls with automatic retry and circular model fallback."""

    def __init__(self) -> None:
        self._llm: Optional[BaseChatModel] = None
        self._current_model_index: int = 0

        all_names = LLMRegistry.get_all_names()
        try:
            self._current_model_index = all_names.index(settings.DEFAULT_LLM_MODEL)
            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
            logger.info(
                "llm_service_initialized",
                model=settings.DEFAULT_LLM_MODEL,
                total_models=len(all_names),
            )
        except (ValueError, Exception) as e:
            self._current_model_index = 0
            self._llm = LLMRegistry.LLMS[0]["llm"]
            logger.warning(
                "default_model_fallback",
                requested=settings.DEFAULT_LLM_MODEL,
                using=all_names[0] if all_names else "none",
                error=str(e),
            )

    def get_llm(self) -> Optional[BaseChatModel]:
        return self._llm

    def bind_tools(self, tools: List) -> "LLMService":
        if self._llm:
            self._llm = self._llm.bind_tools(tools)
        return self

    # ── Retry + fallback ─────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _call_with_retry(self, messages: List[BaseMessage]) -> BaseMessage:
        if not self._llm:
            raise RuntimeError("LLM not initialised")
        try:
            return await self._llm.ainvoke(messages)
        except (RateLimitError, APITimeoutError, APIError):
            raise
        except OpenAIError as e:
            logger.error("llm_call_failed", error=str(e))
            raise

    def _switch_model(self) -> bool:
        """Circular switch to the next model in the registry."""
        try:
            total = len(LLMRegistry.LLMS)
            next_idx = (self._current_model_index + 1) % total
            entry = LLMRegistry.LLMS[next_idx]
            self._current_model_index = next_idx
            self._llm = entry["llm"]
            logger.warning("model_switched", new_model=entry["name"])
            return True
        except Exception as e:
            logger.error("model_switch_failed", error=str(e))
            return False

    async def call(
        self,
        messages: List[BaseMessage],
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> BaseMessage:
        """Call the LLM with circular fallback across all registered models.

        Args:
            messages: LangChain message list.
            model_name: Override model (optional).
            prompt_name: Name of the prompt file used (for tracing/logging).
            **kwargs: Extra model parameters.
        """
        if model_name:
            self._llm = LLMRegistry.get(model_name, **kwargs)

        # Track which prompt triggered this call
        logger.info(
            "llm_call_start",
            prompt_name=prompt_name or "inline",
            model=getattr(self._llm, "model_name", "unknown"),
            message_count=len(messages),
        )

        total = len(LLMRegistry.LLMS)
        tried = 0
        last_err = None

        while tried < total:
            try:
                result = await self._call_with_retry(messages)
                logger.info(
                    "llm_call_success",
                    prompt_name=prompt_name or "inline",
                    model=getattr(self._llm, "model_name", "unknown"),
                    response_length=len(result.content) if hasattr(result, "content") else 0,
                )
                return result
            except OpenAIError as e:
                last_err = e
                tried += 1
                logger.warning(
                    "llm_call_retry",
                    prompt_name=prompt_name or "inline",
                    tried=tried,
                    error=str(e),
                )
                if tried >= total:
                    break
                self._switch_model()

        raise RuntimeError(
            f"All {tried} models failed. Last error: {last_err}"
        )


# Singleton
llm_service = LLMService()
