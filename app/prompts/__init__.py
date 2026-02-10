"""Prompt loader — reads prompt templates from markdown files.

Prompts are stored as .md files in this directory for easy editing.
The loader caches loaded prompts and supports variable substitution.

Usage:
    from app.prompts import load_prompt

    # Load and format a prompt
    text = load_prompt("query_decompose_system", max_sub_queries=5)

    # Load raw prompt without formatting
    text = load_prompt("slot_extraction_system")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.logging import logger

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=64)
def _read_prompt_file(name: str) -> str:
    """Read a prompt file from the prompts directory. Cached."""
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt(name: str, **kwargs: Any) -> str:
    """Load a prompt template by name, optionally formatting with kwargs.

    Args:
        name: Prompt file name (without .md extension).
        **kwargs: Variables to substitute in the template.

    Returns:
        Formatted prompt string.
    """
    template = _read_prompt_file(name)
    if kwargs:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                "prompt_format_key_missing",
                prompt=name,
                missing_key=str(e),
            )
            return template
    return template


def reload_prompts() -> None:
    """Clear the prompt cache, forcing re-read from disk."""
    _read_prompt_file.cache_clear()
    logger.info("prompt_cache_cleared")
