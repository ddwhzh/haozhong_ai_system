"""Shared test fixtures."""

import sys
import types

import pytest


@pytest.fixture
def sample_query() -> str:
    """Standard test query."""
    return "什么是知识图谱? 它与传统数据库有什么区别?"


@pytest.fixture
def sample_embedding() -> list:
    """1536-dimensional dummy embedding."""
    import random
    random.seed(42)
    return [random.gauss(0, 1) for _ in range(1536)]


@pytest.fixture
def sample_tokens() -> list:
    """Sample tokenized text."""
    return ["知识", "图谱", "是", "一种", "语义", "网络", "数据", "结构"]


@pytest.fixture(autouse=True)
def stub_optional_jose_dependency(monkeypatch):
    """Provide a lightweight jose stub when python-jose is absent.

    Some smoke tests import the FastAPI app, which imports middleware using:
      from jose import JWTError, jwt
    """
    try:
        __import__("jose")
    except ModuleNotFoundError:
        fake_jose = types.ModuleType("jose")

        class FakeJWTError(Exception):
            """Fallback JWTError type for tests."""

        class FakeJWT:
            """Fallback jwt API surface for tests."""

            @staticmethod
            def decode(*args, **kwargs):  # noqa: ANN002, ANN003
                return {}

            @staticmethod
            def encode(*args, **kwargs):  # noqa: ANN002, ANN003
                return "test-token"

        fake_jose.JWTError = FakeJWTError
        fake_jose.jwt = FakeJWT
        monkeypatch.setitem(sys.modules, "jose", fake_jose)


@pytest.fixture(autouse=True)
def stub_optional_langgraph_postgres_dependency(monkeypatch):
    """Provide langgraph postgres checkpointer stub when extra is absent."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401
    except ModuleNotFoundError:
        if "langgraph.checkpoint" not in sys.modules:
            checkpoint_mod = types.ModuleType("langgraph.checkpoint")
            monkeypatch.setitem(sys.modules, "langgraph.checkpoint", checkpoint_mod)

        postgres_mod = types.ModuleType("langgraph.checkpoint.postgres")
        aio_mod = types.ModuleType("langgraph.checkpoint.postgres.aio")

        class FakeAsyncPostgresSaver:
            """Fallback AsyncPostgresSaver used only in tests."""

            def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                pass

            async def setup(self):
                return None

        aio_mod.AsyncPostgresSaver = FakeAsyncPostgresSaver
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", postgres_mod)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", aio_mod)
