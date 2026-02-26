"""Application configuration management.

This module handles environment-specific configuration loading, parsing, and management
for the application. It includes environment detection, .env file loading, and
configuration value parsing.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from dotenv import load_dotenv


# Define environment types
class Environment(str, Enum):
    """Application environment types.

    Defines the possible environments the application can run in:
    development, staging, production, and test.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Determine environment
def get_environment() -> Environment:
    """Get the current environment.

    Returns:
        Environment: The current environment (development, staging, production, or test)
    """
    match os.getenv("APP_ENV", "development").lower():
        case "production" | "prod":
            return Environment.PRODUCTION
        case "staging" | "stage":
            return Environment.STAGING
        case "test":
            return Environment.TEST
        case _:
            return Environment.DEVELOPMENT


# Load appropriate .env file based on environment
def load_env_file():
    """Load environment-specific .env file."""
    env = get_environment()
    print(f"Loading environment: {env}")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Define env files in priority order
    env_files = [
        os.path.join(base_dir, f".env.{env.value}.local"),
        os.path.join(base_dir, f".env.{env.value}"),
        os.path.join(base_dir, ".env.local"),
        os.path.join(base_dir, ".env"),
    ]

    # Load the first env file that exists
    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(dotenv_path=env_file)
            print(f"Loaded environment from {env_file}")
            return env_file

    # Fall back to default if no env file found
    return None


ENV_FILE = load_env_file()


# Parse list values from environment variables
def parse_list_from_env(env_key, default=None):
    """Parse a comma-separated list from an environment variable."""
    value = os.getenv(env_key)
    if not value:
        return default or []

    # Remove quotes if they exist
    value = value.strip("\"'")
    # Handle single value case
    if "," not in value:
        return [value]
    # Split comma-separated values
    return [item.strip() for item in value.split(",") if item.strip()]


# Parse dict of lists from environment variables with prefix
def parse_dict_of_lists_from_env(prefix, default_dict=None):
    """Parse dictionary of lists from environment variables with a common prefix."""
    result = default_dict or {}

    # Look for all env vars with the given prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            endpoint = key[len(prefix) :].lower()  # Extract endpoint name
            # Parse the values for this endpoint
            if value:
                value = value.strip("\"'")
                if "," in value:
                    result[endpoint] = [item.strip() for item in value.split(",") if item.strip()]
                else:
                    result[endpoint] = [value]

    return result


class Settings:
    """Application settings without using pydantic."""

    def __init__(self):
        """Initialize application settings from environment variables.

        Loads and sets all configuration values from environment variables,
        with appropriate defaults for each setting. Also applies
        environment-specific overrides based on the current environment.
        """
        # Set the environment
        self.ENVIRONMENT = get_environment()

        # Application Settings
        self.PROJECT_NAME = os.getenv("PROJECT_NAME", "haozhong Multi-Agent System")
        self.VERSION = os.getenv("VERSION", "1.0.0")
        self.DESCRIPTION = os.getenv(
            "DESCRIPTION",
            "Multi-Agent system with managed pyramid architecture: Retrieval, Generation, Evaluation agents",
        )
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "t", "yes")

        # CORS Settings
        self.ALLOWED_ORIGINS = parse_list_from_env("ALLOWED_ORIGINS", ["*"])

        # Langfuse Configuration
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # LLM Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
        self.DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "glm-4-flash")
        self.DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.2"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        self.MAX_LLM_CALL_RETRIES = int(os.getenv("MAX_LLM_CALL_RETRIES", "3"))

        # Long term memory Configuration
        self.LONG_TERM_MEMORY_MODEL = os.getenv("LONG_TERM_MEMORY_MODEL", "gpt-5-nano")
        self.LONG_TERM_MEMORY_EMBEDDER_MODEL = os.getenv("LONG_TERM_MEMORY_EMBEDDER_MODEL", "text-embedding-3-small")
        self.LONG_TERM_MEMORY_COLLECTION_NAME = os.getenv("LONG_TERM_MEMORY_COLLECTION_NAME", "longterm_memory")
        # JWT Configuration
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_DAYS", "30"))

        # Logging Configuration
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "json")  # "json" or "console"
        self.API_PRESENTATION_MAX_WORKERS = int(os.getenv("API_PRESENTATION_MAX_WORKERS", "4"))

        # Postgres Configuration
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "food_order_db")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
        self.POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.CHECKPOINT_TABLES = ["checkpoint_blobs", "checkpoint_writes", "checkpoints"]

        # Rate Limiting Configuration
        self.RATE_LIMIT_DEFAULT = parse_list_from_env("RATE_LIMIT_DEFAULT", ["200 per day", "50 per hour"])

        # Rate limit endpoints defaults
        default_endpoints = {
            "chat": ["30 per minute"],
            "chat_stream": ["20 per minute"],
            "messages": ["50 per minute"],
            "register": ["10 per hour"],
            "login": ["20 per minute"],
            "root": ["10 per minute"],
            "health": ["20 per minute"],
        }

        # Update rate limit endpoints from environment variables
        self.RATE_LIMIT_ENDPOINTS = default_endpoints.copy()
        for endpoint in default_endpoints:
            env_key = f"RATE_LIMIT_{endpoint.upper()}"
            value = parse_list_from_env(env_key)
            if value:
                self.RATE_LIMIT_ENDPOINTS[endpoint] = value

        # Neo4j Configuration
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
        self.NEO4J_POOL_SIZE = int(os.getenv("NEO4J_POOL_SIZE", "50"))
        self.NEO4J_CONNECT_MAX_RETRIES = int(os.getenv("NEO4J_CONNECT_MAX_RETRIES", "3"))
        self.NEO4J_RETRY_COOLDOWN_SECONDS = float(
            os.getenv("NEO4J_RETRY_COOLDOWN_SECONDS", "20")
        )

        # Vector Store Configuration
        self.VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "documents")
        self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # Retrieval Agent Configuration
        self.RETRIEVAL_MAX_SUB_QUERIES = int(os.getenv("RETRIEVAL_MAX_SUB_QUERIES", "5"))
        self.RETRIEVAL_KG_MAX_HOPS = int(os.getenv("RETRIEVAL_KG_MAX_HOPS", "3"))
        self.RETRIEVAL_CASCADE_ROUNDS = int(os.getenv("RETRIEVAL_CASCADE_ROUNDS", "3"))
        self.RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
        self.RETRIEVAL_VECTOR_MODE = os.getenv("RETRIEVAL_VECTOR_MODE", "dense").lower()
        self.RETRIEVAL_HYBRID_DENSE_WEIGHT = float(os.getenv("RETRIEVAL_HYBRID_DENSE_WEIGHT", "0.6"))
        self.RETRIEVAL_HYBRID_SPARSE_WEIGHT = float(os.getenv("RETRIEVAL_HYBRID_SPARSE_WEIGHT", "0.4"))
        self.RETRIEVAL_HYBRID_RRF_K = int(os.getenv("RETRIEVAL_HYBRID_RRF_K", "60"))
        self.RETRIEVAL_HYBRID_CANDIDATES = int(os.getenv("RETRIEVAL_HYBRID_CANDIDATES", "30"))
        self.RETRIEVAL_HYBRID_MAX_TERMS = int(os.getenv("RETRIEVAL_HYBRID_MAX_TERMS", "8"))
        self.RETRIEVAL_BACKTRACK_THRESHOLD = float(os.getenv("RETRIEVAL_BACKTRACK_THRESHOLD", "0.4"))
        self.RETRIEVAL_MAX_BRANCH_RETRIES = int(os.getenv("RETRIEVAL_MAX_BRANCH_RETRIES", "1"))
        self.RETRIEVAL_MAX_TREE_DEPTH = int(os.getenv("RETRIEVAL_MAX_TREE_DEPTH", "3"))
        self.BRAVE_SEARCH_ENABLED = os.getenv("BRAVE_SEARCH_ENABLED", "false").lower() in ("true", "1", "t", "yes")
        self.BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
        self.BRAVE_SEARCH_BASE_URL = os.getenv(
            "BRAVE_SEARCH_BASE_URL",
            "https://api.search.brave.com/res/v1/web/search",
        )
        self.BRAVE_SEARCH_TOP_K = int(os.getenv("BRAVE_SEARCH_TOP_K", "5"))
        self.BRAVE_SEARCH_TIMEOUT_SECONDS = float(os.getenv("BRAVE_SEARCH_TIMEOUT_SECONDS", "8"))
        self.BRAVE_AUTO_INDEX_TO_VECTOR = os.getenv("BRAVE_AUTO_INDEX_TO_VECTOR", "false").lower() in ("true", "1", "t", "yes")
        self.BRAVE_AUTO_INDEX_MAX_ITEMS = int(os.getenv("BRAVE_AUTO_INDEX_MAX_ITEMS", "3"))
        self.BRAVE_AUTO_INDEX_MIN_CONTENT_CHARS = int(os.getenv("BRAVE_AUTO_INDEX_MIN_CONTENT_CHARS", "30"))
        self.KEYWORD_EXTRACTION_USE_LLM = os.getenv("KEYWORD_EXTRACTION_USE_LLM", "false").lower() in ("true", "1", "t", "yes")
        self.KEYWORD_EXTRACTION_MAX_TERMS = int(os.getenv("KEYWORD_EXTRACTION_MAX_TERMS", "10"))

        # Generation Agent Configuration
        self.GENERATION_MAX_PROPOSALS = int(os.getenv("GENERATION_MAX_PROPOSALS", "5"))
        self.GENERATION_PROPOSAL_TEMPERATURE = float(os.getenv("GENERATION_PROPOSAL_TEMPERATURE", "0.3"))

        # Evaluation Agent Configuration
        self.EVALUATION_LLM = os.getenv("EVALUATION_LLM", "gpt-4o")
        self.EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY", self.OPENAI_API_KEY)
        self.EVALUATION_CONFIDENCE_THRESHOLD = float(os.getenv("EVALUATION_CONFIDENCE_THRESHOLD", "0.7"))
        self.EVALUATION_SPARSE_WEIGHT = float(os.getenv("EVALUATION_SPARSE_WEIGHT", "0.3"))
        self.EVALUATION_QUALITY_THRESHOLD = float(os.getenv("EVALUATION_QUALITY_THRESHOLD", "0.4"))

        # Pipeline Configuration
        self.PIPELINE_MAX_ITERATIONS = int(os.getenv("PIPELINE_MAX_ITERATIONS", "10"))
        self.HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() in ("true", "1", "t", "yes")

        # Research Agent Configuration
        self.RESEARCH_MAX_ITERATIONS = int(os.getenv("RESEARCH_MAX_ITERATIONS", "5"))
        self.RESEARCH_MAX_PAPERS_PER_SOURCE = int(os.getenv("RESEARCH_MAX_PAPERS_PER_SOURCE", "20"))
        self.RESEARCH_ARXIV_TIMEOUT = float(os.getenv("RESEARCH_ARXIV_TIMEOUT", "15"))
        self.RESEARCH_S2_API_KEY = os.getenv("RESEARCH_S2_API_KEY", "")
        self.RESEARCH_S2_TIMEOUT = float(os.getenv("RESEARCH_S2_TIMEOUT", "15"))
        self.RESEARCH_SANDBOX_TIMEOUT = int(os.getenv("RESEARCH_SANDBOX_TIMEOUT", "300"))
        self.RESEARCH_SANDBOX_MEMORY = os.getenv("RESEARCH_SANDBOX_MEMORY", "2g")
        self.RESEARCH_SANDBOX_CPU = int(os.getenv("RESEARCH_SANDBOX_CPU", "1"))
        self.RESEARCH_SANDBOX_IMAGE = os.getenv("RESEARCH_SANDBOX_IMAGE", "python:3.11-slim")

        # Apply environment-specific settings
        self.apply_environment_settings()

    def apply_environment_settings(self):
        """Apply environment-specific settings based on the current environment."""
        env_settings = {
            Environment.DEVELOPMENT: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "200 per hour"],
            },
            Environment.STAGING: {
                "DEBUG": False,
                "LOG_LEVEL": "INFO",
                "RATE_LIMIT_DEFAULT": ["500 per day", "100 per hour"],
            },
            Environment.PRODUCTION: {
                "DEBUG": False,
                "LOG_LEVEL": "WARNING",
                "RATE_LIMIT_DEFAULT": ["200 per day", "50 per hour"],
            },
            Environment.TEST: {
                "DEBUG": True,
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "console",
                "RATE_LIMIT_DEFAULT": ["1000 per day", "1000 per hour"],  # Relaxed for testing
            },
        }

        # Get settings for current environment
        current_env_settings = env_settings.get(self.ENVIRONMENT, {})

        # Apply settings if not explicitly set in environment variables
        for key, value in current_env_settings.items():
            env_var_name = key.upper()
            # Only override if environment variable wasn't explicitly set
            if env_var_name not in os.environ:
                setattr(self, key, value)


# Create settings instance
settings = Settings()
