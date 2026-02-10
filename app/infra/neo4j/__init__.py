"""Neo4j knowledge graph infrastructure."""

from app.infra.neo4j.client import Neo4jClient, neo4j_client
from app.infra.neo4j.queries import KGQueries

__all__ = ["Neo4jClient", "neo4j_client", "KGQueries"]
