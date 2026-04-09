"""
Graph backend factory.

Usage:
    from src.graph import get_graph_backend
    graph = get_graph_backend()   # NetworkX by default

Override via env var:
    GRAPH_BACKEND=neo4j  →  Neo4jBackend
    GRAPH_BACKEND=networkx (default)

The returned instance is a module-level singleton — import graph_store
instead of calling this directly in application code.
"""

import os
from src.graph.backend import GraphBackend

_instance: GraphBackend | None = None


def get_graph_backend() -> GraphBackend:
    """Return the module-level singleton graph backend."""
    global _instance
    if _instance is None:
        backend = os.getenv("GRAPH_BACKEND", "networkx").lower()
        if backend == "neo4j":
            from src.graph.neo4j_backend import Neo4jBackend
            _instance = Neo4jBackend()
        else:
            from src.graph.networkx_backend import NetworkXBackend
            _instance = NetworkXBackend()
    return _instance


def reset_graph_backend() -> None:
    """Force re-initialisation (used in tests)."""
    global _instance
    if _instance is not None:
        try:
            _instance.close()
        except Exception:
            pass
    _instance = None
