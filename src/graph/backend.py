"""
Abstract GraphBackend interface.
Implementations: NetworkXBackend (default), Neo4jBackend (opt-in via GRAPH_BACKEND=neo4j).
"""

from abc import ABC, abstractmethod
from typing import Optional


class GraphBackend(ABC):

    @abstractmethod
    def upsert_node(self, memory_id: int, user_id: str, strength: float,
                    importance: float, category: str) -> None:
        """Create or update a node for a memory."""

    @abstractmethod
    def upsert_edge(self, source_id: int, target_id: int,
                    relation: str, weight: float) -> None:
        """Create or strengthen a directed edge between two memory nodes."""

    @abstractmethod
    def get_neighbors(self, memory_id: int, user_id: str,
                      max_depth: int = 2) -> list:
        """
        BFS from memory_id up to max_depth hops.
        Returns list of {"memory_id": int, "distance": int, "edge_weight": float}.
        Only traverses nodes belonging to user_id.
        """

    @abstractmethod
    def boost_node_and_neighbors(self, memory_id: int, user_id: str,
                                  boost: float = 0.2,
                                  max_depth: int = 1) -> list:
        """
        Propagate a recall boost through depth-1 neighbors.
        Returns list of memory_ids that were boosted (for vector DB recall_count bump).
        """

    @abstractmethod
    def get_node_strength(self, memory_id: int) -> Optional[float]:
        """Return the cached strength of a node, or None if not found."""

    @abstractmethod
    def update_node_strength(self, memory_id: int, strength: float) -> None:
        """Refresh the cached strength after vector DB recomputes it."""

    @abstractmethod
    def get_all_nodes_for_user(self, user_id: str) -> list:
        """Return all node dicts for chain-aware pruning."""

    @abstractmethod
    def delete_node(self, memory_id: int) -> None:
        """Remove a node and all its edges from the graph."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""
