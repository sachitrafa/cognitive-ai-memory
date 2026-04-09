"""
NetworkX-based graph backend — default, zero setup.
Graph is persisted as a pickle file at ~/.yourmemory/graph.pkl.
"""

import os
import pickle
import threading
from pathlib import Path
from typing import Optional

from src.graph.backend import GraphBackend


def _graph_path() -> Path:
    custom = os.getenv("YOURMEMORY_PATH")
    base = Path(custom) if custom else Path.home() / ".yourmemory"
    base.mkdir(parents=True, exist_ok=True)
    return base / "graph.pkl"


class NetworkXBackend(GraphBackend):

    def __init__(self):
        import networkx as nx
        self._nx = nx
        self._lock = threading.RLock()
        self._path = _graph_path()
        self._G: "nx.DiGraph" = self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return self._nx.DiGraph()

    def _flush(self):
        with open(self._path, "wb") as f:
            pickle.dump(self._G, f)

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #

    def upsert_node(self, memory_id: int, user_id: str, strength: float,
                    importance: float, category: str) -> None:
        with self._lock:
            self._G.add_node(memory_id,
                             user_id=user_id,
                             strength=strength,
                             importance=importance,
                             category=category,
                             recall_proxy=0.0)
            self._flush()

    def update_node_strength(self, memory_id: int, strength: float) -> None:
        with self._lock:
            if self._G.has_node(memory_id):
                self._G.nodes[memory_id]["strength"] = strength
                self._flush()

    def get_node_strength(self, memory_id: int) -> Optional[float]:
        with self._lock:
            if self._G.has_node(memory_id):
                return self._G.nodes[memory_id].get("strength")
            return None

    def get_all_nodes_for_user(self, user_id: str) -> list:
        with self._lock:
            return [
                {"memory_id": n, **data}
                for n, data in self._G.nodes(data=True)
                if data.get("user_id") == user_id
            ]

    def delete_node(self, memory_id: int) -> None:
        with self._lock:
            if self._G.has_node(memory_id):
                self._G.remove_node(memory_id)
                self._flush()

    # ------------------------------------------------------------------ #
    # Edge operations
    # ------------------------------------------------------------------ #

    def upsert_edge(self, source_id: int, target_id: int,
                    relation: str, weight: float) -> None:
        with self._lock:
            if self._G.has_edge(source_id, target_id):
                # Strengthen existing edge slightly
                existing = self._G[source_id][target_id].get("weight", weight)
                self._G[source_id][target_id]["weight"] = min(1.0, existing + weight * 0.1)
            else:
                self._G.add_edge(source_id, target_id,
                                  relation=relation,
                                  weight=weight)
            self._flush()

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def get_neighbors(self, memory_id: int, user_id: str,
                      max_depth: int = 2) -> list:
        with self._lock:
            if not self._G.has_node(memory_id):
                return []

            visited = {}  # memory_id → {"distance": int, "edge_weight": float}
            queue = [(memory_id, 0, 1.0)]  # (node, depth, cumulative_weight)

            while queue:
                node, depth, cum_weight = queue.pop(0)
                if depth > max_depth:
                    continue

                # Both successors and predecessors (undirected traversal)
                neighbors = list(self._G.successors(node)) + list(self._G.predecessors(node))
                for nbr in neighbors:
                    if nbr == memory_id:
                        continue
                    nbr_data = self._G.nodes.get(nbr, {})
                    if nbr_data.get("user_id") != user_id:
                        continue
                    edge_w = (self._G[node][nbr].get("weight", 0.5)
                              if self._G.has_edge(node, nbr)
                              else self._G[nbr][node].get("weight", 0.5))
                    new_weight = cum_weight * edge_w
                    if nbr not in visited or visited[nbr]["edge_weight"] < new_weight:
                        visited[nbr] = {"memory_id": nbr,
                                        "distance": depth + 1,
                                        "edge_weight": new_weight}
                        if depth + 1 < max_depth:
                            queue.append((nbr, depth + 1, new_weight))

            return list(visited.values())

    def boost_node_and_neighbors(self, memory_id: int, user_id: str,
                                  boost: float = 0.2,
                                  max_depth: int = 1) -> list:
        with self._lock:
            boosted = []
            neighbors = self.get_neighbors(memory_id, user_id, max_depth=max_depth)
            for nbr in neighbors:
                nid = nbr["memory_id"]
                if self._G.has_node(nid):
                    edge_w = nbr["edge_weight"]
                    self._G.nodes[nid]["recall_proxy"] = (
                        self._G.nodes[nid].get("recall_proxy", 0.0) + boost * edge_w
                    )
                    boosted.append(nid)
            if boosted:
                self._flush()
            return boosted

    def close(self) -> None:
        with self._lock:
            self._flush()
