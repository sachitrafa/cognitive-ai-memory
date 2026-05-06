"""
Neo4j graph backend — opt-in via GRAPH_BACKEND=neo4j.
Requires: pip install yourmemory[neo4j]  (neo4j driver)

Connection: set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars.
Defaults: bolt://localhost:7687 / neo4j / neo4j
"""

import os
from typing import Optional

from src.graph.backend import GraphBackend


def _get_driver():
    try:
        from neo4j import GraphDatabase  # type: ignore
    except ImportError:
        raise ImportError(
            "neo4j driver not installed. Run: pip install 'yourmemory[neo4j]'"
        )
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pw   = os.getenv("NEO4J_PASSWORD", "neo4j")
    return GraphDatabase.driver(uri, auth=(user, pw))


class Neo4jBackend(GraphBackend):
    """
    Graph backend that persists nodes/edges in Neo4j.

    Schema:
      (:Memory {memory_id, user_id, strength, importance, category, recall_proxy})
      -[:RELATES {relation, weight}]->
    """

    def __init__(self):
        self._driver = _get_driver()
        self._ensure_indexes()

    def _ensure_indexes(self):
        with self._driver.session() as s:
            s.run(
                "CREATE INDEX memory_id_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.memory_id)"
            )
            s.run(
                "CREATE INDEX memory_user_idx IF NOT EXISTS "
                "FOR (m:Memory) ON (m.user_id)"
            )

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #

    def upsert_node(self, memory_id: int, user_id: str, strength: float,
                    importance: float, category: str) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MERGE (m:Memory {memory_id: $mid})
                SET m.user_id    = $uid,
                    m.strength   = $strength,
                    m.importance = $importance,
                    m.category   = $category,
                    m.recall_proxy = coalesce(m.recall_proxy, 0.0)
                """,
                mid=memory_id, uid=user_id,
                strength=strength, importance=importance, category=category,
            )

    def update_node_strength(self, memory_id: int, strength: float) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (m:Memory {memory_id: $mid}) SET m.strength = $strength",
                mid=memory_id, strength=strength,
            )

    def get_node_strength(self, memory_id: int) -> Optional[float]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (m:Memory {memory_id: $mid}) RETURN m.strength AS strength",
                mid=memory_id,
            ).single()
            return result["strength"] if result else None

    def get_all_nodes_for_user(self, user_id: str) -> list:
        with self._driver.session() as s:
            records = s.run(
                """
                MATCH (m:Memory {user_id: $uid})
                RETURN m.memory_id AS memory_id,
                       m.strength   AS strength,
                       m.importance AS importance,
                       m.category   AS category,
                       m.recall_proxy AS recall_proxy
                """,
                uid=user_id,
            )
            return [
                {
                    "memory_id":   r["memory_id"],
                    "user_id":     user_id,
                    "strength":    r["strength"],
                    "importance":  r["importance"],
                    "category":    r["category"],
                    "recall_proxy": r["recall_proxy"],
                }
                for r in records
            ]

    def delete_node(self, memory_id: int) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (m:Memory {memory_id: $mid}) DETACH DELETE m",
                mid=memory_id,
            )

    # ------------------------------------------------------------------ #
    # Edge operations
    # ------------------------------------------------------------------ #

    def upsert_edge(self, source_id: int, target_id: int,
                    relation: str, weight: float) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MATCH (a:Memory {memory_id: $src}), (b:Memory {memory_id: $tgt})
                MERGE (a)-[r:RELATES]->(b)
                ON CREATE SET r.relation = $relation, r.weight = $weight
                ON MATCH  SET r.weight   = CASE
                    WHEN $weight > r.weight THEN $weight
                    ELSE r.weight
                END
                """,
                src=source_id, tgt=target_id,
                relation=relation, weight=weight,
            )

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #

    def get_neighbors(self, memory_id: int, user_id: str,
                      max_depth: int = 2) -> list:
        """BFS up to max_depth hops (undirected) — only returns user_id nodes."""
        with self._driver.session() as s:
            records = s.run(
                """
                MATCH path = (start:Memory {memory_id: $mid})-[:RELATES*1..$depth]-(nbr:Memory)
                WHERE nbr.user_id = $uid AND nbr.memory_id <> $mid
                WITH nbr,
                     length(path) AS distance,
                     reduce(w = 1.0, r IN relationships(path) | w * r.weight) AS edge_weight
                RETURN nbr.memory_id AS memory_id, distance, edge_weight
                ORDER BY edge_weight DESC
                """,
                mid=memory_id, uid=user_id, depth=max_depth,
            )
            # Deduplicate: keep highest edge_weight per neighbor
            seen: dict[int, dict] = {}
            for r in records:
                nid = r["memory_id"]
                if nid not in seen or r["edge_weight"] > seen[nid]["edge_weight"]:
                    seen[nid] = {
                        "memory_id":   nid,
                        "distance":    r["distance"],
                        "edge_weight": r["edge_weight"],
                    }
            return list(seen.values())

    def boost_node_and_neighbors(self, memory_id: int, user_id: str,
                                  boost: float = 0.2,
                                  max_depth: int = 1) -> list:
        neighbors = self.get_neighbors(memory_id, user_id, max_depth=max_depth)
        if not neighbors:
            return []
        boosted = []
        with self._driver.session() as s:
            for nbr in neighbors:
                nid      = nbr["memory_id"]
                edge_w   = nbr["edge_weight"]
                s.run(
                    """
                    MATCH (m:Memory {memory_id: $mid})
                    SET m.recall_proxy = coalesce(m.recall_proxy, 0.0) + $delta
                    """,
                    mid=nid, delta=boost * edge_w,
                )
                boosted.append(nid)
        return boosted

    def close(self) -> None:
        self._driver.close()
