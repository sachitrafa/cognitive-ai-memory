"""
High-level graph façade used by the rest of the application.

Four public functions:
  index_memory(memory_id, user_id, content, strength, importance, category)
      → extract SVO triples, upsert node, upsert edges (similarity + entity)

  expand_with_graph(seed_ids, user_id, top_k) → list of extra memory_ids
      → multi-hop BFS from each seed; deduplicated, sorted by edge_weight

  propagate_recall(memory_id, user_id) → list of memory_ids that were boosted
      → boost recall_proxy on depth-1 neighbours (for recall_count bump in DB)

  chain_safe_to_prune(memory_id, user_id, threshold) → bool
      → True only when ALL graph neighbours are also below the threshold

All four are silent no-ops if the graph backend is unavailable.

Edge strategy:
  Round 1 (semantic):  cosine similarity ≥ 0.4  → weight = sim × verb_weight
  Round 2 (entity):    shared named entity mention → weight = ENTITY_EDGE_WEIGHT (0.45)

Entity edges connect memories that wouldn't score high in embedding space but
share a bridge entity — the key pattern in multi-hop reasoning chains.
Example: "Obama was born in Hawaii" ←→ "Hawaii became a US state in 1959"
share the entity "Hawaii" even though their embeddings are dissimilar.
"""

import sys
from typing import Optional

from src.graph.svo_extract import extract_triples

# Lazy: don't crash the server if networkx is missing
_graph = None


def _g():
    global _graph
    if _graph is None:
        try:
            from src.graph import get_graph_backend
            _graph = get_graph_backend()
        except Exception as exc:
            print(f"[graph_store] backend unavailable: {exc}", file=sys.stderr)
    return _graph


# ------------------------------------------------------------------ #
# Index a stored memory (call after INSERT)
# ------------------------------------------------------------------ #

def index_memory(
    memory_id:  int,
    user_id:    str,
    content:    str,
    strength:   float,
    importance: float,
    category:   str,
    embedding:  list | None = None,
) -> None:
    """
    Register a memory node in the graph and create semantically-weighted edges.

    Edge strategy:
    - If `embedding` is provided (always the case from store_memory):
        Query the DB for the top-5 most similar *existing* memories by cosine
        similarity (above a 0.3 floor).  Edge weight = similarity × verb_weight
        from the SVO triple (or 0.5 default if spaCy unavailable).
        This means edges connect memories that are *actually related*, not just
        stored around the same time.
    - Fallback (no embedding): connect to 5 most recent nodes at weight 0.4.
    """
    g = _g()
    if g is None:
        return

    try:
        g.upsert_node(memory_id, user_id, strength, importance, category)
    except Exception as exc:
        print(f"[graph_store] upsert_node failed: {exc}", file=sys.stderr)
        return

    # ── Find semantically similar neighbours via the vector DB ──────────
    if embedding is not None:
        similar = _similar_nodes(memory_id, user_id, embedding, top_k=5, min_sim=0.4)
    else:
        similar = []

    # ── Get verb weight from SVO triple (if spaCy available) ────────────
    triples = extract_triples(content)
    verb_weight = max((t["weight"] for t in triples), default=0.5)
    relation    = triples[0]["predicate"] if triples else "related"

    # ── Similarity edges: weight = cosine_similarity × verb_weight ──────
    sim_linked: set[int] = set()
    for nbr in similar:
        target_id   = nbr["memory_id"]
        edge_weight = round(nbr["similarity"] * verb_weight, 4)
        try:
            g.upsert_edge(memory_id, target_id, relation, edge_weight)
            sim_linked.add(target_id)
        except Exception as exc:
            print(f"[graph_store] upsert_edge failed: {exc}", file=sys.stderr)

    # ── Entity edges: connect memories sharing named entity mentions ──────
    # Always apply even if a weak similarity edge already exists — entity edges
    # can upgrade a low-weight similarity edge to ENTITY_EDGE_WEIGHT via
    # the max() policy in upsert_edge.
    entity_nbrs = _entity_linked_nodes(memory_id, user_id, content,
                                       already_linked=set())   # no exclusion
    for en in entity_nbrs:
        target_id = en["memory_id"]
        try:
            g.upsert_edge(memory_id, target_id,
                          f"entity:{en['entity']}", ENTITY_EDGE_WEIGHT)
        except Exception as exc:
            print(f"[graph_store] entity_edge failed: {exc}", file=sys.stderr)


def _similar_nodes(
    memory_id: int,
    user_id:   str,
    embedding: list,
    top_k:     int   = 5,
    min_sim:   float = 0.3,
) -> list:
    """
    Query the vector DB for the top-k most similar existing memories
    (excluding memory_id itself).  Returns [{memory_id, similarity}].
    """
    from src.db.connection import get_backend, get_conn
    from src.db.connection import duckdb_rows
    backend = get_backend()
    conn    = get_conn()

    try:
        if backend == "duckdb":
            result = conn.execute("""
                SELECT id,
                       array_cosine_similarity(embedding, ?::FLOAT[768]) AS sim
                FROM memories
                WHERE user_id = ? AND id != ?
                  AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
                ORDER BY sim DESC
                LIMIT ?
            """, [embedding, user_id, memory_id, embedding, min_sim, top_k])
            rows = duckdb_rows(result)
            return [{"memory_id": r["id"], "similarity": r["sim"]} for r in rows]

        elif backend == "postgres":
            emb_str = f"[{','.join(str(x) for x in embedding)}]"
            cur = conn.cursor()
            cur.execute("""
                SELECT id,
                       1 - (embedding <=> %s::vector) AS sim
                FROM memories
                WHERE user_id = %s AND id != %s
                  AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY sim DESC
                LIMIT %s
            """, (emb_str, user_id, memory_id, emb_str, min_sim, top_k))
            rows = cur.fetchall()
            cur.close()
            return [{"memory_id": r[0], "similarity": r[1]} for r in rows]

        else:  # sqlite — compute cosine in Python
            import json
            import numpy as np
            cur = conn.cursor()
            cur.execute(
                "SELECT id, embedding FROM memories WHERE user_id = ? AND id != ?",
                (user_id, memory_id),
            )
            va = np.array(embedding, dtype=float)
            results = []
            for row in cur.fetchall():
                if row[1] is None:
                    continue
                vb  = np.array(json.loads(row[1]), dtype=float)
                den = np.linalg.norm(va) * np.linalg.norm(vb)
                sim = float(np.dot(va, vb) / den) if den else 0.0
                if sim >= min_sim:
                    results.append({"memory_id": row[0], "similarity": sim})
            results.sort(key=lambda x: x["similarity"], reverse=True)
            cur.close()
            return results[:top_k]
    except Exception as exc:
        print(f"[graph_store] _similar_nodes failed: {exc}", file=sys.stderr)
        return []
    finally:
        conn.close()


# ------------------------------------------------------------------ #
# Entity-based neighbours (bridge-entity multi-hop linking)
# ------------------------------------------------------------------ #

# NER labels worth bridging across — skips dates/numbers that appear everywhere
_BRIDGE_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "NORP", "LAW",
})
ENTITY_EDGE_WEIGHT = 0.55   # calibrated so score = W_VECTOR × 0.55 = 0.33, competitive with min Round 1 hit


def _entity_linked_nodes(
    memory_id:      int,
    user_id:        str,
    content:        str,
    already_linked: set[int],
    top_k:          int = 5,
    min_entity_len: int = 3,
) -> list[dict]:
    """
    Find existing memories that share a named entity with `content`.

    Uses spaCy NER (en_core_web_sm) to extract entities, then does a
    case-insensitive LIKE search in the vector DB.  Only returns nodes
    not already connected via similarity edges.

    Returns [{"memory_id": int, "entity": str}].
    """
    try:
        from src.services.extract import _nlp
        if _nlp is None:
            return []
    except Exception:
        return []

    doc = _nlp(content)
    entities = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in _BRIDGE_LABELS and len(ent.text.strip()) >= min_entity_len
    ]
    if not entities:
        return []

    # For multi-word entities, also search for word-prefix sub-strings so that
    # "Shirley Temple Black" (F2) matches "Shirley Temple" in F1.
    search_terms: list[str] = []
    seen_terms: set[str] = set()
    for ent in entities[:6]:
        words = ent.split()
        # Add N-word, (N-1)-word, ..., 2-word prefixes  (min 2 words to avoid noise)
        for n in range(len(words), 1, -1):
            candidate = " ".join(words[:n])
            if candidate not in seen_terms:
                seen_terms.add(candidate)
                search_terms.append(candidate)
        # Also add the entity as-is if it's a single word (length >= min_entity_len)
        if len(words) == 1 and ent not in seen_terms:
            seen_terms.add(ent)
            search_terms.append(ent)

    if not search_terms:
        return []

    from src.db.connection import get_backend, get_conn, duckdb_rows
    backend = get_backend()
    conn    = get_conn()
    try:
        found: dict[int, str] = {}
        for entity in search_terms[:10]:  # cap total searches
            like_pat = f"%{entity}%"
            if backend == "duckdb":
                result = conn.execute("""
                    SELECT id FROM memories
                    WHERE user_id = ? AND id != ?
                      AND lower(content) LIKE lower(?)
                    LIMIT ?
                """, [user_id, memory_id, like_pat, top_k])
                rows = duckdb_rows(result)
                ids  = [r["id"] for r in rows]
            elif backend == "postgres":
                cur = conn.cursor()
                cur.execute("""
                    SELECT id FROM memories
                    WHERE user_id = %s AND id != %s
                      AND lower(content) LIKE lower(%s)
                    LIMIT %s
                """, (user_id, memory_id, like_pat, top_k))
                ids = [r[0] for r in cur.fetchall()]
                cur.close()
            else:  # sqlite
                cur = conn.cursor()
                cur.execute("""
                    SELECT id FROM memories
                    WHERE user_id = ? AND id != ?
                      AND lower(content) LIKE lower(?)
                    LIMIT ?
                """, (user_id, memory_id, like_pat, top_k))
                ids = [r[0] for r in cur.fetchall()]
                cur.close()

            for mid in ids:
                if mid not in already_linked and mid not in found:
                    found[mid] = entity

        return [{"memory_id": mid, "entity": ent} for mid, ent in found.items()]
    except Exception as exc:
        print(f"[graph_store] _entity_linked_nodes failed: {exc}", file=sys.stderr)
        return []
    finally:
        conn.close()


# ------------------------------------------------------------------ #
# Expand vector search results with graph neighbours
# ------------------------------------------------------------------ #

def expand_with_graph(
    seed_ids: list[int],
    user_id:  str,
    top_k:    int = 5,
) -> list[tuple[int, float]]:
    """
    BFS from each seed memory_id; return up to top_k (id, edge_weight) pairs
    not in seeds, sorted by edge_weight descending.

    edge_weight = cosine_similarity × verb_weight from SVO extraction —
    represents how strongly each expanded node is connected to its seed.
    """
    g = _g()
    if g is None or not seed_ids:
        return []

    seen_seeds = set(seed_ids)
    candidates: dict[int, float] = {}  # memory_id → best edge_weight

    for seed in seed_ids:
        try:
            neighbours = g.get_neighbors(seed, user_id, max_depth=2)
        except Exception as exc:
            print(f"[graph_store] get_neighbors failed: {exc}", file=sys.stderr)
            continue
        for nbr in neighbours:
            nid = nbr["memory_id"]
            if nid in seen_seeds:
                continue
            ew = nbr["edge_weight"]
            if nid not in candidates or candidates[nid] < ew:
                candidates[nid] = ew

    ranked = sorted(candidates, key=lambda k: candidates[k], reverse=True)
    return [(nid, candidates[nid]) for nid in ranked[:top_k]]


# ------------------------------------------------------------------ #
# Propagate recall boost through graph edges
# ------------------------------------------------------------------ #

def propagate_recall(memory_id: int, user_id: str) -> list[int]:
    """
    Boost recall_proxy on depth-1 neighbours after a memory is recalled.

    Returns the list of boosted memory_ids so the caller can increment
    recall_count in the vector DB.
    """
    g = _g()
    if g is None:
        return []
    try:
        return g.boost_node_and_neighbors(memory_id, user_id,
                                          boost=0.2, max_depth=1)
    except Exception as exc:
        print(f"[graph_store] propagate_recall failed: {exc}", file=sys.stderr)
        return []


# ------------------------------------------------------------------ #
# Chain-aware pruning gate
# ------------------------------------------------------------------ #

def chain_safe_to_prune(
    memory_id: int,
    user_id:   str,
    threshold: float,
) -> bool:
    """
    Return True if it is safe to prune this memory.

    A memory is safe to prune only when ALL of its graph neighbours are
    also below `threshold`.  If any neighbour is still strong, the memory
    is kept alive (chain integrity).

    Falls back to True (prune normally) if the graph backend is unavailable.
    """
    g = _g()
    if g is None:
        return True

    try:
        neighbours = g.get_neighbors(memory_id, user_id, max_depth=1)
    except Exception as exc:
        print(f"[graph_store] chain_safe_to_prune failed: {exc}", file=sys.stderr)
        return True

    if not neighbours:
        return True  # isolated node — prune normally

    for nbr in neighbours:
        nid = nbr["memory_id"]
        strength = g.get_node_strength(nid)
        if strength is not None and strength >= threshold:
            return False  # at least one strong neighbour → keep alive

    return True
