import json
from datetime import datetime, timezone
from dotenv import load_dotenv

from .embed import embed
from .decay import compute_strength
from src.db.connection import get_backend, get_conn
from src.graph.graph_store import expand_with_graph, propagate_recall

load_dotenv()

# Memories below this similarity are excluded from results
SIMILARITY_THRESHOLD = 0.50
SIMILARITY_THRESHOLD_SHORT = 0.50   # same as default — no special handling for short queries
SIMILARITY_THRESHOLD_FALLBACK = 0.20 # used when nothing found above threshold
# Memories above this similarity get recall_count reinforced
REINFORCE_THRESHOLD  = 0.75


def _parse_dt(value) -> datetime:
    """Normalize last_accessed_at to a UTC-aware datetime."""
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return datetime.now(timezone.utc)


def _cosine(a: list, b: list) -> float:
    import numpy as np
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def retrieve(user_id: str, query: str, top_k: int = 5, agent_id: str = None) -> dict:
    """
    Round 1 — vector search (cosine similarity):
       - DuckDB:   native array_cosine_similarity (default)
       - Postgres: pgvector operator
       - SQLite:   Python numpy cosine (legacy)

    Round 2 — graph expansion (multi-hop BFS from Round 1 seeds):
       Fetches additional memories linked by graph edges, merged and re-ranked.

    Recall propagation:
       High-similarity memories boost their graph neighbours' recall_count.
    """
    query_embedding = embed(query)
    backend = get_backend()
    is_short = len(query.split()) <= 3

    # Round 1: vector search
    if backend == "postgres":
        result = _retrieve_postgres(user_id, query_embedding, top_k, agent_id)
    elif backend == "duckdb":
        result = _retrieve_duckdb(user_id, query_embedding, top_k, agent_id, is_short=is_short)
    else:
        result = _retrieve_sqlite(user_id, query_embedding, top_k, agent_id, is_short=is_short)

    # Round 2: graph expansion
    seed_ids = [m["id"] for m in result.get("memories", [])]
    if seed_ids:
        extra_ids = expand_with_graph(seed_ids, user_id, top_k=top_k)
        existing_ids = set(seed_ids)
        new_ids = [i for i in extra_ids if i not in existing_ids]
        if new_ids:
            extra = _fetch_by_ids(new_ids, user_id, backend)
            result = _merge_graph_results(result, extra, top_k)

        # Recall propagation: boost graph neighbours of reinforced memories
        reinforced = [m for m in result.get("memories", [])
                      if m.get("similarity", 0) >= REINFORCE_THRESHOLD]
        for m in reinforced:
            boosted_ids = propagate_recall(m["id"], user_id)
            if boosted_ids:
                _bump_recall_count(boosted_ids, backend)

    return result


# ── Graph expansion helpers ───────────────────────────────────────────────────

def _fetch_by_ids(ids: list, user_id: str, backend: str) -> list:
    """Fetch memory rows by primary key list, return scored dicts."""
    if not ids:
        return []
    conn = get_conn()
    rows = []
    try:
        if backend == "postgres":
            from psycopg2.extras import RealDictCursor
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT id, content, category, importance, recall_count, last_accessed_at,"
                "       agent_id, visibility FROM memories WHERE id = ANY(%s) AND user_id = %s",
                (ids, user_id),
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
        elif backend == "duckdb":
            from src.db.connection import duckdb_rows
            placeholders = ", ".join("?" * len(ids))
            result = conn.execute(
                f"SELECT id, content, category, importance, recall_count, last_accessed_at,"
                f"       agent_id, visibility FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                ids + [user_id],
            )
            rows = duckdb_rows(result)
        else:
            cur = conn.cursor()
            placeholders = ", ".join("?" * len(ids))
            cur.execute(
                f"SELECT id, content, category, importance, recall_count, last_accessed_at,"
                f"       agent_id, visibility FROM memories WHERE id IN ({placeholders}) AND user_id = ?",
                ids + [user_id],
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
    except Exception:
        pass
    finally:
        conn.close()

    scored = []
    for m in rows:
        strength = compute_strength(
            last_accessed_at=_parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        # Graph-expanded memories: use strength as score proxy (no direct similarity)
        scored.append({
            **m,
            "similarity": 0.5,   # neutral; edge_weight already filtered by graph
            "strength":   strength,
            "score":      0.5 * strength,
            "via_graph":  True,
        })
    return scored


def _merge_graph_results(result: dict, extra: list, top_k: int) -> dict:
    """Merge graph-expanded memories into vector results, re-rank, trim to top_k."""
    if not extra:
        return result

    existing = result.get("memories", [])
    existing_ids = {m["id"] for m in existing}
    new_entries = [m for m in extra if m["id"] not in existing_ids]
    if not new_entries:
        return result

    merged = existing + new_entries
    merged.sort(key=lambda x: x["score"], reverse=True)
    top = merged[:top_k]

    facts       = [m for m in top if m["category"] == "fact"]
    assumptions = [m for m in top if m["category"] == "assumption"]
    strategies  = [m for m in top if m["category"] == "strategy"]
    failures    = [m for m in top if m["category"] == "failure"]

    context_parts = []
    if facts:
        context_parts.append("[Facts]\n" + "\n".join(m["content"] for m in facts))
    if assumptions:
        context_parts.append("[Assumptions]\n" + "\n".join(m["content"] for m in assumptions))
    if strategies:
        context_parts.append("[Strategies]\n" + "\n".join(m["content"] for m in strategies))
    if failures:
        context_parts.append("[Failures]\n" + "\n".join(m["content"] for m in failures))

    return {
        "memoriesFound": len(top),
        "context":       "\n\n".join(context_parts),
        "memories": [
            {
                "id":         m["id"],
                "content":    m["content"],
                "category":   m["category"],
                "agent_id":   m.get("agent_id"),
                "visibility": m.get("visibility"),
                "importance": round(m["importance"], 4),
                "strength":   round(m["strength"], 4),
                "similarity": round(m["similarity"], 4),
                "score":      round(m["score"], 4),
            }
            for m in top
        ],
    }


def _bump_recall_count(ids: list, backend: str) -> None:
    """Increment recall_count for a list of memory IDs (graph propagation)."""
    if not ids:
        return
    conn = get_conn()
    try:
        if backend == "postgres":
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ANY(%s)",
                (ids,),
            )
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            placeholders = ", ".join("?" * len(ids))
            conn.execute(
                f"UPDATE memories SET recall_count = recall_count + 1 WHERE id IN ({placeholders})",
                ids,
            )
        else:
            cur = conn.cursor()
            for mid in ids:
                cur.execute(
                    "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ?", (mid,)
                )
            conn.commit()
            cur.close()
    except Exception:
        pass
    finally:
        conn.close()


# ── Postgres path ─────────────────────────────────────────────────────────────

def _retrieve_postgres(user_id, query_embedding, top_k, agent_id):
    import psycopg2
    from psycopg2.extras import RealDictCursor

    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)

    if agent_id:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s
              AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = %s))
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, user_id, agent_id, embedding_str, SIMILARITY_THRESHOLD, embedding_str, top_k * 2))
    else:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s
              AND visibility = 'shared'
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding_str, user_id, embedding_str, SIMILARITY_THRESHOLD, embedding_str, top_k * 2))

    candidates = [dict(row) for row in cur.fetchall()]
    return _score_and_return(candidates, top_k, cur, conn, backend="postgres")


# ── DuckDB path ───────────────────────────────────────────────────────────────

def _retrieve_duckdb(user_id, query_embedding, top_k, agent_id, is_short=False):
    from src.db.connection import duckdb_rows
    threshold = SIMILARITY_THRESHOLD_SHORT if is_short else SIMILARITY_THRESHOLD
    conn = get_conn()

    if agent_id:
        cur = conn.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility,
                   array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
            FROM memories
            WHERE user_id = ?
              AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
              AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, user_id, agent_id, query_embedding, threshold, top_k * 2])
    else:
        cur = conn.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility,
                   array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
            FROM memories
            WHERE user_id = ?
              AND visibility = 'shared'
              AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, user_id, query_embedding, threshold, top_k * 2])

    candidates = duckdb_rows(cur)

    # Fallback to lower threshold if nothing found
    if not candidates:
        if agent_id:
            cur = conn.execute("""
                SELECT id, content, category, importance, recall_count, last_accessed_at,
                       agent_id, visibility,
                       array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
                FROM memories
                WHERE user_id = ?
                  AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
                  AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
                ORDER BY similarity DESC LIMIT ?
            """, [query_embedding, user_id, agent_id, query_embedding, SIMILARITY_THRESHOLD_FALLBACK, top_k * 2])
        else:
            cur = conn.execute("""
                SELECT id, content, category, importance, recall_count, last_accessed_at,
                       agent_id, visibility,
                       array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
                FROM memories
                WHERE user_id = ?
                  AND visibility = 'shared'
                  AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
                ORDER BY similarity DESC LIMIT ?
            """, [query_embedding, user_id, query_embedding, SIMILARITY_THRESHOLD_FALLBACK, top_k * 2])
        candidates = duckdb_rows(cur)

    return _score_and_return_duckdb(candidates, top_k, conn)


def _score_and_return_duckdb(candidates, top_k, conn):
    if not candidates:
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}

    scored = []
    for m in candidates:
        strength = compute_strength(
            last_accessed_at=_parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        scored.append({**m, "strength": strength, "score": m["similarity"] * strength})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    relevant_ids = [m["id"] for m in top if m["similarity"] >= REINFORCE_THRESHOLD]
    if relevant_ids:
        for mid in relevant_ids:
            conn.execute("""
                UPDATE memories
                SET recall_count = recall_count + 1, last_accessed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [mid])

    facts       = [m for m in top if m["category"] == "fact"]
    assumptions = [m for m in top if m["category"] == "assumption"]
    strategies  = [m for m in top if m["category"] == "strategy"]
    failures    = [m for m in top if m["category"] == "failure"]

    context_parts = []
    if facts:
        context_parts.append("[Facts]\n" + "\n".join(m["content"] for m in facts))
    if assumptions:
        context_parts.append("[Assumptions]\n" + "\n".join(m["content"] for m in assumptions))
    if strategies:
        context_parts.append("[Strategies]\n" + "\n".join(m["content"] for m in strategies))
    if failures:
        context_parts.append("[Failures]\n" + "\n".join(m["content"] for m in failures))

    conn.close()
    return {
        "memoriesFound": len(top),
        "context": "\n\n".join(context_parts),
        "memories": [
            {
                "id":         m["id"],
                "content":    m["content"],
                "category":   m["category"],
                "agent_id":   m.get("agent_id"),
                "visibility": m.get("visibility"),
                "importance": round(m["importance"], 4),
                "strength":   round(m["strength"], 4),
                "similarity": round(m["similarity"], 4),
                "score":      round(m["score"], 4),
            }
            for m in top
        ],
    }


# ── SQLite path ───────────────────────────────────────────────────────────────

def _retrieve_sqlite(user_id, query_embedding, top_k, agent_id, is_short=False):
    conn = get_conn()
    cur  = conn.cursor()

    if agent_id:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility, embedding
            FROM memories
            WHERE user_id = ?
              AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
        """, (user_id, agent_id))
    else:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility, embedding
            FROM memories
            WHERE user_id = ? AND visibility = 'shared'
        """, (user_id,))

    rows = cur.fetchall()
    threshold = SIMILARITY_THRESHOLD_SHORT if is_short else SIMILARITY_THRESHOLD

    def _filter(rows, threshold):
        candidates = []
        for row in rows:
            raw_emb = row["embedding"]
            if raw_emb is None:
                continue
            sim = _cosine(query_embedding, json.loads(raw_emb))
            if sim >= threshold:
                d = dict(row)
                d["similarity"] = sim
                candidates.append(d)
        return candidates

    candidates = _filter(rows, threshold)

    # Fallback: lower threshold if nothing found
    if not candidates:
        candidates = _filter(rows, SIMILARITY_THRESHOLD_FALLBACK)

    return _score_and_return(candidates, top_k, cur, conn, backend="sqlite")


# ── Shared scoring + return ───────────────────────────────────────────────────

def _score_and_return(candidates, top_k, cur, conn, backend):
    if not candidates:
        cur.close()
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}

    scored = []
    for m in candidates:
        strength = compute_strength(
            last_accessed_at=_parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        scored.append({**m, "strength": strength, "score": m["similarity"] * strength})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    if not top:
        cur.close()
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}

    relevant_ids = [m["id"] for m in top if m["similarity"] >= REINFORCE_THRESHOLD]
    if relevant_ids:
        if backend == "postgres":
            cur.execute("""
                UPDATE memories
                SET recall_count = recall_count + 1, last_accessed_at = NOW()
                WHERE id = ANY(%s)
            """, (relevant_ids,))
        else:
            for mid in relevant_ids:
                cur.execute("""
                    UPDATE memories
                    SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (mid,))
    conn.commit()

    facts       = [m for m in top if m["category"] == "fact"]
    assumptions = [m for m in top if m["category"] == "assumption"]
    strategies  = [m for m in top if m["category"] == "strategy"]
    failures    = [m for m in top if m["category"] == "failure"]

    context_parts = []
    if facts:
        context_parts.append("[Facts]\n" + "\n".join(m["content"] for m in facts))
    if assumptions:
        context_parts.append("[Assumptions]\n" + "\n".join(m["content"] for m in assumptions))
    if strategies:
        context_parts.append("[Strategies]\n" + "\n".join(m["content"] for m in strategies))
    if failures:
        context_parts.append("[Failures]\n" + "\n".join(m["content"] for m in failures))
    context = "\n\n".join(context_parts)

    cur.close()
    conn.close()

    return {
        "memoriesFound": len(top),
        "context": context,
        "memories": [
            {
                "id":         m["id"],
                "content":    m["content"],
                "category":   m["category"],
                "agent_id":   m.get("agent_id"),
                "visibility": m.get("visibility"),
                "importance": round(m["importance"], 4),
                "strength":   round(m["strength"], 4),
                "similarity": round(m["similarity"], 4),
                "score":      round(m["score"], 4),
            }
            for m in top
        ],
    }
