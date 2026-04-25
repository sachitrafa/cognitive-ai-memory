import json
import math
import sys
from datetime import datetime, timezone

from .embed import embed
from .decay import compute_strength
from .utils import parse_dt
from src.db.connection import get_backend, get_conn
from src.graph.graph_store import expand_with_graph, propagate_recall

SIMILARITY_THRESHOLD          = 0.50
SIMILARITY_THRESHOLD_FALLBACK = 0.20
REINFORCE_THRESHOLD           = 0.75

# Hybrid scoring weights: BM25 keyword + vector×decay
W_BM25   = 0.4
W_VECTOR = 0.6


# ── BM25 / FTS helpers ───────────────────────────────────────────────────────

def _normalize_bm25_sqlite(raw: float) -> float:
    """
    SQLite FTS5 bm25() returns negative values — more negative = better match.
    Convert to [0, 1] via sigmoid of the negated score.
    """
    return 1.0 / (1.0 + math.exp(raw))   # raw is negative, so exp(raw) < 1


def _fts_search_sqlite(conn, user_id: str, agent_id: str | None,
                       query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} for FTS5 matches."""
    try:
        cur = conn.cursor()
        if agent_id:
            cur.execute("""
                SELECT m.id, bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                  AND m.user_id = ?
                  AND (m.visibility = 'shared'
                       OR (m.visibility = 'private' AND m.agent_id = ?))
                ORDER BY bm25_score
                LIMIT ?
            """, (query, user_id, agent_id, limit))
        else:
            cur.execute("""
                SELECT m.id, bm25(memories_fts) AS bm25_score
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.id
                WHERE memories_fts MATCH ?
                  AND m.user_id = ?
                  AND m.visibility = 'shared'
                ORDER BY bm25_score
                LIMIT ?
            """, (query, user_id, limit))
        rows = cur.fetchall()
        cur.close()
        return {r[0]: _normalize_bm25_sqlite(r[1]) for r in rows}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (sqlite): {exc}", file=sys.stderr)
        return {}


def _fts_search_duckdb(conn, user_id: str, agent_id: str | None,
                       query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} using DuckDB FTS extension."""
    try:
        conn.execute("LOAD fts;")
        conn.execute(
            "PRAGMA create_fts_index('memories', 'id', 'content', overwrite=1);"
        )
        if agent_id:
            result = conn.execute("""
                SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                FROM memories
                WHERE score IS NOT NULL
                  AND user_id = ?
                  AND (visibility = 'shared'
                       OR (visibility = 'private' AND agent_id = ?))
                ORDER BY score DESC
                LIMIT ?
            """, [query, user_id, agent_id, limit])
        else:
            result = conn.execute("""
                SELECT id, fts_main_memories.match_bm25(id, ?) AS score
                FROM memories
                WHERE score IS NOT NULL
                  AND user_id = ?
                  AND visibility = 'shared'
                ORDER BY score DESC
                LIMIT ?
            """, [query, user_id, limit])
        from src.db.connection import duckdb_rows
        rows = duckdb_rows(result)
        # DuckDB FTS returns positive scores; normalize with score/(score+1)
        return {r["id"]: r["score"] / (r["score"] + 1.0) for r in rows if r["score"]}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (duckdb): {exc}", file=sys.stderr)
        return {}


def _fts_search_postgres(conn, user_id: str, agent_id: str | None,
                         query: str, limit: int) -> dict[int, float]:
    """Return {memory_id: bm25_norm} using Postgres tsvector + ts_rank."""
    try:
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        tsq = " & ".join(query.split())   # simple AND query — good for identifiers
        if agent_id:
            cur.execute("""
                SELECT id, ts_rank_cd(content_tsv, to_tsquery('english', %s)) AS score
                FROM memories
                WHERE content_tsv @@ to_tsquery('english', %s)
                  AND user_id = %s
                  AND (visibility = 'shared'
                       OR (visibility = 'private' AND agent_id = %s))
                ORDER BY score DESC
                LIMIT %s
            """, (tsq, tsq, user_id, agent_id, limit))
        else:
            cur.execute("""
                SELECT id, ts_rank_cd(content_tsv, to_tsquery('english', %s)) AS score
                FROM memories
                WHERE content_tsv @@ to_tsquery('english', %s)
                  AND user_id = %s
                  AND visibility = 'shared'
                ORDER BY score DESC
                LIMIT %s
            """, (tsq, tsq, user_id, limit))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        # ts_rank_cd already returns 0–1
        return {r["id"]: float(r["score"]) for r in rows}
    except Exception as exc:
        print(f"[retrieve] FTS search failed (postgres): {exc}", file=sys.stderr)
        return {}


def retrieve(user_id: str, query: str, top_k: int = 5, agent_id: str = None) -> dict:
    """
    Round 1 — vector search (cosine similarity):
       - DuckDB:   native array_cosine_similarity
       - Postgres: pgvector operator
       - SQLite:   Python numpy cosine

    Round 2 — graph expansion (multi-hop BFS from Round 1 seeds).
    Recall propagation: high-similarity memories boost graph neighbours.
    """
    query_embedding = embed(query)
    backend = get_backend()

    if backend == "postgres":
        result = _retrieve_postgres(user_id, query, query_embedding, top_k, agent_id)
    elif backend == "duckdb":
        result = _retrieve_duckdb(user_id, query, query_embedding, top_k, agent_id)
    else:
        result = _retrieve_sqlite(user_id, query, query_embedding, top_k, agent_id)

    seed_ids = [m["id"] for m in result.get("memories", [])]
    if seed_ids:
        extra_ids = expand_with_graph(seed_ids, user_id, top_k=top_k)
        new_ids = [i for i in extra_ids if i not in set(seed_ids)]
        if new_ids:
            extra = _fetch_by_ids(new_ids, user_id, backend)
            result = _merge_graph_results(result, extra, top_k)

        reinforced = [m for m in result.get("memories", [])
                      if m.get("similarity", 0) >= REINFORCE_THRESHOLD]
        for m in reinforced:
            boosted_ids = propagate_recall(m["id"], user_id)
            if boosted_ids:
                _bump_recall_count(boosted_ids, backend)

    return result


# ── Shared helpers ────────────────────────────────────────────────────────────

def _score_candidates(candidates: list, fts_scores: dict[int, float] | None = None) -> list:
    """
    Add strength + hybrid score to each candidate.

    Hybrid formula:
        score = W_BM25 × bm25_norm + W_VECTOR × (cosine × ebbinghaus)

    fts_scores: {memory_id → normalized BM25 score} from keyword search.
    If a candidate has no BM25 hit, its BM25 component is 0 and the vector
    signal carries it alone — preserving existing behaviour for pure semantic queries.
    """
    fts = fts_scores or {}
    scored = []
    for m in candidates:
        strength = compute_strength(
            last_accessed_at=parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        bm25_norm    = fts.get(m["id"], 0.0)
        vector_score = m["similarity"] * strength
        hybrid_score = W_BM25 * bm25_norm + W_VECTOR * vector_score
        scored.append({
            **m,
            "strength":   strength,
            "bm25":       round(bm25_norm, 4),
            "score":      hybrid_score,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _build_context(top: list) -> str:
    """Build the categorized context string from a ranked memory list."""
    categories = ["fact", "assumption", "strategy", "failure"]
    labels     = {"fact": "[Facts]", "assumption": "[Assumptions]",
                  "strategy": "[Strategies]", "failure": "[Failures]"}
    parts = []
    for cat in categories:
        group = [m for m in top if m["category"] == cat]
        if group:
            parts.append(labels[cat] + "\n" + "\n".join(m["content"] for m in group))
    return "\n\n".join(parts)


def _format_result(top: list) -> dict:
    """Turn a ranked memory list into the standard API response dict."""
    return {
        "memoriesFound": len(top),
        "context": _build_context(top),
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
                "bm25":       round(m.get("bm25", 0.0), 4),
                "score":      round(m["score"], 4),
            }
            for m in top
        ],
    }


def _reinforce(top: list, conn, backend: str) -> None:
    """Bump recall_count + last_accessed_at for high-similarity memories."""
    relevant_ids = [m["id"] for m in top if m["similarity"] >= REINFORCE_THRESHOLD]
    if not relevant_ids:
        return
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute(
            "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()"
            " WHERE id = ANY(%s)",
            (relevant_ids,),
        )
        cur.close()
    elif backend == "duckdb":
        for mid in relevant_ids:
            conn.execute(
                "UPDATE memories SET recall_count = recall_count + 1,"
                " last_accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                [mid],
            )
    else:
        cur = conn.cursor()
        for mid in relevant_ids:
            cur.execute(
                "UPDATE memories SET recall_count = recall_count + 1,"
                " last_accessed_at = datetime('now') WHERE id = ?",
                (mid,),
            )
        cur.close()


def _finish(candidates: list, top_k: int, conn, backend: str,
            fts_scores: dict[int, float] | None = None) -> dict:
    """Score → rank → reinforce → format → close connection."""
    scored = _score_candidates(candidates, fts_scores)
    top = scored[:top_k]
    if not top:
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}
    _reinforce(top, conn, backend)
    if backend != "duckdb":
        conn.commit()
    conn.close()
    return _format_result(top)


# ── Graph expansion helpers ───────────────────────────────────────────────────

def _fetch_by_ids(ids: list, user_id: str, backend: str) -> list:
    """Fetch memory rows by primary key, return scored dicts."""
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
    except Exception as exc:
        print(f"[retrieve] _fetch_by_ids failed: {exc}", file=sys.stderr)
    finally:
        conn.close()

    scored = []
    for m in rows:
        strength = compute_strength(
            last_accessed_at=parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        scored.append({
            **m,
            "similarity": 0.5,
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
    return _format_result(merged[:top_k])


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
    except Exception as exc:
        print(f"[retrieve] _bump_recall_count failed: {exc}", file=sys.stderr)
    finally:
        conn.close()


# ── Postgres path ─────────────────────────────────────────────────────────────

def _retrieve_postgres(user_id, query, query_embedding, top_k, agent_id):
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
        """, (embedding_str, user_id, agent_id, embedding_str,
              SIMILARITY_THRESHOLD, embedding_str, top_k * 2))
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
        """, (embedding_str, user_id, embedding_str,
              SIMILARITY_THRESHOLD, embedding_str, top_k * 2))
    candidates = [dict(row) for row in cur.fetchall()]
    cur.close()
    fts_scores = _fts_search_postgres(conn, user_id, agent_id, query=query, limit=top_k * 2)
    return _finish(candidates, top_k, conn, backend="postgres", fts_scores=fts_scores)


# ── DuckDB path ───────────────────────────────────────────────────────────────

def _retrieve_duckdb(user_id, query, query_embedding, top_k, agent_id):
    from src.db.connection import duckdb_rows
    conn = get_conn()

    def _query(threshold):
        if agent_id:
            return duckdb_rows(conn.execute("""
                SELECT id, content, category, importance, recall_count, last_accessed_at,
                       agent_id, visibility,
                       array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
                FROM memories
                WHERE user_id = ?
                  AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
                  AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
                ORDER BY similarity DESC LIMIT ?
            """, [query_embedding, user_id, agent_id, query_embedding, threshold, top_k * 2]))
        return duckdb_rows(conn.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility,
                   array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
            FROM memories
            WHERE user_id = ?
              AND visibility = 'shared'
              AND array_cosine_similarity(embedding, ?::FLOAT[768]) >= ?
            ORDER BY similarity DESC LIMIT ?
        """, [query_embedding, user_id, query_embedding, threshold, top_k * 2]))

    candidates = _query(SIMILARITY_THRESHOLD)
    if not candidates:
        candidates = _query(SIMILARITY_THRESHOLD_FALLBACK)

    fts_scores = _fts_search_duckdb(conn, user_id, agent_id, query=query, limit=top_k * 2)
    return _finish(candidates, top_k, conn, backend="duckdb", fts_scores=fts_scores)


# ── SQLite path ───────────────────────────────────────────────────────────────

def _retrieve_sqlite(user_id, query, query_embedding, top_k, agent_id):
    from .utils import cosine
    conn = get_conn()
    cur  = conn.cursor()
    if agent_id:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility, embedding
            FROM memories
            WHERE user_id = ? AND (visibility = 'shared' OR (visibility = 'private' AND agent_id = ?))
        """, (user_id, agent_id))
    else:
        cur.execute("""
            SELECT id, content, category, importance, recall_count, last_accessed_at,
                   agent_id, visibility, embedding
            FROM memories WHERE user_id = ? AND visibility = 'shared'
        """, (user_id,))
    rows = cur.fetchall()
    cur.close()

    def _filter(threshold):
        result = []
        for row in rows:
            raw_emb = row["embedding"]
            if raw_emb is None:
                continue
            sim = cosine(query_embedding, json.loads(raw_emb))
            if sim >= threshold:
                d = dict(row)
                d["similarity"] = sim
                result.append(d)
        return result

    candidates = _filter(SIMILARITY_THRESHOLD)
    if not candidates:
        candidates = _filter(SIMILARITY_THRESHOLD_FALLBACK)

    fts_scores = _fts_search_sqlite(conn, user_id, agent_id, query=query, limit=top_k * 2)
    return _finish(candidates, top_k, conn, backend="sqlite", fts_scores=fts_scores)
