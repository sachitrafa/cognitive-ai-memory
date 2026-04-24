import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from src.services.extract import is_question, categorize
from src.services.embed import embed
from src.services.decay import compute_strength
from src.services.resolve import resolve
from src.db.connection import get_backend, get_conn, emb_to_db, duckdb_rows

load_dotenv()

router = APIRouter()

DEFAULT_IMPORTANCE = 0.5

_TS = {
    "postgres": "NOW()",
    "duckdb":   "now()",
    "sqlite":   "datetime('now')",
}


class MemoryRequest(BaseModel):
    userId: str
    content: str
    importance: float = DEFAULT_IMPORTANCE


class UpdateMemoryRequest(BaseModel):
    content: str
    importance: float = DEFAULT_IMPORTANCE


def _parse_dt(value) -> datetime:
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return datetime.now(timezone.utc)


def _exec(conn, cur, backend: str, sql_pg: str, sql_other: str, params):
    if backend == "postgres":
        cur.execute(sql_pg, params)
    elif backend == "duckdb":
        conn.execute(sql_other.replace("{TS}", _TS["duckdb"]), params)
    else:
        cur.execute(sql_other.replace("{TS}", _TS["sqlite"]), params)


# ── POST /memories ─────────────────────────────────────────────────────────────

@router.post("/memories")
def add_memory(req: MemoryRequest):
    if is_question(req.content):
        raise HTTPException(status_code=422, detail="Questions are not stored as memories.")

    req.importance = max(0.0, min(1.0, req.importance))

    category   = categorize(req.content)
    embedding  = embed(req.content)
    backend    = get_backend()
    conn       = get_conn()
    cur        = conn.cursor()

    try:
        resolution    = resolve(req.userId, req.content, embedding, conn)
        action        = resolution["action"]
        final_content = resolution["content"]
        existing      = resolution["existing"]

        if action == "reinforce":
            _exec(conn, cur, backend,
                "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW() WHERE id = %s RETURNING id",
                "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = {TS} WHERE id = ?",
                (existing["id"],) if backend != "duckdb" else [existing["id"]])
            memory_id = existing["id"]
            category  = existing["category"]

        elif action in ("replace", "merge"):
            new_embedding = embed(final_content)
            new_emb_str   = emb_to_db(new_embedding, backend)
            new_category  = categorize(final_content)
            try:
                _exec(conn, cur, backend,
                    "UPDATE memories SET content = %s, embedding = %s::vector, category = %s, recall_count = recall_count + 1, last_accessed_at = NOW() WHERE id = %s RETURNING id",
                    "UPDATE memories SET content = ?, embedding = ?, category = ?, recall_count = recall_count + 1, last_accessed_at = {TS} WHERE id = ?",
                    (final_content, new_emb_str, new_category, existing["id"]))
                memory_id = existing["id"]
                category  = new_category
            except Exception:
                conn.rollback()
                _exec(conn, cur, backend,
                    "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW() WHERE user_id = %s AND content = %s RETURNING id",
                    "UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = {TS} WHERE user_id = ? AND content = ?",
                    (req.userId, final_content))
                memory_id = existing["id"]
                category  = existing["category"]

        else:  # "new"
            emb_str = emb_to_db(embedding, backend)
            if backend == "postgres":
                cur.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = memories.recall_count + 1, last_accessed_at = NOW()
                    RETURNING id
                """, (req.userId, final_content, category, req.importance, emb_str))
                row = cur.fetchone()
                memory_id = row[0] if row else None
            elif backend == "duckdb":
                conn.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = recall_count + 1, last_accessed_at = now()
                """, [req.userId, final_content, category, req.importance, emb_str])
                result    = conn.execute("SELECT id FROM memories WHERE user_id = ? AND content = ?", [req.userId, final_content])
                row       = result.fetchone()
                memory_id = row[0] if row else None
            else:  # sqlite
                cur.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                """, (req.userId, final_content, category, req.importance, emb_str))
                memory_id = cur.lastrowid

        conn.commit()
    finally:
        cur.close()
        conn.close()

    return {
        "stored":   1,
        "id":       memory_id,
        "content":  final_content,
        "category": category,
        "action":   action,
    }


# ── PUT /memories/{id} ─────────────────────────────────────────────────────────

@router.put("/memories/{memory_id}")
def update_memory(memory_id: int, req: UpdateMemoryRequest):
    req.importance = max(0.0, min(1.0, req.importance))
    category  = categorize(req.content)
    embedding = embed(req.content)
    backend   = get_backend()
    emb_str   = emb_to_db(embedding, backend)
    conn      = get_conn()
    cur       = conn.cursor()

    try:
        if backend == "postgres":
            cur.execute("""
                UPDATE memories
                SET content = %s, embedding = %s::vector, category = %s, importance = %s,
                    recall_count = recall_count + 1, last_accessed_at = NOW()
                WHERE id = %s
                RETURNING id, content, category, importance
            """, (req.content, emb_str, category, req.importance, memory_id))
            row = cur.fetchone()
        elif backend == "duckdb":
            conn.execute("""
                UPDATE memories
                SET content = ?, embedding = ?, category = ?, importance = ?,
                    recall_count = recall_count + 1, last_accessed_at = now()
                WHERE id = ?
            """, [req.content, emb_str, category, req.importance, memory_id])
            result = conn.execute("SELECT id, content, category, importance FROM memories WHERE id = ?", [memory_id])
            row    = result.fetchone()
        else:  # sqlite
            cur.execute("""
                UPDATE memories
                SET content = ?, embedding = ?, category = ?, importance = ?,
                    recall_count = recall_count + 1, last_accessed_at = datetime('now')
                WHERE id = ?
            """, (req.content, emb_str, category, req.importance, memory_id))
            cur.execute("SELECT id, content, category, importance FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
        conn.commit()
    finally:
        cur.close()
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")

    return {"updated": 1, "id": row[0], "content": row[1], "category": row[2], "importance": row[3]}


# ── GET /memories ──────────────────────────────────────────────────────────────

@router.get("/memories")
# READ-ONLY — must never update recall_count or last_accessed_at.
# Used by the /ui browser; bumping counts here would corrupt decay scores.
def list_memories(
    userId: str = Query(..., description="User whose memories to list"),
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = Query(None),
):
    backend = get_backend()
    conn    = get_conn()
    cur     = conn.cursor()

    try:
        if backend == "postgres":
            from psycopg2.extras import RealDictCursor
            cur.close()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            sql    = "SELECT id, content, category, importance, recall_count, last_accessed_at, created_at FROM memories WHERE user_id = %s"
            params = [userId]
            if category:
                sql += " AND category = %s"
                params.append(category)
            sql += " ORDER BY last_accessed_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        else:
            sql    = "SELECT id, content, category, importance, recall_count, last_accessed_at, created_at FROM memories WHERE user_id = ?"
            params = [userId]
            if category:
                sql += " AND category = ?"
                params.append(category)
            sql += " ORDER BY last_accessed_at DESC LIMIT ?"
            params.append(limit)
            result = conn.execute(sql, params)
            rows   = duckdb_rows(result)
    finally:
        cur.close()
        conn.close()

    memories = []
    for m in rows:
        strength = compute_strength(
            last_accessed_at=_parse_dt(m["last_accessed_at"]),
            recall_count=m["recall_count"],
            importance=m["importance"],
            category=m["category"],
        )
        memories.append({
            "id":               m["id"],
            "content":          m["content"],
            "category":         m["category"],
            "importance":       round(m["importance"], 4),
            "recall_count":     m["recall_count"],
            "strength":         round(strength, 4),
            "last_accessed_at": str(m["last_accessed_at"]),
            "created_at":       str(m["created_at"]),
        })

    return {"total": len(memories), "memories": memories}


# ── DELETE /memories/{id} ──────────────────────────────────────────────────────

@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: int):
    backend = get_backend()
    conn    = get_conn()
    cur     = conn.cursor()

    try:
        if backend == "postgres":
            cur.execute("DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,))
            row = cur.fetchone()
        else:
            cur.execute("SELECT id FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
            if row:
                cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")

    return {"deleted": 1, "id": memory_id}
