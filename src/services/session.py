"""
Shared in-process state for session wrap-up scoring and recall throttling.

Imported by both the HTTP routes (src/routes/) and the MCP tool handler
(memory_mcp.py) so both servers maintain consistent behaviour within their
own process lifetime.
"""

import os
import sys
import time
import threading
from collections import defaultdict

# ── Recall throttling ─────────────────────────────────────────────────────────
# YOURMEMORY_RECALL_COOLDOWN: seconds to cache recall results per (user, query).
# 0 = disabled (default).
RECALL_COOLDOWN: int = int(os.getenv("YOURMEMORY_RECALL_COOLDOWN", "0"))
_recall_cache: dict[str, tuple[float, dict]] = {}   # cache_key → (ts, result)


def recall_cached(user_id: str, query: str) -> dict | None:
    """Return cached result if within cooldown window, else None."""
    if RECALL_COOLDOWN <= 0:
        return None
    key = f"{user_id}:{query}"
    cached = _recall_cache.get(key)
    if cached and (time.time() - cached[0]) < RECALL_COOLDOWN:
        return cached[1]
    return None


def recall_cache_set(user_id: str, query: str, result: dict) -> None:
    if RECALL_COOLDOWN > 0:
        _recall_cache[f"{user_id}:{query}"] = (time.time(), result)


# ── Session wrap-up scoring ───────────────────────────────────────────────────
# YOURMEMORY_SESSION_IDLE: seconds of inactivity before a session is flushed.
SESSION_IDLE: int = int(os.getenv("YOURMEMORY_SESSION_IDLE", "1800"))   # 30 min

_session_hits: dict[str, set]   = defaultdict(set)   # user_id → {memory_id}
_session_last: dict[str, float] = {}                  # user_id → last recall ts
_watchdog_started = False
_watchdog_lock    = threading.Lock()


def session_track(user_id: str, memory_ids: list[int]) -> None:
    """Record recalled memory IDs for the current session."""
    now = time.time()
    # Flush if session was idle before we add new hits
    if user_id in _session_last and (now - _session_last[user_id]) >= SESSION_IDLE:
        flush_session(user_id)
    for mid in memory_ids:
        _session_hits[user_id].add(mid)
    _session_last[user_id] = now


def flush_session(user_id: str) -> None:
    """Bump recall_count for every memory recalled in the just-ended session."""
    ids = list(_session_hits.pop(user_id, set()))
    _session_last.pop(user_id, None)
    if not ids:
        return
    try:
        from src.db.connection import get_backend, get_conn
        backend = get_backend()
        conn    = get_conn()
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
                ph = ", ".join("?" * len(ids))
                conn.execute(
                    f"UPDATE memories SET recall_count = recall_count + 1 WHERE id IN ({ph})",
                    ids,
                )
            else:
                cur = conn.cursor()
                for mid in ids:
                    cur.execute(
                        "UPDATE memories SET recall_count = recall_count + 1 WHERE id = ?",
                        (mid,),
                    )
                conn.commit()
                cur.close()
        finally:
            conn.close()
        print(f"[session] wrap-up: boosted {len(ids)} memories for {user_id}", file=sys.stderr)
    except Exception as exc:
        print(f"[session] wrap-up failed: {exc}", file=sys.stderr)


def _watchdog_loop() -> None:
    """Background thread: flush sessions idle for longer than SESSION_IDLE."""
    while True:
        time.sleep(60)
        now = time.time()
        for uid, last in list(_session_last.items()):
            if now - last >= SESSION_IDLE:
                flush_session(uid)


def start_watchdog() -> None:
    """Start the session watchdog thread once (idempotent)."""
    global _watchdog_started
    with _watchdog_lock:
        if _watchdog_started:
            return
        t = threading.Thread(target=_watchdog_loop, daemon=True, name="session-watchdog")
        t.start()
        _watchdog_started = True
