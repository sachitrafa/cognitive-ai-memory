from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.services.retrieve import retrieve
from src.services.session import (
    recall_cached, recall_cache_set, session_track, start_watchdog,
)

router = APIRouter()
start_watchdog()   # idempotent — safe to call on import


class RetrieveRequest(BaseModel):
    userId:      str
    query:       str
    topK:        int            = Field(5, ge=1, le=500)
    currentPath: Optional[str] = None   # spatial boost: current file/dir path


@router.post("/retrieve")
def retrieve_memories(req: RetrieveRequest):
    user_id = req.userId.strip().lower()

    # ── Recall throttling ──────────────────────────────────────────────────
    cached = recall_cached(user_id, req.query)
    if cached is not None:
        return cached

    result = retrieve(user_id, req.query, req.topK, current_path=req.currentPath)

    # ── Session wrap-up tracking ───────────────────────────────────────────
    session_track(user_id, [m["id"] for m in result.get("memories", [])])

    # ── Cache result ───────────────────────────────────────────────────────
    recall_cache_set(user_id, req.query, result)

    # ── Record activity ────────────────────────────────────────────────────
    try:
        from src.services.decay import record_activity
        record_activity(user_id)
    except Exception:
        pass

    return result
