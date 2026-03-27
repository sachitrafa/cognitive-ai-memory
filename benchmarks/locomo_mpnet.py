"""
LoCoMo Recall@5 Benchmark — all-mpnet-base-v2
----------------------------------------------
Full benchmark on locomo10.json using the current YourMemory approach:
  - all-mpnet-base-v2 embeddings (sentence-transformers, in-process)
  - Ebbinghaus decay scoring: cosine_sim × importance × exp(-λ × days) × (1 + recall_count × 0.2)
  - Session summaries stored per session with real timestamps
  - Recall@5: does the correct answer appear in top-5 retrieved results?

Usage:
    python benchmarks/locomo_mpnet.py
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/yourmemory")

from src.services.embed import embed

LOCOMO_PATH  = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
TOP_K        = 5
SIM_THRESH   = 0.30
PRUNE_THRESH = 0.05
IMPORTANCE   = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def parse_date(s: str) -> datetime:
    try:
        return dateparser.parse(s, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def answer_hit(answer: str, chunks: list) -> bool:
    al  = answer.lower().strip()
    ctx = " ".join(str(c) for c in chunks).lower()
    if al in ctx:
        return True
    toks = [t for t in al.split() if len(t) > 3]
    if not toks:
        return al in ctx
    return sum(1 for t in toks if t in ctx) / len(toks) >= 0.5


# ---------------------------------------------------------------------------
# In-process memory store (mirrors production scoring)
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        self.memories = []

    def add(self, text: str, stored_at: datetime):
        self.memories.append({
            "content":      text,
            "embedding":    embed(text),
            "stored_at":    stored_at,
            "importance":   IMPORTANCE,
            "recall_count": 0,
            "last_accessed": stored_at,
        })

    def search(self, query: str, query_time: datetime) -> list:
        if not self.memories:
            return []
        q_vec = embed(query)
        scored = []
        for m in self.memories:
            sim = cosine_sim(q_vec, m["embedding"])
            if sim < SIM_THRESH:
                continue
            days = max(0, (query_time - m["last_accessed"]).total_seconds() / 86400)
            lam  = 0.16 * (1 - m["importance"] * 0.8)
            strength = m["importance"] * math.exp(-lam * days) * (1 + m["recall_count"] * 0.2)
            if strength < PRUNE_THRESH:
                continue
            scored.append((sim * strength, m["content"]))
        scored.sort(reverse=True)
        return [c for _, c in scored[:TOP_K]]

    def clear(self):
        self.memories = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    store   = MemoryStore()
    results = []
    total_hits = total_qa = 0

    for idx, sample in enumerate(data):
        conv = sample["conversation"]
        sa   = conv.get("speaker_a", "A")
        sb   = conv.get("speaker_b", "B")

        all_qa = [
            q for q in sample["qa"]
            if q.get("category") in (1, 2, 3, 4)
            and isinstance(q.get("answer", ""), str)
        ]

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1])
        )
        summaries = sample.get("session_summary", {})
        now       = datetime.now(timezone.utc)

        store.clear()
        for sk in session_keys:
            dt      = conv.get(sk + "_date_time", "")
            sd      = parse_date(dt) if dt else now
            summary = summaries.get(sk + "_summary", "")
            if summary:
                store.add(summary, stored_at=sd)

        last_dt    = conv.get(session_keys[-1] + "_date_time", "") if session_keys else ""
        query_time = parse_date(last_dt) if last_dt else now

        hits = 0
        for qa in all_qa:
            results_top = store.search(qa["question"], query_time=query_time)
            if answer_hit(qa["answer"], results_top):
                hits += 1

        pct = round(hits / len(all_qa) * 100) if all_qa else 0
        total_hits += hits
        total_qa   += len(all_qa)

        print(f"Sample {idx+1:2d} | {sa} & {sb:<20} | {hits:3d}/{len(all_qa):3d} = {pct:3d}%  ({len(store.memories)} sessions)")
        results.append({"sample": idx+1, "speakers": f"{sa} & {sb}", "hits": hits, "total": len(all_qa), "pct": pct})

    overall = round(total_hits / total_qa * 100) if total_qa else 0

    print()
    print("=" * 62)
    print(f"{'Sample':<8} {'Speakers':<28} {'Recall@5':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['sample']:<8} {r['speakers']:<28} {r['pct']:>9}%")
    print("-" * 50)
    print(f"{'TOTAL':<8} {str(total_qa)+' QA pairs':<28} {overall:>9}%")
    print("=" * 62)
    print(f"\nYourMemory Recall@5 (all-mpnet-base-v2): {overall}%")

    return overall


if __name__ == "__main__":
    run()
