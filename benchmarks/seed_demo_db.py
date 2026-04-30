"""
Seed a clean demo database for the YourMemory demo video.

Creates ~/.yourmemory/demo.duckdb with 18 realistic developer memories
across four categories, varied ages, and varied recall counts — so the
dashboard scene shows a full spectrum: strong, fading, and near-prune bars.

Usage:
    python benchmarks/seed_demo_db.py

Then start YourMemory pointing at the demo DB:
    YOURMEMORY_DB=~/.yourmemory/demo.duckdb yourmemory

Dashboard:  http://localhost:3033/ui?userId=sachit
"""

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEMO_DB = os.path.expanduser("~/.yourmemory/demo.duckdb")
os.environ["YOURMEMORY_DB"] = DEMO_DB

from src.services.embed import embed
from src.db.connection import get_conn
from src.db.migrate import migrate

USER_ID = "sachit"

# ── Memory definitions ────────────────────────────────────────────────────────
# Each entry: (content, category, importance, days_old, recall_count, agent_id)
# days_old + recall_count together control the visible strength bar.
#
# Strength tiers visible in dashboard:
#   strong  (≥ 50%)  → days_old 0–7,  high recall or high importance
#   fading  (5–50%)  → days_old 10–20, low recall
#   prune   (< 10%)  → days_old 25+,   zero recall, low importance

MEMORIES = [
    # ── STRONG (green bars) ──────────────────────────────────────────────
    (
        "Sachit uses FastAPI with Python and type hints for all web APIs.",
        "fact", 0.9, 1, 6, "user",
    ),
    (
        "Sachit's project uses DuckDB locally and PostgreSQL + pgvector in production.",
        "fact", 0.9, 2, 4, "user",
    ),
    (
        "Sachit deploys on Fly.io — migrated from Railway in April 2026.",
        "fact", 0.85, 0, 5, "user",
    ),
    (
        "Sachit prefers pytest with the -v flag for all Python tests.",
        "fact", 0.9, 3, 3, "user",
    ),
    (
        "Pagination fixed the 30s timeout on /users — was loading 50k rows at once, now 200ms.",
        "strategy", 0.9, 5, 4, "user",
    ),
    (
        "YourMemory MCP server runs at localhost:3033 — dashboard at /ui.",
        "fact", 0.85, 1, 3, "user",
    ),

    # ── MEDIUM (yellow/amber bars) ────────────────────────────────────────
    (
        "Sachit uses Tailwind CSS for all frontend styling — no CSS-in-JS.",
        "fact", 0.8, 10, 1, "user",
    ),
    (
        "The decay job runs every 24h via APScheduler — prune threshold is strength < 0.05.",
        "fact", 0.75, 8, 1, "user",
    ),
    (
        "Redis caching failed because NumPy arrays aren't JSON serialisable — use pickle or msgpack.",
        "failure", 0.8, 7, 2, "user",
    ),
    (
        "Sachit's preferred commit style: imperative mood, one sentence, no Co-Authored-By lines.",
        "fact", 0.8, 12, 1, "user",
    ),
    (
        "Graph BFS expansion (depth=2) surfaces related memories that keyword search misses.",
        "strategy", 0.75, 9, 1, "user",
    ),

    # ── FADING (orange bars) ──────────────────────────────────────────────
    (
        "OAuth redirect URI must match exactly — staging used app-staging.example.com not app.example.com.",
        "failure", 0.7, 22, 0, "user",
    ),
    (
        "Docker build cache was causing stale env vars — added --no-cache to the CI step.",
        "failure", 0.65, 18, 0, "user",
    ),
    (
        "Sachit is considering adding a Neo4j backend option for larger graph traversals.",
        "assumption", 0.5, 20, 0, "user",
    ),

    # ── NEAR PRUNE (red bars) ──────────────────────────────────────────────
    (
        "Tried Celery for background jobs — too heavy, switched to APScheduler.",
        "failure", 0.4, 35, 0, "user",
    ),
    (
        "Considered using Pinecone for vector storage — decided against it (cost + latency).",
        "assumption", 0.35, 40, 0, "user",
    ),
    (
        "Initial spaCy model was en_core_web_lg — switched to en_core_web_sm for startup speed.",
        "fact", 0.3, 45, 0, "user",
    ),

    # ── AGENT MEMORY (shows agent tab in dashboard) ───────────────────────
    (
        "coding-agent uses staging DB for all integration tests — never touches prod.",
        "fact", 0.8, 3, 2, "coding-agent",
    ),
]


def seed():
    print(f"Seeding demo DB: {DEMO_DB}")
    print("Running migrations...")
    migrate()

    conn = get_conn()

    # Wipe existing demo data
    conn.execute(f"DELETE FROM memories WHERE user_id = '{USER_ID}'")
    conn.execute(f"DELETE FROM user_activity WHERE user_id = '{USER_ID}'")

    now = datetime.now(timezone.utc)

    print(f"Embedding and inserting {len(MEMORIES)} memories...\n")

    for i, (content, category, importance, days_old, recall_count, agent_id) in enumerate(MEMORIES):
        emb = embed(content)
        emb_str = str(list(emb))

        created_at    = now - timedelta(days=days_old + 1)
        last_accessed = now - timedelta(days=days_old)

        conn.execute("""
            INSERT INTO memories
                (user_id, content, category, importance, embedding,
                 recall_count, created_at, last_accessed_at, agent_id, visibility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'shared')
        """, [
            USER_ID, content, category, importance, emb_str,
            recall_count, created_at, last_accessed, agent_id,
        ])

        bar_len = 20
        filled  = int(bar_len * (i + 1) / len(MEMORIES))
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {i+1:>2}/{len(MEMORIES)}  {category:<12}  {content[:55]}...")

    # Seed activity log (past 30 active days)
    for d in range(30):
        active_date = (now - timedelta(days=d)).date()
        conn.execute(
            "INSERT OR IGNORE INTO user_activity (user_id, active_on) VALUES (?, ?)",
            [USER_ID, active_date],
        )

    conn.commit()
    conn.close()

    print(f"\nDone. {len(MEMORIES)} memories seeded.")
    print()
    print("Start YourMemory with the demo DB:")
    print(f"  YOURMEMORY_DB={DEMO_DB} yourmemory")
    print()
    print("Then open:")
    print(f"  http://localhost:3033/ui")


if __name__ == "__main__":
    seed()
