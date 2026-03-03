import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from .embed import embed
from .decay import compute_strength

load_dotenv()

# Memories below this are excluded from results entirely
SIMILARITY_THRESHOLD = 0.50
# Memories must exceed this to get recall_count reinforced
REINFORCE_THRESHOLD = 0.75


def get_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def retrieve(user_id: str, query: str, top_k: int = 5) -> dict:
    """
    1. Embed the query.
    2. Find candidates by cosine similarity, keeping only those above threshold.
    3. Score each candidate: similarity × Ebbinghaus strength.
    4. Update recall_count only for memories that passed the threshold.
    5. Return context string + structured list.
    """
    query_embedding = embed(query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Fetch candidates with their cosine similarity score (1 - distance)
    # Only retrieve rows that exceed the similarity threshold at the DB level
    cur.execute("""
        SELECT
            id, content, category, importance, recall_count, last_accessed_at,
            1 - (embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE user_id = %s
          AND 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, user_id, embedding_str, SIMILARITY_THRESHOLD, embedding_str, top_k * 2))

    candidates = cur.fetchall()

    if not candidates:
        cur.close()
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}

    # Filter by similarity threshold, then score = similarity × strength
    scored = []
    for m in candidates:
        if m["similarity"] < SIMILARITY_THRESHOLD:
            continue
        strength = compute_strength(
            last_accessed_at=m["last_accessed_at"],
            recall_count=m["recall_count"],
            importance=m["importance"],
        )
        scored.append({**m, "strength": strength, "score": m["similarity"] * strength})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    if not top:
        cur.close()
        conn.close()
        return {"memoriesFound": 0, "context": "", "memories": []}

    # Only reinforce memories that are genuinely relevant (high similarity)
    relevant_ids = [m["id"] for m in top if m["similarity"] >= REINFORCE_THRESHOLD]
    if relevant_ids:
        cur.execute("""
            UPDATE memories
            SET recall_count     = recall_count + 1,
                last_accessed_at = NOW()
            WHERE id = ANY(%s)
        """, (relevant_ids,))
    conn.commit()

    facts       = [m for m in top if m["category"] == "fact"]
    assumptions = [m for m in top if m["category"] == "assumption"]

    # Build context string with sections
    context_parts = []
    if facts:
        context_parts.append("[Facts]\n" + "\n".join(m["content"] for m in facts))
    if assumptions:
        context_parts.append("[Assumptions]\n" + "\n".join(m["content"] for m in assumptions))
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
                "importance": round(m["importance"], 4),
                "strength":   round(m["strength"], 4),
                "similarity": round(m["similarity"], 4),
                "score":      round(m["score"], 4),
            }
            for m in top
        ],
    }
