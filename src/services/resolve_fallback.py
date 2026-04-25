"""
Semantic deduplication for POST /memories - Fallback version without spaCy.

Detects near-duplicate memories via cosine similarity and applies one of:
  - reinforce : sim ≥ 0.85  — paraphrase, bump recall_count only
  - replace   : 0.65–0.85 + contradiction detected — overwrite with incoming
  - merge     : 0.65–0.85 + no contradiction — entity-append to existing
  - new       : sim < 0.65  — genuinely distinct, plain INSERT
"""

import json
import re
from src.services.utils import cosine as _cosine
from src.db.connection import get_backend

DEDUP_THRESHOLD     = 0.65   # below → always new memory
REINFORCE_THRESHOLD = 0.85   # at or above → reinforce (near-identical paraphrase)

# Simple contradiction detection patterns (fallback)
_CONTRADICTION_PATTERNS = [
    (r'\b(love|like|prefer|enjoy)\b', r'\b(hate|dislike|avoid)\b'),
    (r'\b(start|begin|use)\b', r'\b(stop|quit|avoid)\b'),
    (r'\b(want|need)\b', r'\b(refuse|reject)\b'),
    (r'\b(good|great|excellent)\b', r'\b(bad|terrible|awful)\b'),
    (r'\b(yes|true|correct)\b', r'\b(no|false|wrong)\b'),
]


def find_near_duplicate(user_id: str, embedding: list, conn) -> dict | None:
    """
    Return the closest existing memory if cosine similarity >= DEDUP_THRESHOLD,
    else None. Uses the caller's open connection.
    """
    backend = get_backend()

    if backend == "postgres":
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        cur = conn.cursor()
        cur.execute("""
            SELECT id, content, category, importance, recall_count,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 1
        """, (embedding_str, user_id, embedding_str))
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        sim = row[5]
        if sim < DEDUP_THRESHOLD:
            return None
        return {"id": row[0], "content": row[1], "category": row[2],
                "importance": row[3], "recall_count": row[4], "similarity": sim}

    if backend == "duckdb":
        from src.db.connection import duckdb_row
        cur = conn.execute("""
            SELECT id, content, category, importance, recall_count,
                   array_cosine_similarity(embedding, ?::FLOAT[768]) AS similarity
            FROM memories
            WHERE user_id = ?
            ORDER BY similarity DESC
            LIMIT 1
        """, [embedding, user_id])
        row = duckdb_row(cur)
        if row is None or row["similarity"] < DEDUP_THRESHOLD:
            return None
        return row

    # SQLite: numpy cosine over all user memories
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, category, importance, recall_count, embedding
        FROM memories WHERE user_id = ?
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()

    best, sim = None, -1.0
    for row in rows:
        raw = row[5] if isinstance(row, tuple) else row["embedding"]
        if raw is None:
            continue
        s = _cosine(embedding, json.loads(raw))
        if s > sim:
            sim, best = s, row
    if best is None or sim < DEDUP_THRESHOLD:
        return None
    return {"id": best[0], "content": best[1], "category": best[2],
            "importance": best[3], "recall_count": best[4], "similarity": sim}


def detect_contradiction(existing_text: str, incoming_text: str) -> bool:
    """
    Fallback contradiction detection using regex patterns.
    Return True if the incoming text contradicts the existing one.
    """
    existing_lower = existing_text.lower()
    incoming_lower = incoming_text.lower()
    
    for positive_pattern, negative_pattern in _CONTRADICTION_PATTERNS:
        # Check if existing has positive and incoming has negative
        if re.search(positive_pattern, existing_lower) and re.search(negative_pattern, incoming_lower):
            return True
        # Check if existing has negative and incoming has positive  
        if re.search(negative_pattern, existing_lower) and re.search(positive_pattern, incoming_lower):
            return True
    
    return False


def merge_entities(existing_text: str, incoming_text: str) -> str:
    """
    Fallback entity merging using simple heuristics.
    Append capitalized words and quoted strings from incoming that are absent from existing.
    Returns the merged string, or existing_text unchanged if nothing new found.
    """
    existing_lower = existing_text.lower()
    
    # Extract potential entities using simple patterns
    candidates = []
    
    # Capitalized words (potential proper nouns)
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', incoming_text)
    candidates.extend(capitalized_words)
    
    # Quoted strings
    quoted_strings = re.findall(r'"([^"]+)"', incoming_text)
    quoted_strings.extend(re.findall(r"'([^']+)'", incoming_text))
    candidates.extend(quoted_strings)
    
    # Technical terms (words with numbers, dots, underscores)
    tech_terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9._-]*[a-zA-Z0-9]\b', incoming_text)
    candidates.extend([t for t in tech_terms if '.' in t or '_' in t or any(c.isdigit() for c in t)])

    # Filter out terms already present in existing text
    new_terms = [t for t in candidates if t.lower() not in existing_lower and len(t.strip()) > 2]

    # Deduplicate while preserving order
    seen, deduped = set(), []
    for t in new_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)

    if not deduped:
        return existing_text
    if len(deduped) == 1:
        return f"{existing_text} with {deduped[0]}"
    return f"{existing_text} with {', '.join(deduped[:-1])} and {deduped[-1]}"


def resolve(user_id: str, content: str, embedding: list, conn) -> dict:
    """
    Facade: decide what to do with an incoming memory.

    Returns:
        {
          "action":   "new" | "reinforce" | "replace" | "merge",
          "content":  str,          # final content to store/update
          "existing": dict | None,  # matched row if any
        }
    """
    match = find_near_duplicate(user_id, embedding, conn)

    if match is None:
        return {"action": "new", "content": content, "existing": None}

    sim = match["similarity"]

    if sim >= REINFORCE_THRESHOLD:
        return {"action": "reinforce", "content": match["content"], "existing": match}

    # DEDUP_THRESHOLD ≤ sim < REINFORCE_THRESHOLD
    if detect_contradiction(match["content"], content):
        return {"action": "replace", "content": content, "existing": match}

    merged = merge_entities(match["content"], content)
    if merged == match["content"]:
        # No new entities found — treat as paraphrase
        return {"action": "reinforce", "content": match["content"], "existing": match}

    return {"action": "merge", "content": merged, "existing": match}