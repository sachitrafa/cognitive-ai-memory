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
SUBJECT_SIM_THRESHOLD = 0.60  # subject spans below this → different entities → new
SUBJECT_WORDS = 2             # leading words used as subject proxy (1 = subject noun, 2 = noun+verb pair)


def _same_subject(text1: str, text2: str) -> bool:
    """
    Return True if the two sentences are likely about the same entity.

    Embeds the leading words of each sentence and compares them semantically.
    Uses the embedding model — no hardcoded word lists or grammar rules — so it
    generalises to any sentence format or language.
    """
    from src.services.embed import embed as _embed
    s1 = " ".join(text1.split()[:SUBJECT_WORDS])
    s2 = " ".join(text2.split()[:SUBJECT_WORDS])
    return _cosine(_embed(s1), _embed(s2)) >= SUBJECT_SIM_THRESHOLD


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
    Fallback contradiction detection (no spaCy).
    Detects three patterns:
    1. Polarity patterns (regex)
    2. Negation flip: "X verb Y" vs "X did not/never verb Y"
    3. Number conflict: same context, different 3-4 digit numbers
    """
    existing_lower = existing_text.lower()
    incoming_lower = incoming_text.lower()

    # ── 1. Polarity patterns ─────────────────────────────────────────────
    for positive_pattern, negative_pattern in _CONTRADICTION_PATTERNS:
        if re.search(positive_pattern, existing_lower) and re.search(negative_pattern, incoming_lower):
            return True
        if re.search(negative_pattern, existing_lower) and re.search(positive_pattern, incoming_lower):
            return True

    # ── 2. Explicit negation flip ─────────────────────────────────────────
    negation_markers = (r'\bnot\b', r'\bnever\b', r'\bno\b', r"\bdidn't\b",
                        r"\bwasn't\b", r"\bweren't\b", r"\bhasn't\b", r"\bhaven't\b")
    e_negated = any(re.search(p, existing_lower) for p in negation_markers)
    i_negated = any(re.search(p, incoming_lower) for p in negation_markers)
    if e_negated != i_negated:
        # One sentence is negated and the other is not — check they share context
        stop = {"the", "a", "an", "in", "on", "at", "of", "and", "or", "was",
                "is", "were", "are", "to", "it", "its", "for", "that", "this",
                "not", "never", "no"}
        words_e = {w.strip('.,;:') for w in existing_lower.split() if w not in stop}
        words_i = {w.strip('.,;:') for w in incoming_lower.split() if w not in stop}
        if len(words_e & words_i) >= 3:
            return True

    # ── 3. Number conflict in similar context ─────────────────────────────
    nums_e = set(re.findall(r'\b\d{3,4}\b', existing_text))
    nums_i = set(re.findall(r'\b\d{3,4}\b', incoming_text))
    if nums_e and nums_i and nums_e != nums_i:
        stop = {"the", "a", "an", "in", "on", "at", "of", "and", "or", "was",
                "is", "were", "are", "to", "it", "its", "for", "that", "this"}
        words_e = {w.lower() for w in existing_text.split() if w.lower() not in stop and not w.isdigit()}
        words_i = {w.lower() for w in incoming_text.split() if w.lower() not in stop and not w.isdigit()}
        if len(words_e & words_i) >= 4:
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

    # Different subjects = facts about different entities — never merge across them.
    # Uses embedding comparison on leading words, so it generalises to any format.
    if not _same_subject(match["content"], content):
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