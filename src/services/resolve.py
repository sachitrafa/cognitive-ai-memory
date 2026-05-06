"""
Semantic deduplication for POST /memories.

Detects near-duplicate memories via cosine similarity and applies one of:
  - reinforce : sim ≥ 0.85  — paraphrase, bump recall_count only
  - replace   : 0.65–0.85 + contradiction detected — overwrite with incoming
  - merge     : 0.65–0.85 + no contradiction — entity-append to existing
  - new       : sim < 0.65  — genuinely distinct, plain INSERT
"""

import json
from src.services.extract import _nlp
from src.services.utils import cosine as _cosine
from src.db.connection import get_backend

DEDUP_THRESHOLD     = 0.65   # below → always new memory
REINFORCE_THRESHOLD = 0.85   # at or above → reinforce (near-identical paraphrase)
SUBJECT_SIM_THRESHOLD = 0.60  # subject spans below this → different entities → new
SUBJECT_WORDS = 2             # leading words used as subject proxy (1 = subject noun, 2 = noun+verb pair)


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


_POSITIVE_VERBS = {
    "love", "like", "prefer", "enjoy", "use", "want", "start",
    "adopt", "recommend", "favor", "support", "trust", "appreciate",
}
_NEGATIVE_VERBS = {
    "hate", "dislike", "avoid", "stop", "refuse", "abandon",
    "reject", "distrust", "dislike", "despise",
}


def _polarity(doc) -> int:
    """
    Return +1 (positive), -1 (negative), or 0 (neutral) for a doc.
    Uses root verb lemma + negation detection — no sentiment model needed.
    """
    for token in doc:
        # Use both lemma and raw text to handle spaCy lemmatization bugs
        # e.g. "hates" → lemma "hat" (wrong) but text.rstrip("s") → "hate"
        lemma = token.lemma_.lower()
        raw = token.text.lower().rstrip("s")  # crude but catches loves/hates/likes/dislikes
        is_negated = any(child.dep_ == "neg" for child in token.children)

        if lemma in _POSITIVE_VERBS or raw in _POSITIVE_VERBS:
            return -1 if is_negated else +1
        if lemma in _NEGATIVE_VERBS or raw in _NEGATIVE_VERBS:
            return +1 if is_negated else -1
    return 0


def detect_contradiction(existing_text: str, incoming_text: str) -> bool:
    """
    Return True if the incoming text contradicts the existing one.
    Detects three types of contradiction:

    1. Polarity flip  — positive verb ↔ negative verb (love→hate, use→avoid)
    2. Negation flip  — same root verb, negated in one sentence but not the other
                        e.g. "appeared in" vs "did not appear in"
    3. Number conflict — same context words but different 3-4 digit numbers
                         e.g. "released in 2005" vs "released in 2012"
    """
    import re as _re

    if _nlp is None:
        return False

    doc_e = _nlp(existing_text)
    doc_i = _nlp(incoming_text)

    # ── 1. Polarity flip ─────────────────────────────────────────────────
    existing_pol = _polarity(doc_e)
    incoming_pol = _polarity(doc_i)
    if existing_pol != 0 and incoming_pol != 0 and existing_pol != incoming_pol:
        return True

    # ── 2. Negation flip ──────────────────────────────────────────────────
    # Pass A: shared ROOT verb with opposite negation ("appeared" vs "did not appear")
    def _verb_negation_map(doc) -> dict[str, bool]:
        result = {}
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl", "ccomp"):
                negated = any(c.dep_ == "neg" for c in token.children)
                result[token.lemma_.lower()] = negated
        return result

    neg_e = _verb_negation_map(doc_e)
    neg_i = _verb_negation_map(doc_i)
    for verb, e_neg in neg_e.items():
        if verb in neg_i and neg_i[verb] != e_neg:
            return True

    # Pass B: whole-sentence negation asymmetry with shared context words
    # Catches "Armstrong walked on Moon" vs "Armstrong was NOT involved in Moon landings"
    # even when the two sentences use different verbs.
    e_negated = any(c.dep_ == "neg" for c in doc_e)
    i_negated = any(c.dep_ == "neg" for c in doc_i)
    if e_negated != i_negated:
        _stop = {"the", "a", "an", "in", "on", "at", "of", "and", "or", "was",
                 "is", "were", "are", "to", "it", "its", "for", "that", "this",
                 "not", "never", "no"}
        words_e = {t.lemma_.lower() for t in doc_e
                   if not t.is_stop and t.pos_ not in ("PUNCT", "NUM") and t.lemma_.lower() not in _stop}
        words_i = {t.lemma_.lower() for t in doc_i
                   if not t.is_stop and t.pos_ not in ("PUNCT", "NUM") and t.lemma_.lower() not in _stop}
        if len(words_e & words_i) >= 2:
            return True

    # ── 3. Number conflict in similar context ────────────────────────────
    nums_e = set(_re.findall(r'\b\d{3,4}\b', existing_text))
    nums_i = set(_re.findall(r'\b\d{3,4}\b', incoming_text))
    if nums_e and nums_i and nums_e != nums_i:
        # Only flag as contradiction when both sentences share enough non-numeric
        # context (≥ 4 words) to be talking about the same thing.
        stop = {"the", "a", "an", "in", "on", "at", "of", "and", "or", "was",
                "is", "were", "are", "to", "it", "its", "for", "that", "this"}
        words_e = {w.lower() for w in existing_text.split() if w.lower() not in stop and not w.isdigit()}
        words_i = {w.lower() for w in incoming_text.split() if w.lower() not in stop and not w.isdigit()}
        if len(words_e & words_i) >= 4:
            return True

    return False


def merge_entities(existing_text: str, incoming_text: str) -> str:
    """
    Append entities/noun-chunks from incoming that are absent from existing.
    Returns the merged string, or existing_text unchanged if nothing new found.
    Falls back to returning existing_text unchanged if spaCy is unavailable.
    """
    if _nlp is None:
        return existing_text
    existing_lower = existing_text.lower()
    incoming_doc   = _nlp(incoming_text)

    # Layer 1: named entities
    candidates = [ent.text for ent in incoming_doc.ents]
    # Layer 2: noun chunks
    candidates += [chunk.text for chunk in incoming_doc.noun_chunks]
    # Layer 3: capitalised tokens (catches tech names like MongoDB, Spring, Vue)
    candidates += [
        tok.text for tok in incoming_doc
        if tok.text[0].isupper() and not tok.is_stop and len(tok.text) > 2
    ]

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


def _same_subject(text1: str, text2: str) -> bool:
    """
    Return True if the two sentences are likely about the same entity.

    Embeds the leading words of each sentence and compares them semantically.
    Uses the embedding model — no hardcoded word lists or grammar rules — so it
    generalises to any sentence format or language.

    "Sachit uses DuckDB"          vs "YourMemory uses DuckDB"    → False
    "The YourMemory project uses" vs "YourMemory stores"         → True
    "YourMemory's decay function" vs "YourMemory uses"           → True
    """
    from src.services.embed import embed as _embed
    s1 = " ".join(text1.split()[:SUBJECT_WORDS])
    s2 = " ".join(text2.split()[:SUBJECT_WORDS])
    return _cosine(_embed(s1), _embed(s2)) >= SUBJECT_SIM_THRESHOLD


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

    # Check contradiction first — even near-identical sentences can be opposites
    # e.g. "dislike JavaScript" vs "love JavaScript" → sim ~0.92 but must replace
    if detect_contradiction(match["content"], content):
        return {"action": "replace", "content": content, "existing": match}

    if sim >= REINFORCE_THRESHOLD:
        return {"action": "reinforce", "content": match["content"], "existing": match}

    merged = merge_entities(match["content"], content)
    if merged == match["content"]:
        # No new entities found — treat as paraphrase
        return {"action": "reinforce", "content": match["content"], "existing": match}

    return {"action": "merge", "content": merged, "existing": match}
