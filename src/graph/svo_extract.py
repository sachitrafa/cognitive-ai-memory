"""
SVO (Subject-Verb-Object) triple extraction using spaCy.
Used to create graph edges when a memory is stored.

Falls back to empty list if spaCy is unavailable.
"""

# Relation label → initial edge weight
RELATION_WEIGHT_MAP = {
    "cause":   0.8,
    "use":     0.7,
    "prefer":  0.7,
    "build":   0.7,
    "work":    0.6,
    "know":    0.6,
    "have":    0.5,
    "avoid":   0.6,
    "like":    0.6,
    "hate":    0.6,
    "love":    0.6,
    "_default": 0.5,
}


def extract_triples(text: str) -> list:
    """
    Extract (subject, predicate, object) triples from text using spaCy.

    Returns list of dicts:
      [{"subject": str, "predicate": str, "object": str, "weight": float}]

    Returns [] if spaCy is unavailable or no triples found.

    Examples:
      "Sachit uses Python at MongoDB"
      → [{"subject": "Sachit", "predicate": "uses", "object": "Python", "weight": 0.7}]

      "Sachit works at MongoDB"
      → [{"subject": "Sachit", "predicate": "works", "object": "MongoDB", "weight": 0.6}]
    """
    try:
        from src.services.extract import _nlp
        if _nlp is None:
            return []
    except Exception:
        return []

    doc = _nlp(text)
    triples = []

    for token in doc:
        # Find root verbs (main predicate of a clause)
        if token.dep_ not in ("ROOT", "relcl", "advcl", "ccomp", "xcomp"):
            continue
        if token.pos_ != "VERB":
            continue

        # Subject
        subject = None
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = _span_text(child)
                break

        # Object
        obj = None
        for child in token.children:
            if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                obj = _span_text(child)
                break
        # Also check prep → pobj
        if obj is None:
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj = _span_text(grandchild)
                            break
                    if obj:
                        break

        if not subject or not obj:
            continue

        # Skip stopwords and single-char entities
        if len(subject.strip()) <= 2 or len(obj.strip()) <= 2:
            continue

        # Predicate: lemma of verb, prefixed with negation if present
        predicate = token.lemma_.lower()
        for child in token.children:
            if child.dep_ == "neg":
                predicate = f"not_{predicate}"
                break

        weight = RELATION_WEIGHT_MAP.get(token.lemma_.lower(),
                                          RELATION_WEIGHT_MAP["_default"])

        triples.append({
            "subject":   subject,
            "predicate": predicate,
            "object":    obj,
            "weight":    weight,
        })

    return triples


def _span_text(token) -> str:
    """
    Get the meaningful text span for a token — uses subtree for
    compound nouns (e.g. 'dark mode', 'MongoDB Atlas').
    """
    # Collect compound/flat children to get full noun phrase
    parts = []
    for child in token.subtree:
        if child.dep_ in ("compound", "flat", "amod") or child == token:
            parts.append((child.i, child.text))
    if not parts:
        return token.text
    parts.sort()
    return " ".join(p[1] for p in parts)
