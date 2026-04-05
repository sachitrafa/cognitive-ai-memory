import re
import sys

_QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}

_IMPERATIVE_PATTERNS = [
    r'^(please|use|try|do|don\'t|make|create|add|remove|delete|update)',
    r'^(convert|transform|change|modify|fix|help|show|tell)',
    r'^(install|run|execute|start|stop|restart|configure)',
]

# Load spaCy if available — falls back to regex if model not installed yet
# Run `yourmemory-setup` once after pip install to download the model
_nlp = None
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    print(
        "YourMemory: spaCy model not found. Run `yourmemory-setup` once to install it.\n"
        "  Falling back to built-in regex categorization.",
        file=sys.stderr,
    )
except Exception:
    pass


def is_question(text: str) -> bool:
    """Return True if the text is a question — questions are not stored as memories."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first_word = re.split(r"\s+", stripped.lower())[0]
    return first_word in _QUESTION_WORDS


def categorize(text: str) -> str:
    """
    Classify text as fact or assumption.
    Uses spaCy dependency parse when available, regex heuristics otherwise.
    Run `yourmemory-setup` to enable spaCy.
    """
    if _nlp is not None:
        doc = _nlp(text)
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
        return "fact" if has_subject else "assumption"

    text_lower = text.lower().strip()
    for pattern in _IMPERATIVE_PATTERNS:
        if re.match(pattern, text_lower):
            return "assumption"
    return "fact"
