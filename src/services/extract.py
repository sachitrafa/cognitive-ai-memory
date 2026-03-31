import re
import spacy

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "https://github.com/explosion/spacy-models/releases/download/"
         "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"],
        check=True,
    )
    _nlp = spacy.load("en_core_web_sm")

_QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}


def is_question(text: str) -> bool:
    """Return True if the text is a question — questions are not stored as memories."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first_word = re.split(r"\s+", stripped.lower())[0]
    return first_word in _QUESTION_WORDS


def categorize(text: str) -> str:
    """
    Use spaCy dependency parse to classify:
      fact       — declarative sentence with an explicit subject
      assumption — imperative sentence with no subject (command/instruction)
    """
    doc = _nlp(text)
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
    return "fact" if has_subject else "assumption"
