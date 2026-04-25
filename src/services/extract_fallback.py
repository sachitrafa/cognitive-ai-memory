import re

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
    Fallback categorization without spaCy:
    Simple heuristics to classify as fact vs assumption
    """
    text_lower = text.lower().strip()
    
    # Commands/imperatives typically start with verbs or "please"
    imperative_patterns = [
        r'^(please|use|try|do|don\'t|make|create|add|remove|delete|update)',
        r'^(convert|transform|change|modify|fix|help|show|tell)',
        r'^(install|run|execute|start|stop|restart|configure)'
    ]
    
    for pattern in imperative_patterns:
        if re.match(pattern, text_lower):
            return "assumption"
    
    # Default to fact for declarative statements
    return "fact"