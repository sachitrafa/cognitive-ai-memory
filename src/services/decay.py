import math
from datetime import datetime, timezone

# Base decay rates per category.
# Higher λ = faster decay = shorter survival time.
#
# Category survival (importance=0.5, never recalled, prune threshold=0.05):
#   fact      λ=0.16  → ~24 days
#   assumption λ=0.20 → ~19 days
#   failure   λ=0.35  → ~11 days  (environment changes, old failures go stale fast)
#   strategy  λ=0.10  → ~38 days  (successful strategies are more durable)

DECAY_RATES = {
    "fact":       0.16,
    "assumption": 0.20,
    "failure":    0.35,
    "strategy":   0.10,
}
DEFAULT_DECAY_RATE = 0.16


def compute_strength(
    last_accessed_at: datetime,
    recall_count: int,
    importance: float = 0.5,
    category: str = "fact",
) -> float:
    """
    Ebbinghaus forgetting curve with importance-modulated decay rate,
    tuned per memory category:

        base_λ      = DECAY_RATES[category]
        effective_λ = base_λ × (1 - importance × 0.8)
        strength    = importance × e^(-effective_λ × days) × (1 + recall_count × 0.2)

    Failure memories decay fastest — a rate-limit from 3 months ago is likely stale.
    Strategy memories decay slowest — successful patterns stay relevant longer.
    """
    now = datetime.now(timezone.utc)
    if last_accessed_at.tzinfo is None:
        last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)

    base_lambda = DECAY_RATES.get(category, DEFAULT_DECAY_RATE)
    days = (now - last_accessed_at).total_seconds() / 86400
    effective_lambda = base_lambda * (1 - importance * 0.8)
    strength = importance * math.exp(-effective_lambda * days) * (1 + recall_count * 0.2)

    return round(min(1.0, strength), 6)
