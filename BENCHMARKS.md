# YourMemory Benchmarks

Evaluation of YourMemory against Mem0 (free tier) across three metrics.
All benchmarks were run on **15 March 2026**.

---

## 1. Long-Context Recall Accuracy — LoCoMo Dataset

**Dataset:** [LoCoMo](https://github.com/snap-research/locomo) (Snap Research) — a public long-context memory benchmark consisting of multi-session conversations spanning weeks to months. We used `locomo10.json` (10 conversation pairs, ~199 QA pairs each, categories 1–4).

**Method:** Session summaries were stored in both systems. Each QA pair was evaluated at the end of all sessions. A hit is recorded if the correct answer appears in the top-5 retrieved results.

**Metric:** Recall@5

| Sample | Speakers | YourMemory | Mem0 |
|--------|----------|:----------:|:----:|
| 1 | Caroline & Melanie | 40% | 35% |
| 2 | Jon & Gina | 65% | 5% |
| 3 | John & Maria | 20% | 15% |
| 4 | Joanna & Nate | 30% | 5% |
| 5 | Tim & John | 20% | 10% |
| 6 | Audrey & Andrew | 40% | 25% |
| 7 | James & John | 35% | 15% |
| 8 | Deborah & Jolene | 20% | 10% |
| 9 | Evan & Sam | 30% | 20% |
| 10 | Calvin & Dave | 45% | 45% |
| **Total** | **200 QA pairs** | **34%** | **18%** |

**YourMemory leads by +16 percentage points.** YourMemory wins 9 out of 10 samples and ties 1. Mem0's automatic fact extraction condenses session content and loses specific details (dates, names, events) that LoCoMo QA pairs target. YourMemory preserves full session summaries, retaining those details while Ebbinghaus decay keeps the most relevant content ranked highest.

---

## 2. Stale Memory Precision

**Method:** 5 contradiction pairs were constructed where an outdated fact and a newer replacement fact both exist in the memory store (e.g. "Sachit prefers React" stored 45 days ago vs "Sachit switched to Vue.js" stored 3 days ago). Both facts have equal importance scores. The system is queried and scored on whether it returns the current fact ranked above the stale one.

**Metric:** Precision — does the current fact rank above the stale one?

| Scenario | YourMemory | Mem0 |
|----------|:----------:|:----:|
| Framework preference (React → Vue) | ✓ Correct | ✗ Tie |
| Job title change | ✓ Correct | ✗ Tie |
| Database migration (MongoDB → PostgreSQL) | ✓ Correct | ✗ Tie |
| Location change | ✓ Correct | ✗ Tie |
| Project rename | ✓ Correct | ✗ Tie |
| **Precision** | **5/5 (100%)** | **0/5 (0%)** |

**YourMemory leads by +100 percentage points.** Ebbinghaus decay automatically demotes stale memories without any manual deletion or reranking. Mem0 has no temporal decay signal — when two facts share the same importance score, it cannot distinguish old from new.

---

## 3. Token Efficiency

**Method:** A synthetic set of 15 memories spanning 0–60 days was evaluated. Memories with Ebbinghaus strength below the prune threshold (0.05) are excluded from retrieval entirely. Token counts are based on the top-5 memories injected into context.

**Metric:** Context tokens injected per query

| | Baseline (no decay) | YourMemory |
|---|:-------------------:|:----------:|
| Total memories | 15 | 15 |
| Pruned memories | 0 | 3 (20%) |
| Tokens in top-5 context | 74 | 71 |
| **Token reduction** | — | **4.1%** |

Token savings compound at scale. A system with 200+ memories over 6 months will prune a significantly larger fraction, reducing noise injected into the model context and lowering API costs.

---

## Scoring Formula

YourMemory retrieval score combines semantic relevance with biological memory strength:

```
score = cosine_similarity × Ebbinghaus_strength

Ebbinghaus_strength = importance × e^(−λ_eff × days) × (1 + recall_count × 0.2)
λ_eff = 0.16 × (1 − importance × 0.8)
```

Memories below strength `0.05` are pruned entirely. Memories above similarity `0.75` have their `recall_count` reinforced on retrieval.

---

## Dataset Reference

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> **LoCoMo: Long Context Multimodal Benchmark for Dialogue.**
> *SNAP Research.*
> GitHub: [https://github.com/snap-research/locomo](https://github.com/snap-research/locomo)
