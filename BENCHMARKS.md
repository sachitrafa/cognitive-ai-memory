# YourMemory Benchmarks

Evaluation of YourMemory against Mem0 and Zep Cloud across three metrics.

---

## 1. Long-Context Recall Accuracy — LoCoMo Dataset

**Dataset:** [LoCoMo](https://github.com/snap-research/locomo) (Snap Research) — a public long-context memory benchmark consisting of multi-session conversations spanning weeks to months. We used `locomo10.json` (10 conversation pairs, 1,534 QA pairs total, categories 1–4).

**Method:** Session summaries were stored in both systems. Each QA pair was evaluated at the end of all sessions. A hit is recorded if the correct answer appears in the top-5 retrieved results.

**Embedding model:** `all-mpnet-base-v2` (sentence-transformers, 768 dims, runs fully in-process — no external service required)

**Metric:** Recall@5

### vs Mem0 (free tier) — 15 March 2026

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

### vs Zep Cloud — 27 March 2026 (all-mpnet-base-v2)

| Sample | Speakers | YourMemory | Zep Cloud |
|--------|----------|:----------:|:---------:|
| 1 | Caroline & Melanie | 29% | 24% |
| 2 | Jon & Gina | 47% | 46% |
| 3 | John & Maria | 25% | 22% |
| 4 | Joanna & Nate | 38% | 18% |
| 5 | Tim & John | 31% | 24% |
| 6 | Audrey & Andrew | 33% | 20% |
| 7 | James & John | 37% | 23% |
| 8 | Deborah & Jolene | 28% | 16% |
| 9 | Evan & Sam | 32% | 18% |
| 10 | Calvin & Dave | 43% | 26% |
| **Total** | **1,534 QA pairs** | **34%** | **22%** |

**YourMemory leads by +12 percentage points (54% relative improvement).** YourMemory wins 9 out of 10 samples and ties 1. Zep Cloud uses LLM-based fact extraction per thread, which summarises and condenses content — losing the specific details (dates, names, events) that LoCoMo QA pairs target. YourMemory's Ebbinghaus decay scores full session summaries by recency and importance, preserving those details while ranking the most relevant content highest.

---

## 2. Token Efficiency

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
