# YourMemory Benchmarks

Three external datasets, two internal efficiency measurements. All scripts are reproducible — datasets and API keys loaded from environment variables, no hardcoded credentials.

---

## 1. Long-Context Retrieval — LongMemEval-S (28 April 2026)

**Dataset:** [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — `longmemeval_s_cleaned.json`, 500 questions each backed by ~53 haystack sessions.

**Script:** [`benchmarks/longmemeval_fullstack.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/longmemeval_fullstack.py)

**Input:** Raw user-turn text from haystack sessions (real conversation turns, not summaries).

**Metric:** `recall_all@5` — all gold sessions must appear in top-5 results.

**Pipeline:** cosine similarity (threshold 0.50 / 0.20 fallback) + BM25 re-rank + graph BFS expansion (depth-2).

**Model:** `multi-qa-mpnet-base-dot-v1` (retrieval-tuned, question→passage, 768 dims).

### Overall Results

| System | Recall-all@5 | Recall-any@5 | nDCG-any@5 |
|--------|:------------:|:------------:|:----------:|
| **YourMemory** (BM25 + vector + graph BFS) | **84.8%** | **95.8%** | **87.4%** |
| YourMemory (cosine-only, `all-mpnet-base-v2`) | 84.0% | — | 86.2% |

Graph BFS expansion adds ~+0.8pp on recall-all@5. The switch to `multi-qa-mpnet-base-dot-v1` (retrieval-tuned for question→passage) accounts for the improvement over the symmetric `all-mpnet-base-v2` baseline.

### Breakdown by Question Type

| Question Type | Recall-all@5 | Recall-any@5 | nDCG-any@5 |
|---------------|:------------:|:------------:|:----------:|
| knowledge-update | 97.4% | 100.0% | 94.6% |
| single-session-assistant | 96.4% | 96.4% | 94.8% |
| single-session-user | 95.7% | 95.7% | 91.5% |
| single-session-preference | 83.3% | 83.3% | 80.4% |
| multi-session | 75.9% | 97.0% | 85.6% |
| temporal-reasoning | 75.9% | 94.7% | 81.3% |

Temporal-reasoning and multi-session questions are the hardest — the system retrieves *a* correct session 95%+ of the time (recall-any), but surfacing *all* required sessions in top-5 drops to 75.9%. These are the cases most dependent on correct time-anchored linking.

---

## 2. Long-Context Recall Accuracy — LoCoMo-10 (20 April 2026)

**Dataset:** [snap-research/LoCoMo](https://github.com/snap-research/locomo) — `locomo10.json`, 10 multi-session conversation samples spanning weeks to months.

**Script:** [`benchmarks/locomo_4way.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/locomo_4way.py)

**Model:** `all-mpnet-base-v2` (symmetric similarity, 768 dims). *Note: the current default model is `multi-qa-mpnet-base-dot-v1`, which scores 55% on LoCoMo session-summary retrieval but 84.8% on LongMemEval raw-passage retrieval — see Section 1.*

**Input:** `session_summary` fields — identical text fed to every system in the same order.

**Queries:** 1,534 QA pairs across categories 1–4, string answers only.

**Metric:** Recall@5 — does the correct answer appear in the top-5 retrieved chunks?

**Hit rule:** exact substring match OR ≥50% of meaningful tokens (len > 3) present in retrieved context. Applied identically to every system.

**Isolation:** each system gets a fresh user/container per sample; cleanup runs after every sample.

### Results

| System | Configuration | Recall@5 | Hits | 95% CI | Samples |
|--------|---------------|:--------:|:----:|:------:|:-------:|
| **YourMemory** | BM25 + vector + graph + Ebbinghaus decay | **59%** | **899/1,534** | 56–61% | **10/10** |
| Zep Cloud | Thread memory · `memory.search` limit=5 | 28% | 428/1,534 | 26–30% | 10/10 |
| Supermemory | Cloud API · no rerank · limit=5 | 31%* | 470/1,534 | 28–33% | 4/10* |
| Mem0 | Cloud API · `search` limit=5 | 18%* | 272/1,534 | 16–20% | 6/10* |

\* Supermemory exhausted its free-tier quota (10,000 queries) at sample 5. Mem0 exhausted its quota (1,000 ops) at sample 7. Hits computed over all 1,534 pairs using 0 for unfinished samples — figures would likely improve on a full run.

### Per-Sample Breakdown (YourMemory vs Zep — both completed all 10 samples)

| Sample | Speakers | QA pairs | YourMemory | Zep Cloud |
|--------|----------|:--------:|:----------:|:---------:|
| 1 | Caroline & Melanie | 146 | 64% | 26% |
| 2 | Jon & Gina | 81 | 54% | 37% |
| 3 | John & Maria | 152 | 64% | 32% |
| 4 | Joanna & Nate | 199 | 57% | 26% |
| 5 | Tim & John | 178 | 68% | 27% |
| 6 | Audrey & Andrew | 123 | 58% | 28% |
| 7 | James & John | 150 | 57% | 33% |
| 8 | Deborah & Jolene | 191 | 61% | 25% |
| 9 | Evan & Sam | 156 | 56% | 26% |
| 10 | Calvin & Dave | 158 | 43% | 25% |
| **Total** | | **1,534** | **59%** | **28%** |

**YourMemory leads Zep by +31 pp (111% relative) across all 10 samples.** The gap comes from architecture: YourMemory stores full session summaries and scores on BM25 + vector + graph. Zep's LLM-based extraction condenses sessions into abstract facts, losing the specific dates, names, and events that LoCoMo QA pairs target.

---

## 3. Multi-Hop Reasoning — HotpotQA (6 May 2026)

**Dataset:** [HotpotQA](https://hotpotqa.github.io/) distractor set — 113,000 questions each requiring **two** supporting facts from **different** Wikipedia articles. Each question is paired with 8 distractor paragraphs alongside the 2 gold paragraphs.

**Script:** [`benchmarks/hotpotqa_reasoning.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/hotpotqa_reasoning.py)

**Design:** Store both gold supporting facts as separate memories. Query with the original multi-hop question. Score whether both facts appear in top-5.

**Metric:** `BOTH_FOUND@5` — both supporting facts surfaced in the top-5 results.

**Two question types:**
- **Bridge** — Fact 1 names an entity; Fact 2 is about that entity's property. The two facts have low embedding similarity because they're about different subjects on the surface.
- **Comparison** — Both facts provide a measurable attribute (date, count, age); the question asks which is larger/older/longer.

**Why pure vector retrieval fails bridge questions:** The query matches Fact 1 well but has low cosine similarity with Fact 2 — because Fact 2 is about the *answer* to Fact 1, not about anything in the original question. A system that only retrieves by similarity stops at Fact 1 and never finds Fact 2.

**What YourMemory does:** At store time, spaCy NER extracts named entities from every memory and creates graph edges between memories that share entity mentions. At query time, once Fact 1 is retrieved in Round 1, its entity links are traversed to surface Fact 2 — regardless of embedding similarity to the query.

### Results (200 questions — 166 bridge, 34 comparison)

| System | BOTH_FOUND@5 | ONE_FOUND | NONE_FOUND |
|--------|:------------:|:---------:|:----------:|
| **YourMemory** (vector + BM25 + entity graph) | **71.5%** (143/200) | 26.5% (53/200) | 2.0% (4/200) |
| YourMemory (similarity graph only — no entity edges) | 59.5% (119/200) | 38.0% (76/200) | 2.5% (5/200) |

### By Question Type

| Type | Questions | BOTH_FOUND | ONE_FOUND | NONE_FOUND |
|------|:---------:|:----------:|:---------:|:----------:|
| Bridge | 166 | **68.7%** (114) | 28.9% (48) | 2.4% (4) |
| Comparison | 34 | **85.3%** (29) | 14.7% (5) | 0% (0) |

Entity-based graph edges add **+12 pp** overall, with the largest gain on bridge questions (+14 pp). Comparison questions score higher because both facts tend to share enough vocabulary with the query to be retrieved in Round 1 without graph traversal.

The remaining 28.5% BOTH_FOUND gap on bridge questions represents cases where the bridge entity appears in neither the query nor either retrieved fact — the graph cannot connect what it never indexed.

---

## 4. Workflow Efficiency — Token and LLM Call Savings

**Method:** A multi-session developer workflow simulated across 3 sessions. Stateless baseline (full conversation history carried forward) vs YourMemory (top-k recalled facts only).

**Script:** [`benchmarks/two_session_comparison.py`](benchmarks/two_session_comparison.py)

### Token Savings

Context window grows O(n) without memory — every session carries all prior history regardless of relevance. YourMemory replaces that with a flat memory block (~76–91 tokens).

| Metric | Baseline | YourMemory | Δ |
|--------|:--------:|:----------:|:--:|
| Session 1 context tokens | 978 | 978 | 0% |
| Session 2 context tokens | 1,170 | 843 | −27.9% |
| Session 3 context tokens | 1,170 | 843 | −27.9% |
| **Total (3 sessions)** | **3,318** | **2,664** | **−19.7%** |
| Stale tokens injected | 1,148 | 0 | −100% |
| Estimated cost — 3 sessions (claude-sonnet-4-6) | $0.018 | $0.014 | −19.7% |
| Estimated cost — 30 sessions | $0.654 | $0.104 | **−84.1%** |

Memory block size stays flat while baseline grows linearly. The cost gap compounds every session.

### LLM Call Savings

Without memory, each new session starts cold — the assistant must ask clarifying questions before implementing anything.

| Session | Baseline LLM calls | YourMemory LLM calls | Saved |
|---------|:-----------------:|:--------------------:|:-----:|
| Session 1 | 4 (0 clarify + 4 work) | 4 (0 clarify + 4 work) | 0 |
| Session 2 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| Session 3 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| **Total** | **14** | **12** | **−14%** |

---

## 5. Decay-Based Token Pruning

**Method:** 15 synthetic memories spanning 0–60 days. Memories below Ebbinghaus strength threshold (0.05) are excluded from retrieval. Token counts based on top-5 memories injected into context.

| | Baseline (no decay) | YourMemory |
|---|:-------------------:|:----------:|
| Total memories | 15 | 15 |
| Pruned (strength < 0.05) | 0 | 3 (20%) |
| Tokens in top-5 context | 74 | 71 |
| **Token reduction** | — | **4.1%** |

Token savings compound at scale — a system with 200+ memories over 6 months prunes a substantially larger fraction, reducing noise in model context.

---

## Scoring Formula

**Hybrid ranking (BM25 + vector — decay excluded from ranking):**
```
hybrid_score = 0.4 × bm25_norm + 0.6 × cosine_similarity
```

**Ebbinghaus decay (pruning and graph scoring only):**
```
strength = importance × e^(−λ_eff × days) × (1 + recall_count × 0.2)
λ_eff    = base_λ × (1 − importance × 0.8)

base_λ: fact=0.16, strategy=0.10, assumption=0.20, failure=0.35
```

Decay is intentionally excluded from the ranking formula — multiplying cosine by strength would penalise old-but-valid memories below newer irrelevant ones. Instead, decay governs the 24h pruning job (threshold 0.05) and graph node scores. Memories above similarity `0.75` have their `recall_count` reinforced on retrieval.

---

## Dataset References

> Wu, X., Wang, L., Xu, T., Shi, W., & Ma, Y. (2024).
> **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.**
> *arXiv 2410.10813.*
> GitHub: [https://github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> **LoCoMo: Long Context Multimodal Benchmark for Dialogue.**
> *SNAP Research.*
> GitHub: [https://github.com/snap-research/locomo](https://github.com/snap-research/locomo)

> Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018).
> **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.**
> *EMNLP 2018.*
> GitHub: [https://github.com/hotpotqa/hotpot](https://github.com/hotpotqa/hotpot)
