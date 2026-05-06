# YourMemory Benchmarks

---

## 1. Long-Context Retrieval — LongMemEval-S (28 April 2026)

**Dataset:** [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — `longmemeval_s_cleaned.json`, 500 questions each backed by ~53 haystack sessions.

**Script:** [`benchmarks/longmemeval_fullstack.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/longmemeval_fullstack.py)

**Input:** Raw user-turn text from haystack sessions (real conversation turns, not summaries).

**Metric:** `recall_all@5` (all gold sessions in top-5), `nDCG_any@5`.

**Pipeline:** cosine similarity (threshold 0.50 / 0.20 fallback) + BM25 re-rank + graph BFS expansion (depth-2).

**Model:** `multi-qa-mpnet-base-dot-v1` (retrieval-tuned, question→passage, 768 dims).

### Results

| System | Recall-all@5 | nDCG-any@5 |
|--------|:------------:|:----------:|
| **YourMemory** (full stack · BM25 + vector + graph BFS) | **84.8%** | **86.8%** |
| YourMemory (cosine-only baseline, `all-mpnet-base-v2`) | 84.0% | 86.2% |

Graph BFS expansion adds ~+0.8pp on recall-all@5 over cosine-only. The switch to `multi-qa-mpnet-base-dot-v1` (retrieval-tuned for question→passage matching) accounts for the improvement over the symmetric `all-mpnet-base-v2` baseline.

---

## 2. Long-Context Recall Accuracy — LoCoMo-10 (20 April 2026)

**Dataset:** [snap-research/LoCoMo](https://github.com/snap-research/locomo) — `locomo10.json`, 10 multi-session conversation samples spanning weeks to months.

**Script:** [`benchmarks/locomo_4way.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/locomo_4way.py) — fully reproducible. All API keys loaded from environment variables; no hardcoded credentials.

**Model:** `all-mpnet-base-v2` (symmetric similarity, 768 dims). *Note: the current default model is `multi-qa-mpnet-base-dot-v1`, which scores 55% on LoCoMo session-summary retrieval but 84.8% on LongMemEval raw-passage retrieval — see Section 1.*

**Input:** `session_summary` fields from each conversation sample — identical text fed to every system in the same order.

**Queries:** QA pairs with category in {1, 2, 3, 4}, string answers only. **1,534 QA pairs total.**

**Metric:** Recall@5 — does the correct answer appear in the top-5 retrieved chunks?

**Hit rule:** exact substring match OR ≥50% of meaningful tokens (len > 3) present in the retrieved context. Applied identically to every system.

**Isolation:** each system gets a fresh user/container per sample; cleanup (delete) is called after every sample.

### Results

| System | Configuration | Recall@5 | Hits | 95% CI | Samples completed |
|--------|---------------|:--------:|:----:|:------:|:-----------------:|
| **YourMemory** | Local HTTP server · BM25 + vector + graph + Ebbinghaus decay | **59%** | **899/1,534** | 56–61% | **10/10** |
| Zep Cloud | Thread memory · `memory.search` limit=5 | 28% | 428/1,534 | 26–30% | 10/10 |
| Supermemory | Cloud API · no rerank · limit=5 | 31%* | 470/1,534 | 28–33% | 4/10* |
| Mem0 | Cloud API · `search` limit=5 | 18%* | 272/1,534 | 16–20% | 6/10* |

\* Supermemory exhausted its free-tier search quota (10,000 queries) during sample 5. Mem0 exhausted its free-tier quota (1,000 ops) during sample 7. Their hits and percentages are computed over all 1,534 QA pairs using 0 hits for the samples not completed — the numbers are accurate for what was tested. Recall figures for these two systems would likely improve on a full run.

### Per-sample breakdown (YourMemory vs Zep — both completed all 10 samples)

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

**YourMemory leads Zep by +31 percentage points (111% relative improvement) across all 10 samples.** YourMemory led every single sample. YourMemory's hybrid BM25 + vector + knowledge graph pipeline preserves full session summaries and uses Ebbinghaus decay to rank the most recently reinforced facts highest. Zep Cloud's LLM-based fact extraction condenses sessions, which loses the specific dates, names, and events that LoCoMo QA pairs target.

---

## 3. Workflow Efficiency — Token and LLM Call Savings

**Method:** A realistic multi-session developer workflow was simulated across 3 sessions with different inputs per session. Two approaches were compared: a stateless baseline (no memory, full conversation history carried forward) and YourMemory. The benchmark script is at [`benchmarks/two_session_comparison.py`](benchmarks/two_session_comparison.py).

### Token Savings

Without memory, the context window grows O(n) — every session carries all prior conversation history regardless of relevance. YourMemory replaces that history with a compressed memory block (~76–91 tokens of top-k recalled facts).

| Metric | Baseline | YourMemory | Δ |
|--------|:--------:|:----------:|:--:|
| Session 1 context tokens | 978 | 978 | 0% |
| Session 2 context tokens | 1,170 | 843 | −27.9% |
| Session 3 context tokens | 1,170 | 843 | −27.9% |
| **Total (3 sessions)** | **3,318** | **2,664** | **−19.7%** |
| Stale tokens injected | 1,148 | 0 | −100% |
| Estimated cost — 3 sessions (claude-sonnet-4-6) | $0.018 | $0.014 | −19.7% |
| Estimated cost — 30 sessions | $0.654 | $0.104 | **−84.1%** |

Memory block size stays flat (~76–91 tokens) while baseline history grows linearly. The cost gap compounds every session.

### LLM Call Savings

Without memory the assistant has no context at the start of a new session and must ask clarifying questions before implementing anything. Each clarifying round is a full LLM call that produces zero implementation output.

| Session | Baseline LLM calls | YourMemory LLM calls | Calls saved |
|---------|:-----------------:|:--------------------:|:-----------:|
| Session 1 | 4 (0 clarify + 4 work) | 4 (0 clarify + 4 work) | 0 |
| Session 2 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| Session 3 | 5 (2 clarify + 3 work) | 4 (1 clarify + 3 work) | 1 |
| **Total** | **14** | **12** | **−14%** |

The savings grow as the memory bank fills. By session 3+, YourMemory has accumulated stack, driver, configuration, and constraints — later sessions require zero clarifying calls. Live test confirmed this pattern across 3 sessions storing 10 facts progressively.

---

## 4. Decay-Based Token Pruning

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

YourMemory uses a two-stage retrieval pipeline:

**Ranking (hybrid BM25 + vector — decay excluded):**
```
hybrid_score = 0.4 × bm25_norm + 0.6 × cosine_similarity
```

**Ebbinghaus decay (pruning and graph scoring only):**
```
strength = importance × e^(−λ_eff × days) × (1 + recall_count × 0.2)
λ_eff    = base_λ × (1 − importance × 0.8)

base_λ: fact=0.16, strategy=0.10, assumption=0.20, failure=0.35
```

Decay is intentionally excluded from the ranking formula — multiplying cosine by strength would cause old-but-valid memories to rank below newer irrelevant ones. Instead, decay governs the 24h pruning job (threshold 0.05) and graph node scores. Memories above similarity `0.75` have their `recall_count` reinforced on retrieval.

---

---

## 5. Multi-Hop Reasoning — HotpotQA (6 May 2026)

**Dataset:** [HotpotQA](https://hotpotqa.github.io/) distractor set — 7,405 questions each requiring **two** supporting facts from different Wikipedia articles.

**Script:** [`benchmarks/hotpotqa_reasoning.py`](https://github.com/sachitrafa/YourMemory/blob/main/benchmarks/hotpotqa_reasoning.py)

**Design:** Store both gold supporting facts as separate memories. Query with the original multi-hop question. Score: BOTH_FOUND (both facts in top-5), ONE_FOUND, NONE_FOUND.

**Metric:** `BOTH_FOUND@5` — both supporting facts surfaced in the top-5 results.

**Why this is hard:** Multi-hop questions require connecting facts about different entities. Example: *"What government position was held by the woman who portrayed Corliss Archer?"* — Fact 1 names Shirley Temple; Fact 2 (about Shirley Temple Black's role as Chief of Protocol) can only be found by following that bridge entity. Pure embedding similarity between the two facts is too low to connect them via Round 1 vector search alone.

**What YourMemory does:** Entity-based graph edges link memories that share named entity mentions (spaCy NER), regardless of embedding similarity. When Fact 1 is retrieved in Round 1, Fact 2 is connected via the shared entity and surfaces in the merged top-5.

### Results (200 questions)

| System | BOTH_FOUND@5 | bridge | comparison |
|--------|:------------:|:------:|:----------:|
| **YourMemory** (vector + BM25 + entity graph) | **71.5%** | **69%** | **85%** |
| YourMemory (no entity edges — similarity graph only) | 59.5% | 55% | 82% |

Entity-based graph edges add **+12 pp** on overall multi-hop coverage, with the largest gain on bridge-type questions (+14 pp) where the two facts share a bridge entity that doesn't appear in the query.

---

## Dataset References

> Maharana, A., Lee, D., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024).
> **LoCoMo: Long Context Multimodal Benchmark for Dialogue.**
> *SNAP Research.*
> GitHub: [https://github.com/snap-research/locomo](https://github.com/snap-research/locomo)

> Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018).
> **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.**
> *EMNLP 2018.*
> GitHub: [https://github.com/hotpotqa/hotpot](https://github.com/hotpotqa/hotpot)
