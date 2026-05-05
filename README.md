<!-- mcp-name: io.github.sachitrafa/yourmemory -->
<div align="center">

# YourMemory

**Persistent memory for AI agents — built on the science of how humans remember.**

[![PyPI](https://img.shields.io/pypi/v/yourmemory?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/yourmemory/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/yourmemory?color=brightgreen)](https://pypi.org/project/yourmemory/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![LoCoMo Recall@5](https://img.shields.io/badge/LoCoMo%20Recall%405-59%25-brightgreen)](BENCHMARKS.md)
[![LongMemEval Recall@5](https://img.shields.io/badge/LongMemEval%20Recall--all%405-85%25-brightgreen)](BENCHMARKS.md)
[![Docker Build](https://img.shields.io/github/actions/workflow/status/sachitrafa/YourMemory/docker-publish.yml?branch=main&label=docker&logo=docker)](https://github.com/sachitrafa/YourMemory/actions/workflows/docker-publish.yml)

</div>

---

## The Problem

Every session, your AI assistant starts from zero. It asks the same questions, forgets your preferences, re-learns your stack. There is no memory between conversations.

**YourMemory fixes that.** It gives AI agents a persistent memory layer that works the way human memory does — important things stick, forgotten things fade, outdated facts get replaced automatically. One command to install, zero infrastructure required. Memory starts working the moment you add it to your AI client.

---

## How Well Does It Work?

### LoCoMo-10 — 1,534 QA pairs across 10 multi-session conversations

| System | Recall@5 | 95% CI |
|--------|:--------:|:------:|
| **YourMemory** (BM25 + vector + graph + decay) | **59%** | 56–61% |
| Zep Cloud | 28% | 26–30% |

> **2× better recall than Zep Cloud on the same benchmark.**

*The 59% result used `all-mpnet-base-v2`. The current default model (`multi-qa-mpnet-base-dot-v1`) scores 55% on LoCoMo session-summary retrieval — see [BENCHMARKS.md](BENCHMARKS.md) for details.*

### LongMemEval-S — 500 questions, ~53 sessions each

| System | Recall-all@5 | nDCG@5 |
|--------|:------------:|:------:|
| **YourMemory** (full stack · `multi-qa-mpnet-base-dot-v1`) | **85%** | **87%** |

Full methodology in [BENCHMARKS.md](BENCHMARKS.md). Writeup: [I built memory decay for AI agents using the Ebbinghaus forgetting curve](https://dev.to/sachit_mishra_686a94d1bb5/i-built-memory-decay-for-ai-agents-using-the-ebbinghaus-forgetting-curve-1b0e).

---

## Quick Start

**Supports Python 3.11–3.14. No Docker, no database setup, no external services.**

### Step 1 — Install

```bash
pip install yourmemory
```

### Step 2 — Get your config path

```bash
yourmemory-path
```

Prints your full executable path and a ready-to-paste config block. Copy it.

### Step 3 — Wire into your AI client

<details>
<summary><strong>Claude Code</strong></summary>

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "yourmemory"
    }
  }
}
```

Reload (`Cmd+Shift+P` → `Developer: Reload Window`).

</details>

<details>
<summary><strong>Claude Desktop</strong></summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "yourmemory"
    }
  }
}
```

Restart Claude Desktop.

</details>

<details>
<summary><strong>Cline (VS Code)</strong></summary>

VS Code doesn't inherit your shell PATH. Run `yourmemory-path` first to get the full executable path.

In Cline → **MCP Servers** → **Edit MCP Settings**:

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/full/path/to/yourmemory",
      "args": [],
      "env": { "YOURMEMORY_USER": "your_name" }
    }
  }
}
```

Restart Cline after saving.

</details>

<details>
<summary><strong>Cursor</strong></summary>

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/full/path/to/yourmemory",
      "args": [],
      "env": { "YOURMEMORY_USER": "your_name" }
    }
  }
}
```

</details>

<details>
<summary><strong>Windsurf / OpenCode / any MCP client</strong></summary>

YourMemory is a standard stdio MCP server. Use the full path from `yourmemory-path` if the client doesn't inherit shell PATH.

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/full/path/to/yourmemory",
      "env": { "YOURMEMORY_USER": "your_name" }
    }
  }
}
```

</details>

> **First start is automatic.** On the first run, YourMemory initialises your database, downloads the language model, and injects memory workflow instructions into your AI client config — no manual setup needed.

### Step 4 — Start remembering

That's it. On the first MCP start, YourMemory automatically:
- Initialises your local database at `~/.yourmemory/memories.duckdb`
- Downloads the spaCy language model in the background
- Injects the memory workflow rules into your AI client

Your AI now recalls what it learned in previous sessions, without you telling it to.

---

## Memory Dashboard

Every YourMemory instance ships with a built-in browser UI. When the MCP server is running, open:

```
http://localhost:3033/ui
```

Browse your memories by agent, filter by category, sort by strength, and see which memories are fading.

<details>
<summary><strong>What you'll see</strong></summary>

- **Strength bars** — how close each memory is to being pruned
- **Agent tabs** — switch between All / User / per-agent views
- **Category badges** — fact · strategy · assumption · failure
- **Stats** — total, strong (≥ 50%), fading (5–50%), near prune (< 10%)

</details>

---

## Ask Without Calling the API

The only memory system that can answer questions from memory **without making any LLM API call.**

Every other system (Mem0, Zep, LangMem, Cognee) follows the same pattern: retrieve → inject into context → call your LLM. YourMemory has a `ask` command that short-circuits that loop entirely for trivial factual queries.

```bash
yourmemory ask "what database does this project use"
# → YourMemory uses DuckDB locally and Postgres in production.

yourmemory ask "what port does the dashboard run on"
# → 3033

yourmemory ask "how do I deploy to kubernetes"
# → Not enough memory context to answer without Claude.
```

When memory is strong enough to answer confidently, it responds instantly — zero tokens, zero cloud cost, zero latency. When it isn't, you get a clean decline rather than a hallucinated answer.

### Why this matters

| | Mem0 / Zep / LangMem | YourMemory |
|---|---|---|
| "What port does the server run on?" | Full LLM API call | Answered instantly, $0 |
| "What database does this project use?" | Full LLM API call | Answered instantly, $0 |
| "How do I fix a k8s deployment?" | Full LLM API call | Declines cleanly → Claude |
| Privacy | Query sent to cloud | Query never leaves your machine |

---

## MCP Tools

Three tools, called by your AI automatically.

| Tool | When | What it does |
|------|------|--------------|
| `recall_memory(query, current_path?)` | Start of every task | Surfaces relevant memories ranked by similarity × strength; boosts spatially matched memories |
| `store_memory(content, importance, context_paths?)` | After learning something new | Embeds and stores with biological decay; tags optional file/dir paths for spatial recall |
| `update_memory(id, new_content)` | When a memory is outdated | Re-embeds and replaces; logs old content to audit trail |

```python
# Store with spatial context
store_memory("Sachit prefers tabs over spaces in Python", importance=0.9, category="fact",
             context_paths=["/projects/backend"])

# Next session — spatial boost applied when working in that path:
recall_memory("Python formatting", current_path="/projects/backend")
# → {"content": "Sachit prefers tabs over spaces in Python", "strength": 0.87, "score": 0.81}
```

### Categories control how fast memories fade

| Category | Survives without recall | Use case |
|----------|------------------------|----------|
| `strategy` | ~38 days | Successful patterns |
| `fact` | ~24 days | Preferences, identity |
| `assumption` | ~19 days | Inferred context |
| `failure` | ~11 days | Errors, environment-specific issues |

---

## How It Works

### Ebbinghaus Forgetting Curve

Memory strength decays exponentially — importance and recall frequency slow that decay:

```
effective_λ  = base_λ × (1 - importance × 0.8)
strength     = clamp(importance × e^(−effective_λ × active_days) × (1 + recall_count × 0.2), 0, 1)
hybrid_score = 0.4 × bm25_norm + 0.6 × cosine_similarity
```

`active_days` counts only days the user was active — vacations don't cause memory loss. Decay is used for pruning only, not ranking. Memories below strength `0.05` are pruned automatically every 24 hours.

**Session wrap-up scoring:** recalled memory IDs are tracked per session and get a recall_count boost when the session goes idle (30 min default). Set `YOURMEMORY_SESSION_IDLE` to change the window.

**Recall throttling:** identical (user, query) pairs are cached to avoid redundant retrieval within a configurable window. Set `YOURMEMORY_RECALL_COOLDOWN` (seconds, default 0 = off).

### Hybrid Retrieval: Vector + BM25 + Graph

Retrieval runs in two rounds to surface related context that vocabulary-based search misses:

**Round 1 — Hybrid search:** cosine similarity + BM25 keyword scoring, returns top-k above threshold.

**Round 2 — Graph expansion:** BFS traversal from Round 1 seeds surfaces memories that share context but not vocabulary — connected via semantic edges (cosine similarity ≥ 0.4).

```
recall("Python backend")
  Round 1 → [1] Python/MongoDB    (sim=0.61)
             [2] DuckDB/spaCy     (sim=0.19)
  Round 2 → [5] Docker/Kubernetes (sim=0.29 — below cut-off, surfaced via graph)
```

**Chain-aware pruning:** A decayed memory is kept alive if any graph neighbour is above the prune threshold. Related memories age together.

### Subject-Aware Deduplication

When storing a new memory, YourMemory compares the incoming content against the nearest existing memory. Before merging, it verifies the two memories are about the **same entity** — not just the same topic.

```
"Sachit uses DuckDB"    vs  "YourMemory uses DuckDB"
 subject: Sachit             subject: YourMemory
 → different entities → stored as two separate facts ✓

"YourMemory uses DuckDB"  vs  "YourMemory stores data in DuckDB"
 subject: YourMemory           subject: YourMemory
 → same entity → merged ✓
```

Subject comparison embeds the first two words of each sentence and compares them semantically — no hardcoded word lists, generalises to any sentence structure or language.

---

## Multi-Agent Memory

Multiple agents can share the same YourMemory instance — each with isolated private memories and controlled access to shared context.

```python
from src.services.api_keys import register_agent

result = register_agent(
    agent_id="coding-agent",
    user_id="sachit",
    can_read=["shared", "private"],
    can_write=["shared", "private"],
)
# → result["api_key"]  — ym_xxxx, shown once only
```

Pass `api_key` to any MCP call to authenticate as an agent:

```python
store_memory(content="Staging uses self-signed cert — skip SSL verify",
             importance=0.7, category="failure",
             api_key="ym_xxxx", visibility="private")

recall_memory(query="staging SSL", api_key="ym_xxxx")
# → returns shared memories + this agent's private memories
# → other agents see shared only
```

---

## Stack

| Component | Role |
|-----------|------|
| **DuckDB** | Default vector DB — zero setup, native cosine similarity |
| **NetworkX** | Default graph backend — persists at `~/.yourmemory/graph.pkl` |
| **sentence-transformers** | Local embeddings (`multi-qa-mpnet-base-dot-v1`, 768 dims) |
| **spaCy** | Local NLP for deduplication and SVO triple extraction |
| **APScheduler** | Automatic 24h decay job |
| **PostgreSQL + pgvector** | Optional — for teams or large datasets |
| **Neo4j** | Optional graph backend — `pip install 'yourmemory[neo4j]'` |

<details>
<summary><strong>PostgreSQL setup (optional)</strong></summary>

```bash
pip install yourmemory[postgres]
```

Create a `.env` file:

```bash
DATABASE_URL=postgresql://YOUR_USER@localhost:5432/yourmemory
```

**macOS**
```bash
brew install postgresql@16 pgvector && brew services start postgresql@16
createdb yourmemory
```

**Ubuntu / Debian**
```bash
sudo apt install postgresql postgresql-contrib postgresql-16-pgvector
createdb yourmemory
```

</details>

---

## Architecture

```
Claude / Cline / Cursor / Any MCP client
    │
    ├── recall_memory(query, current_path?, api_key?)
    │       └── throttle check (YOURMEMORY_RECALL_COOLDOWN)
    │               embed → vector similarity (Round 1)
    │               → graph BFS expansion  (Round 2)
    │               → score = sim × strength → top-k
    │               → spatial boost (+0.08) if current_path matches context_paths
    │               → session tracking → recall_count bump on session end
    │
    ├── store_memory(content, importance, category?, context_paths?, visibility?, api_key?)
    │       └── question? → reject
    │               subject-aware dedup → same entity? merge/reinforce : new
    │               embed() → INSERT → index_memory() → graph node + edges
    │               record_activity(user_id) → active days log
    │
    └── update_memory(id, new_content, importance)
            └── log old content → memory_history (audit trail)
                    embed(new_content) → UPDATE → refresh graph node

  Vector DB (Round 1)             Graph DB (Round 2)
  DuckDB (default)                NetworkX (default)
    memories.duckdb                 graph.pkl
    ├── embedding FLOAT[768]        ├── nodes: memory_id, strength
    ├── importance FLOAT            └── edges: sim × verb_weight ≥ 0.4
    ├── recall_count INTEGER
    ├── context_paths JSON        Neo4j (opt-in)
    ├── visibility VARCHAR          └── bolt://localhost:7687
    ├── agent_id VARCHAR
    user_activity (active days log)
    memory_history (supersession audit)
```

---

## Contributing

PRs are welcome. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the people who have already improved YourMemory.

---

## Dataset Reference

Benchmarks use the [LoCoMo](https://github.com/snap-research/locomo) dataset by Snap Research.

> Maharana et al. (2024). *LoCoMo: Long Context Multimodal Benchmark for Dialogue.* Snap Research.

---

## License

Copyright 2026 **Sachit Misra** — Licensed under [CC-BY-NC-4.0](LICENSE).

**Free for:** personal use, education, academic research, open-source projects.  
**Not permitted:** commercial use without a separate written agreement.

Commercial licensing: mishrasachit1@gmail.com
