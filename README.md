<div align="center">

# YourMemory

**Persistent memory for AI agents — built on the science of how humans remember.**

[![PyPI](https://img.shields.io/pypi/v/yourmemory?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/yourmemory/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/yourmemory?color=brightgreen)](https://pypi.org/project/yourmemory/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Recall@5](https://img.shields.io/badge/Recall%405-59%25-brightgreen)](BENCHMARKS.md)
[![Docker Build](https://img.shields.io/github/actions/workflow/status/sachitrafa/YourMemory/docker-publish.yml?branch=main&label=docker&logo=docker)](https://github.com/sachitrafa/YourMemory/actions/workflows/docker-publish.yml)

</div>

---

## The Problem

Every session, your AI assistant starts from zero. It asks the same questions, forgets your preferences, re-learns your stack. There is no memory between conversations.

**YourMemory fixes that.** It gives AI agents a persistent memory layer that works the way human memory does — important things stick, forgotten things fade, outdated facts get replaced automatically. One command to install, zero infrastructure required. Memory starts working the moment you add it to your AI client.

---

## How Well Does It Work?

Tested on [LoCoMo-10](https://github.com/snap-research/locomo) — 1,534 QA pairs across 10 multi-session conversations.

| System | Recall@5 | 95% CI |
|--------|:--------:|:------:|
| **YourMemory** (BM25 + vector + graph + decay) | **59%** | 56–61% |
| Zep Cloud | 28% | 26–30% |

> **2× better recall than Zep Cloud on the same benchmark.**

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

## MCP Tools

Three tools, called by your AI automatically.

| Tool | When | What it does |
|------|------|--------------|
| `recall_memory(query)` | Start of every task | Surfaces relevant memories ranked by similarity × strength |
| `store_memory(content, importance)` | After learning something new | Embeds and stores with biological decay |
| `update_memory(id, new_content)` | When a memory is outdated | Re-embeds and replaces |

```python
# Example session
store_memory("Sachit prefers tabs over spaces in Python", importance=0.9, category="fact")

# Next session — without being told again:
recall_memory("Python formatting")
# → {"content": "Sachit prefers tabs over spaces in Python", "strength": 0.87}
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
effective_λ = base_λ × (1 - importance × 0.8)
strength    = clamp(importance × e^(−effective_λ × days) × (1 + recall_count × 0.2), 0, 1)
score       = cosine_similarity × strength
```

Memories recalled frequently resist decay. Memories below strength `0.05` are pruned automatically every 24 hours.

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
| **sentence-transformers** | Local embeddings (`all-mpnet-base-v2`, 768 dims) |
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
    ├── recall_memory(query, api_key?)
    │       └── embed → vector similarity (Round 1)
    │               → graph BFS expansion  (Round 2)
    │               → score = sim × strength → top-k
    │               → recall propagation → boost neighbours
    │
    ├── store_memory(content, importance, category?, visibility?, api_key?)
    │       └── question? → reject
    │               contradiction check → update if conflict
    │               embed() → INSERT → index_memory() → graph node + edges
    │
    └── update_memory(id, new_content, importance)
            └── embed(new_content) → UPDATE → refresh graph node

  Vector DB (Round 1)             Graph DB (Round 2)
  DuckDB (default)                NetworkX (default)
    memories.duckdb                 graph.pkl
    ├── embedding FLOAT[768]        ├── nodes: memory_id, strength
    ├── importance FLOAT            └── edges: sim × verb_weight ≥ 0.4
    ├── recall_count INTEGER
    ├── visibility VARCHAR        Neo4j (opt-in)
    └── agent_id VARCHAR            └── bolt://localhost:7687
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
