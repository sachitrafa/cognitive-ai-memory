# YourMemory

**+16pp better recall than Mem0 on LoCoMo. 100% stale memory precision. Biologically-inspired memory decay for AI agents.**

Persistent memory for Claude and any MCP-compatible AI — works like human memory. Important things stick, forgotten things fade, outdated facts get pruned automatically.

> Early stage — feedback and ideas welcome.

---

## Benchmarks

Evaluated against Mem0 (free tier) on the public [LoCoMo dataset](https://github.com/snap-research/locomo) (Snap Research) — 10 conversation pairs, 200 QA pairs total.

| Metric | YourMemory | Mem0 | Margin |
|--------|:----------:|:----:|:------:|
| LoCoMo Recall@5 *(200 QA pairs)* | **34%** | 18% | **+16pp** |
| Stale Memory Precision *(5 contradiction pairs)* | **100%** | 0% | **+100pp** |
| Memories pruned *(noise reduction)* | **20%** | 0% | — |

Full methodology and per-sample results in [BENCHMARKS.md](BENCHMARKS.md).
Read the writeup: [I built memory decay for AI agents using the Ebbinghaus forgetting curve](https://dev.to/sachit_mishra_686a94d1bb5/i-built-memory-decay-for-ai-agents-using-the-ebbinghaus-forgetting-curve-1b0e)

---

## How it works

### Ebbinghaus Forgetting Curve

```
base_λ      = DECAY_RATES[category]
effective_λ = base_λ × (1 - importance × 0.8)
strength    = importance × e^(-effective_λ × days) × (1 + recall_count × 0.2)
score       = cosine_similarity × strength
```

Decay rate varies by **category** — failure memories fade fast, strategies persist longer:

| Category | base λ | survives without recall | use case |
|----------|--------|------------------------|----------|
| `strategy` | 0.10 | ~38 days | What worked — successful patterns |
| `fact` | 0.16 | ~24 days | User preferences, identity |
| `assumption` | 0.20 | ~19 days | Inferred context |
| `failure` | 0.35 | ~11 days | What went wrong — environment-specific errors |

Importance additionally modulates the decay rate within each category. Memories recalled frequently gain `recall_count` boosts that counteract decay. Memories below strength `0.05` are pruned automatically.

---

## Setup

**Zero infrastructure required** — uses DuckDB out of the box. Two commands and you're done.

Supports **Python 3.11, 3.12, 3.13, and 3.14**.

### 1. Install

```bash
pip install yourmemory
```

All dependencies installed automatically. No clone, no Docker, no database setup.

### 2. Get your config

Run this once to get your exact config:

```bash
yourmemory-path
```

It prints your full executable path and a ready-to-paste config for any MCP client. Copy it.

### 3. Wire into your AI client

The database is created automatically at `~/.yourmemory/memories.duckdb` on first use.

#### Claude Code

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

Reload Claude Code (`Cmd+Shift+P` → `Developer: Reload Window`).

#### Cline (VS Code)

VS Code doesn't inherit your shell PATH. Run this in terminal to get the exact config to paste:

```bash
yourmemory-path
```

Then in Cline → **MCP Servers** → **Edit MCP Settings**, paste the output. It looks like:

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/full/path/to/yourmemory",
      "args": [],
      "env": {
        "YOURMEMORY_USER": "your_name",
        "DATABASE_URL": ""
      }
    }
  }
}
```

Restart Cline after saving.

#### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/full/path/to/yourmemory",
      "args": [],
      "env": {
        "YOURMEMORY_USER": "your_name",
        "DATABASE_URL": ""
      }
    }
  }
}
```

#### Claude Desktop

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

#### Any MCP-compatible client

YourMemory is a standard stdio MCP server. Works with Claude Code, Claude Desktop, Cline, Cursor, Windsurf, Continue, and Zed. Use the full path from `yourmemory-path` if the client doesn't inherit shell PATH.

### 4. Add memory instructions to your project

Copy `sample_CLAUDE.md` into your project root as `CLAUDE.md` and replace:
- `YOUR_NAME` — your name (e.g. `Alice`)
- `YOUR_USER_ID` — used to namespace memories (e.g. `alice`)

Claude will now follow the recall → store → update workflow automatically on every task.

---

### PostgreSQL (optional — for teams or large datasets)

Install with Postgres support:

```bash
pip install yourmemory[postgres]
```

Then create a `.env` file:

```bash
DATABASE_URL=postgresql://YOUR_USER@localhost:5432/yourmemory
```

The backend is selected automatically — `postgresql://` in `DATABASE_URL` → Postgres + pgvector, anything else → DuckDB.

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

---

## MCP Tools

| Tool | When to call |
|------|-------------|
| `recall_memory` | Start of every task — surface relevant context |
| `store_memory` | After learning a new preference, fact, failure, or strategy |
| `update_memory` | When a recalled memory is outdated or needs merging |

`store_memory` accepts an optional `category` parameter to control decay rate:

```python
# Failure — decays in ~11 days (environment changes fast)
store_memory(
    content="OAuth for client X fails — redirect URI must be app.example.com",
    importance=0.6,
    category="failure"
)

# Strategy — decays in ~38 days (successful patterns stay relevant)
store_memory(
    content="Cursor pagination fixed the 30s timeout on large user queries",
    importance=0.7,
    category="strategy"
)
```

### Example session

```
User: "I prefer tabs over spaces in all my Python projects"

Claude:
  → recall_memory("tabs spaces Python preferences")   # nothing found
  → store_memory("Sachit prefers tabs over spaces in Python", importance=0.9, category="fact")

Next session:
  → recall_memory("Python formatting")
  ← {"content": "Sachit prefers tabs over spaces in Python", "strength": 0.87}
  → Claude now knows without being told again
```

---

## Decay Job

Runs automatically every 24 hours on startup — no cron needed. Memories below strength `0.05` are pruned.

---

## Stack

- **DuckDB** — default backend, zero setup, native vector similarity (same quality as pgvector)
- **sentence-transformers** — local embeddings (`all-mpnet-base-v2`, 768 dims, no external service needed)
- **spaCy 3.8.13+** — local NLP for deduplication and categorization (Python 3.11–3.14 compatible)
- **APScheduler** — automatic 24h decay job
- **MCP** — Claude integration via Model Context Protocol
- **PostgreSQL + pgvector** — optional, for teams / large datasets

---

## Architecture

```
Claude / Cline / Cursor / Any MCP client
    │
    ├── recall_memory(query)
    │       └── embed → cosine similarity → score = sim × strength → top-k
    │
    ├── store_memory(content, importance, category?)
    │       └── is_question? → reject
    │           category: fact | assumption | failure | strategy
    │           embed() → INSERT memories
    │
    └── update_memory(id, new_content)
            └── embed(new_content) → UPDATE memories

DuckDB (default)                    PostgreSQL + pgvector (optional)
    └── memories.duckdb                 └── memories table
        ├── embedding FLOAT[768]            ├── embedding vector(768)
        ├── importance FLOAT               ├── importance float
        ├── recall_count INTEGER           ├── recall_count int
        └── last_accessed_at               └── last_accessed_at
```

---

## Dataset Reference

Benchmarks use the [LoCoMo](https://github.com/snap-research/locomo) dataset by Snap Research — a public long-context memory benchmark for multi-session dialogue.

> Maharana et al. (2024). *LoCoMo: Long Context Multimodal Benchmark for Dialogue.* Snap Research.

---

## License

Copyright 2026 **Sachit Misra**

Licensed under the [Apache License, Version 2.0](LICENSE).
You may use, modify, and distribute this software freely with attribution.
Patent protection included — contributors cannot sue users over patent claims.
