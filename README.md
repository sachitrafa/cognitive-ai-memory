# YourMemory

**+16pp better recall than Mem0 on LoCoMo. 100% stale memory precision. Biologically-inspired memory decay for AI agents.**

Persistent memory for Claude that works like human memory — important things stick, forgotten things fade, outdated facts get demoted automatically.

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

**Zero infrastructure required** — uses SQLite out of the box. Two commands and you're done.

### 1. Install

```bash
pip install yourmemory
```

All dependencies are installed automatically. No clone, no separate download steps needed.

### 2. Wire into your AI client

The database is created automatically at `~/.yourmemory/memories.db` on first use. No `.env` file needed.

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

Cline works best with SSE transport. Start the server first in a terminal:

```bash
yourmemory --sse --port 3000
```

Then in Cline → **MCP Servers** → **Edit MCP Settings**:

```json
{
  "mcpServers": {
    "yourmemory": {
      "url": "http://localhost:3000/sse"
    }
  }
}
```

Save — Cline will connect automatically.

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

YourMemory is a standard stdio MCP server. The command is simply `yourmemory`. Add it to any client that supports MCP servers using the same pattern above.

### 3. Add memory instructions to your project

Copy `sample_CLAUDE.md` into your project root as `CLAUDE.md` and replace:
- `YOUR_NAME` — your name (e.g. `Alice`)
- `YOUR_USER_ID` — used to namespace memories (e.g. `alice`)

Claude will now follow the recall → store → update workflow automatically on every task.

---

### PostgreSQL (optional — for teams or large datasets)

If you have PostgreSQL + pgvector, create a `.env` file:

```bash
DATABASE_URL=postgresql://YOUR_USER@localhost:5432/yourmemory
```

The backend is selected automatically — `postgresql://` in `DATABASE_URL` → Postgres + pgvector, anything else → SQLite.

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

> **One-liner setup script** (macOS/Linux): `bash scripts/setup_db.sh` handles install + DB creation automatically.

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

## REST API

```bash
# Store
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"userId":"u1","content":"Prefers dark mode","importance":0.8}'

# Retrieve
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"userId":"u1","query":"UI preferences"}'

# List all
curl "http://localhost:8000/memories?userId=u1"

# Update
curl -X PUT http://localhost:8000/memories/42 \
  -H "Content-Type: application/json" \
  -d '{"content":"Prefers dark mode in all apps","importance":0.85}'

# Delete
curl -X DELETE http://localhost:8000/memories/42
```

---

## Stack

- **PostgreSQL + pgvector** — vector similarity search
- **sentence-transformers** — local embeddings (`all-mpnet-base-v2`, 768 dims, no external service needed)
- **FastAPI** — REST server
- **APScheduler** — automatic 24h decay job
- **MCP** — Claude integration via Model Context Protocol

---

## Architecture

```
Claude Code
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

PostgreSQL (pgvector)
    └── memories
        ├── embedding vector(768)
        ├── importance float
        ├── recall_count int
        └── last_accessed_at
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
