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

---

## How it works

### Ebbinghaus Forgetting Curve

```
effective_λ = 0.16 × (1 - importance × 0.8)
strength    = importance × e^(-effective_λ × days) × (1 + recall_count × 0.2)
score       = cosine_similarity × strength
```

Importance controls both the starting strength and the rate of decay:

| importance | effective λ | survives without recall |
|------------|-------------|------------------------|
| 1.0 | 0.032 | ~94 days |
| 0.9 | 0.045 | ~64 days |
| 0.5 | 0.096 | ~24 days |
| 0.2 | 0.134 | ~10 days |

Memories recalled frequently gain `recall_count` boosts that counteract decay. Memories below strength `0.05` are pruned automatically.

---

## Setup

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) — runs Postgres
- [Ollama](https://ollama.com) — runs local embeddings

**Install and start:**

```bash
git clone https://github.com/sachitrafa/cognitive-ai-memory
cd cognitive-ai-memory
./setup.sh
```

The script pulls the embedding model, installs the Python package, creates your `.env`, and starts Postgres. DB migration and the decay scheduler run automatically on first boot.

**Start the MCP server:**

```bash
yourmemory
```

**Wire into Claude (`~/.claude/settings.json`):**

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

**Add memory instructions to your project:**

Copy `sample_CLAUDE.md` into your project root as `CLAUDE.md` and replace:
- `YOUR_NAME` — your name (e.g. `Alice`)
- `YOUR_USER_ID` — used to namespace memories (e.g. `alice`)

Claude will now follow the recall → store → update workflow automatically on every task.

---

## MCP Tools

| Tool | When to call |
|------|-------------|
| `recall_memory` | Start of every task — surface relevant context |
| `store_memory` | After learning a new preference, fact, or instruction |
| `update_memory` | When a recalled memory is outdated or needs merging |

### Example session

```
User: "I prefer tabs over spaces in all my Python projects"

Claude:
  → recall_memory("tabs spaces Python preferences")   # nothing found
  → store_memory("Sachit prefers tabs over spaces in Python", importance=0.9)

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
- **Ollama** — local embeddings (`nomic-embed-text`, 768 dims)
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
    ├── store_memory(content, importance)
    │       └── is_question? → reject
    │           categorize() → fact | assumption
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
