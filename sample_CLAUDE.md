# Claude Memory Instructions

## Setup

This project uses **YourMemory** for persistent memory across Claude sessions.
The `yourmemory` MCP server must be running and registered before starting.

### 1. Start the MCP server

```bash
# Docker (recommended)
docker compose up -d

# or local
source venv311/bin/activate
python memory_mcp.py
```

### 2. Register in `~/.claude/settings.json`

```json
{
  "mcpServers": {
    "yourmemory": {
      "command": "/INSTALL_PATH/venv311/bin/python3.11",
      "args": ["/INSTALL_PATH/memory_mcp.py"]
    }
  }
}
```

Replace `/INSTALL_PATH` with the output of `pwd` inside the cloned repo.
Reload Claude Code after saving.

---

## Memory Workflow (follow on every task)

### Step 1 — Recall before acting

At the start of every task, call `recall_memory` with keywords from the request:

```
recall_memory(query="<keywords from prompt>", user_id="YOUR_USER_ID")
```

This surfaces relevant preferences, past decisions, and facts — so the user
never has to repeat themselves.

### Step 2 — Decide: store, update, or ignore

| Case | Condition | Action |
|---|---|---|
| **Contradiction** | New fact conflicts with a recalled memory | `update_memory(memory_id, new_content)` |
| **Extension** | New fact adds detail to a recalled memory | `update_memory(memory_id, merged_sentence)` |
| **New knowledge** | Fact is new and meaningful | `store_memory(content, importance)` |
| **Ignore** | Trivial exchange, question, or conversational filler | Do nothing |

**What counts as knowledge:**
- User preferences, habits, instructions, goals
- Project architectural decisions, constraints, design choices
- Domain facts discovered during the task

**What to ignore:**
- Questions (`why is X?`, `how do I?`)
- Claude's own opinions or responses
- One-word or throwaway messages

**How to phrase memories:**
- User facts: `"<Name> prefers..."` / `"<Name> uses..."`
- Project facts: `"The project uses..."` / `"The API expects..."`

### Step 3 — Complete the task using memory context

Apply recalled context silently — do not announce "I remember that...".
Just use the knowledge to give a better, more personalized response.

### Step 4 — Persist immediately

Call the MCP tool right after identifying the case. Do not batch or defer.

Choose `importance` based on how durable the fact is:

| Importance | When to use |
|---|---|
| `0.9–1.0` | Core identity, permanent preferences |
| `0.7–0.8` | Strong recurring preferences, project-level decisions |
| `0.5` | Regular facts, one-time project choices |
| `0.2–0.3` | Transient session context |

```
store_memory(content="<Name> prefers ...", importance=0.8, user_id="YOUR_USER_ID")
update_memory(memory_id=<id>, new_content="...", importance=0.8)
```

---

## MCP Tools Reference

### `recall_memory`
```
recall_memory(query, user_id, top_k=5)
```
Returns the top-k most relevant, highest-strength memories for the query.

### `store_memory`
```
store_memory(content, importance, user_id)
```
Stores a new memory. Exact duplicates are skipped (recall count bumped instead).

### `update_memory`
```
update_memory(memory_id, new_content, importance)
```
Re-embeds and replaces an existing memory. Use for merges and overrides.

---

## User

- Name: YOUR_NAME
- user_id: `"YOUR_USER_ID"`
