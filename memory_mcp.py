"""
YourMemory MCP Server

Exposes three tools to Claude:
  recall_memory  — retrieve relevant memories before answering
  store_memory   — insert a new memory after learning something new
  update_memory  — merge or replace an existing memory (by id from recall)
"""

import asyncio
import json
import os
import sys
import threading

try:
    import sqlite3  # noqa: F401
except ImportError:
    print(
        "ERROR: sqlite3 is not available in your Python installation.\n"
        "Fix: sudo apt-get install python3-sqlite3  (Ubuntu/Debian)\n"
        "     or rebuild Python with libsqlite3-dev installed.",
        file=sys.stderr,
    )
    sys.exit(1)

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# SSE transport (used when --sse flag or PORT env var is set)
def _run_sse(port: int):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.requests import Request
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ])

    print(f"YourMemory MCP server running on http://0.0.0.0:{port}/sse", file=sys.stderr, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

# Add project root so src.services imports work
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Heavy imports (spaCy model, DB drivers) are deferred to first tool call
# so the MCP handshake completes instantly on startup.
_services = {}

def _load_services():
    if _services:
        return
    from src.services.retrieve import retrieve as _retrieve
    from src.services.embed import embed
    from src.services.extract import is_question, categorize
    from src.services.api_keys import validate_api_key
    from src.services.resolve import resolve
    from src.db.connection import get_backend, get_conn, emb_to_db
    _services["retrieve"]         = _retrieve
    _services["embed"]            = embed
    _services["is_question"]      = is_question
    _services["categorize"]       = categorize
    _services["validate_api_key"] = validate_api_key
    _services["resolve"]          = resolve
    _services["get_backend"]      = get_backend
    _services["get_conn"]         = get_conn
    _services["emb_to_db"]        = emb_to_db

import getpass
DEFAULT_USER       = os.getenv("YOURMEMORY_USER") or getpass.getuser()
DEFAULT_IMPORTANCE = 0.5


# ── MCP Server ────────────────────────────────────────────────────────────────

server = Server("yourmemory")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="recall_memory",
            description=(
                "Retrieve memories relevant to a query. "
                "Call this at the start of every task to get context about the user's preferences, "
                "past instructions, and known facts. Returns a list of memories with their IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or sentence describing what to look for in memory.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": f"User identifier (default: '{DEFAULT_USER}').",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Agent API key (starts with 'ym_'). If provided, also returns this agent's private memories. If omitted, returns shared memories only.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max memories to return (default: 5).",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="store_memory",
            description=(
                "Store a new memory about the user. "
                "Use when you learn a new fact, preference, instruction, past failure, or successful strategy. "
                "Does not conflict with any memory returned by recall_memory."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact, preference, failure, or strategy to remember.",
                    },
                    "importance": {
                        "anyOf": [{"type": "number"}, {"type": "string"}],
                        "description": (
                            "You MUST decide this. How important is this memory? (0.0–1.0)\n"
                            "0.9–1.0 — core identity, permanent preferences (e.g. 'Sachit uses Python')\n"
                            "0.7–0.8 — strong preferences, recurring patterns\n"
                            "0.5     — regular facts, project decisions\n"
                            "0.2–0.3 — transient context, one-off notes from this session"
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Memory category — controls decay rate:\n"
                            "  'fact'       — user preferences, identity, stable knowledge (default, ~24 day survival)\n"
                            "  'assumption' — inferred beliefs, uncertain context (~19 days)\n"
                            "  'failure'    — what went wrong in a past task, environment-specific errors (~11 days, decays fast)\n"
                            "  'strategy'   — what worked well in a past task, approach patterns (~38 days, decays slow)\n"
                            "Use 'failure' when storing e.g. 'OAuth failed for client X due to wrong redirect URI'.\n"
                            "Use 'strategy' when storing e.g. 'Using pagination fixed the timeout on large DB queries'."
                        ),
                    },
                    "user_id": {
                        "type": "string",
                        "description": f"User identifier (default: '{DEFAULT_USER}').",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Agent API key (starts with 'ym_'). Required for agent-scoped memory. If omitted, stored as 'user' with shared visibility.",
                    },
                    "visibility": {
                        "type": "string",
                        "description": "Who can recall this memory: 'shared' (any agent, default) or 'private' (only this agent).",
                    },
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="update_memory",
            description=(
                "Merge or replace an existing memory by its ID. "
                "Use when a recalled memory is outdated (replace) or when new info adds detail "
                "to an existing memory (merge — write the combined sentence as new_content)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "ID of the memory to update (from recall_memory results).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The updated or merged memory text.",
                    },
                    "importance": {
                        "anyOf": [{"type": "number"}, {"type": "string"}],
                        "description": (
                            "You MUST decide this. Re-evaluate importance after the update. (0.0–1.0)\n"
                            "0.9–1.0 — core identity, permanent preferences\n"
                            "0.7–0.8 — strong preferences, recurring patterns\n"
                            "0.5     — regular facts, project decisions\n"
                            "0.2–0.3 — transient context, one-off notes"
                        ),
                    },
                },
                "required": ["memory_id", "new_content"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    _load_services()
    retrieve         = _services["retrieve"]
    embed            = _services["embed"]
    is_question      = _services["is_question"]
    categorize       = _services["categorize"]
    validate_api_key = _services["validate_api_key"]
    resolve          = _services["resolve"]
    get_backend      = _services["get_backend"]
    get_conn         = _services["get_conn"]
    emb_to_db        = _services["emb_to_db"]

    if name == "recall_memory":
        user_id = arguments.get("user_id", DEFAULT_USER)
        query   = arguments["query"]
        top_k   = arguments.get("top_k", 5)
        api_key = arguments.get("api_key")

        agent = None
        if api_key:
            agent = validate_api_key(api_key)
            if not agent:
                return [types.TextContent(type="text", text=json.dumps(
                    {"error": "Invalid or revoked API key."}))]

        agent_id = agent["agent_id"] if agent else None
        result   = retrieve(user_id, query, top_k=top_k, agent_id=agent_id)

        if agent:
            can_read = agent.get("can_read", [])
            if can_read:
                result["memories"] = [
                    m for m in result["memories"]
                    if m["agent_id"] in can_read
                ]
                result["memoriesFound"] = len(result["memories"])

        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    elif name == "store_memory":
        user_id = arguments.get("user_id", DEFAULT_USER)
        api_key = arguments.get("api_key")

        if api_key:
            agent = validate_api_key(api_key)
            if not agent:
                return [types.TextContent(type="text", text=json.dumps(
                    {"error": "Invalid or revoked API key."}))]
            agent_id  = agent["agent_id"]
            can_write = agent.get("can_write", ["shared", "private"])
        else:
            agent_id  = "user"
            can_write = ["shared", "private"]

        visibility = arguments.get("visibility", "shared")
        if visibility not in ("shared", "private"):
            visibility = "shared"
        if visibility not in can_write:
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"Agent '{agent_id}' is not permitted to write '{visibility}' memories."}))]

        content = arguments["content"]

        if is_question(content):
            return [types.TextContent(type="text", text=json.dumps(
                {"error": "Questions are not stored as memories."}))]

        if "importance" not in arguments:
            return [types.TextContent(type="text", text=json.dumps(
                {"error": "importance is required (0.0–1.0). Decide based on how permanent this memory should be."}))]
        importance = max(0.0, min(1.0, float(arguments["importance"])))
        valid_categories = {"fact", "assumption", "failure", "strategy"}
        raw_category = arguments.get("category", "").strip().lower()
        category     = raw_category if raw_category in valid_categories else categorize(content)
        embedding    = embed(content)

        backend = get_backend()
        conn    = get_conn()
        cur     = conn.cursor() if backend != "duckdb" else None

        resolution    = resolve(user_id, content, embedding, conn)
        action        = resolution["action"]
        final_content = resolution["content"]
        existing      = resolution["existing"]

        if action == "reinforce":
            if backend == "postgres":
                cur.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                    WHERE id = %s RETURNING id
                """, (existing["id"],))
            elif backend == "duckdb":
                conn.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = now()
                    WHERE id = ?
                """, [existing["id"]])
            else:
                cur.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (existing["id"],))
            memory_id = existing["id"]
            category  = existing["category"]

        elif action in ("replace", "merge"):
            new_embedding = embed(final_content)
            new_emb_str   = emb_to_db(new_embedding, backend)
            new_category  = categorize(final_content)
            try:
                if backend == "postgres":
                    cur.execute("""
                        UPDATE memories
                        SET content = %s, embedding = %s::vector, category = %s,
                            recall_count = recall_count + 1, last_accessed_at = NOW()
                        WHERE id = %s RETURNING id
                    """, (final_content, new_emb_str, new_category, existing["id"]))
                elif backend == "duckdb":
                    conn.execute("""
                        UPDATE memories
                        SET content = ?, embedding = ?, category = ?,
                            recall_count = recall_count + 1, last_accessed_at = now()
                        WHERE id = ?
                    """, [final_content, new_emb_str, new_category, existing["id"]])
                else:
                    cur.execute("""
                        UPDATE memories
                        SET content = ?, embedding = ?, category = ?,
                            recall_count = recall_count + 1, last_accessed_at = datetime('now')
                        WHERE id = ?
                    """, (final_content, new_emb_str, new_category, existing["id"]))
                memory_id = existing["id"]
                category  = new_category
            except Exception:
                if backend != "duckdb":
                    conn.rollback()
                if backend == "postgres":
                    cur.execute("""
                        UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                        WHERE user_id = %s AND content = %s RETURNING id
                    """, (user_id, final_content))
                elif backend == "duckdb":
                    conn.execute("""
                        UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = now()
                        WHERE user_id = ? AND content = ?
                    """, [user_id, final_content])
                else:
                    cur.execute("""
                        UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                        WHERE user_id = ? AND content = ?
                    """, (user_id, final_content))
                memory_id = existing["id"]
                category  = existing["category"]

        else:  # "new"
            emb_str = emb_to_db(embedding, backend)
            if backend == "postgres":
                cur.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding, agent_id, visibility)
                    VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = memories.recall_count + 1, last_accessed_at = NOW()
                    RETURNING id
                """, (user_id, final_content, category, importance, emb_str, agent_id, visibility))
                memory_id = cur.fetchone()[0]
            elif backend == "duckdb":
                result = conn.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding, agent_id, visibility)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = recall_count + 1, last_accessed_at = now()
                    RETURNING id
                """, [user_id, final_content, category, importance, emb_str, agent_id, visibility])
                row = result.fetchone()
                memory_id = row[0] if row else None
            else:
                cur.execute("""
                    INSERT INTO memories (user_id, content, category, importance, embedding, agent_id, visibility)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, content) DO UPDATE
                        SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                """, (user_id, final_content, category, importance, emb_str, agent_id, visibility))
                memory_id = cur.lastrowid

        if backend != "duckdb":
            conn.commit()
            cur.close()
        conn.close()

        return [types.TextContent(type="text", text=json.dumps(
            {"stored": 1, "id": memory_id, "content": final_content, "category": category,
             "importance": importance, "agent_id": agent_id, "visibility": visibility,
             "action": action}))]

    elif name == "update_memory":
        memory_id   = arguments["memory_id"]
        new_content = arguments["new_content"]
        if "importance" not in arguments:
            return [types.TextContent(type="text", text=json.dumps(
                {"error": "importance is required (0.0–1.0). Re-evaluate after the update."}))]
        importance = max(0.0, min(1.0, float(arguments["importance"])))

        category  = _services["categorize"](new_content)
        embedding = embed(new_content)
        backend   = get_backend()
        emb_str   = emb_to_db(embedding, backend)
        conn      = get_conn()
        cur       = conn.cursor() if backend != "duckdb" else None

        # Fetch owner to scope the dedup query
        if backend == "postgres":
            cur.execute("SELECT user_id FROM memories WHERE id = %s", (memory_id,))
            owner = cur.fetchone()
        elif backend == "duckdb":
            owner = conn.execute("SELECT user_id FROM memories WHERE id = ?", [memory_id]).fetchone()
        else:
            cur.execute("SELECT user_id FROM memories WHERE id = ?", (memory_id,))
            owner = cur.fetchone()

        if owner is None:
            if cur: cur.close()
            conn.close()
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"Memory {memory_id} not found."}))]
        user_id_owner = owner[0]

        # Check if new content clashes with a *different* row
        resolution = resolve(user_id_owner, new_content, embedding, conn)
        if resolution["action"] != "new" and resolution["existing"]["id"] != memory_id:
            existing = resolution["existing"]
            if backend == "postgres":
                cur.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                    WHERE id = %s RETURNING id, content, category, importance
                """, (existing["id"],))
                row = cur.fetchone()
                conn.commit()
                cur.close()
            elif backend == "duckdb":
                conn.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = now()
                    WHERE id = ?
                """, [existing["id"]])
                row = conn.execute(
                    "SELECT id, content, category, importance FROM memories WHERE id = ?",
                    [existing["id"]]
                ).fetchone()
            else:
                cur.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = datetime('now')
                    WHERE id = ?
                """, (existing["id"],))
                cur.execute("SELECT id, content, category, importance FROM memories WHERE id = ?", (existing["id"],))
                row = cur.fetchone()
                conn.commit()
                cur.close()
            conn.close()
            return [types.TextContent(type="text", text=json.dumps(
                {"updated": 1, "id": row[0], "content": row[1], "category": row[2],
                 "importance": row[3], "action": "reinforce_existing"}))]

        if backend == "postgres":
            cur.execute("""
                UPDATE memories
                SET content = %s, embedding = %s::vector, category = %s, importance = %s,
                    recall_count = recall_count + 1, last_accessed_at = NOW()
                WHERE id = %s RETURNING id, content, category, importance
            """, (new_content, emb_str, category, importance, memory_id))
            row = cur.fetchone()
            conn.commit()
            cur.close()
        elif backend == "duckdb":
            conn.execute("""
                UPDATE memories
                SET content = ?, embedding = ?, category = ?, importance = ?,
                    recall_count = recall_count + 1, last_accessed_at = now()
                WHERE id = ?
            """, [new_content, emb_str, category, importance, memory_id])
            row = conn.execute(
                "SELECT id, content, category, importance FROM memories WHERE id = ?",
                [memory_id]
            ).fetchone()
        else:
            cur.execute("""
                UPDATE memories
                SET content = ?, embedding = ?, category = ?, importance = ?,
                    recall_count = recall_count + 1, last_accessed_at = datetime('now')
                WHERE id = ?
            """, (new_content, emb_str, category, importance, memory_id))
            cur.execute("SELECT id, content, category, importance FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
            conn.commit()
            cur.close()

        conn.close()

        if row is None:
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"Memory {memory_id} not found."}))]

        return [types.TextContent(type="text", text=json.dumps(
            {"updated": 1, "id": row[0], "content": row[1], "category": row[2], "importance": row[3]}))]

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


def _start_decay_scheduler():
    """Run the decay job once immediately, then every 24 hours in a background thread."""
    from src.jobs.decay_job import run as run_decay

    def loop():
        run_decay()
        timer = threading.Event()
        while not timer.wait(timeout=86400):
            run_decay()

    t = threading.Thread(target=loop, daemon=True, name="decay-scheduler")
    t.start()


async def main():
    # Run DB migration on startup (creates tables on first run, safe to repeat)
    from src.db.migrate import migrate
    migrate()
    _start_decay_scheduler()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def print_path():
    """Print the full path to the yourmemory executable and ready-to-paste Cline config."""
    import shutil, json as _json, getpass
    path = shutil.which("yourmemory") or sys.executable.replace("python", "yourmemory")
    user = os.getenv("USER") or getpass.getuser() or "your_name"
    config = {
        "mcpServers": {
            "yourmemory": {
                "command": path,
                "args": [],
                "env": {
                    "YOURMEMORY_USER": user,
                    "DATABASE_URL": ""
                },
                "disabled": False,
                "autoApprove": []
            }
        }
    }
    print(f"\nYourMemory path: {path}\n")
    print("Paste this into your Cline MCP settings:\n")
    print(_json.dumps(config, indent=2))

def setup():
    """Run once after pip install to download the spaCy model."""
    import subprocess
    print("YourMemory setup — installing spaCy language model...")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=False,
    )
    if result.returncode == 0:
        print("✓ spaCy model installed successfully.")
    else:
        # Fallback: install via direct wheel URL
        print("Direct download fallback...")
        result2 = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "https://github.com/explosion/spacy-models/releases/download/"
             "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
             "--break-system-packages"],
            check=False,
        )
        if result2.returncode == 0:
            print("✓ spaCy model installed successfully.")
        else:
            print("✗ Could not install spaCy model automatically.")
            print("  Run manually: python -m spacy download en_core_web_sm")
            print("  YourMemory will still work using the built-in regex fallback.")

    # Also run DB migration
    from src.db.migrate import migrate
    migrate()
    print("✓ Database initialised.")
    print("\nSetup complete. Run yourmemory-path to get your MCP config.")


def run():
    from src.db.migrate import migrate
    migrate()
    _start_decay_scheduler()

    # SSE mode: --sse flag or PORT env var
    use_sse = "--sse" in sys.argv
    port    = int(os.getenv("PORT", 0))
    if not port and "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    if use_sse or port:
        _run_sse(port or 3000)
    else:
        asyncio.run(main())


if __name__ == "__main__":
    run()
