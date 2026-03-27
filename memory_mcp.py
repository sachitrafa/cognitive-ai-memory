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

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Add project root so src.services imports work
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Heavy imports (spaCy model, DB drivers) are deferred to first tool call
# so the MCP handshake completes instantly on startup.
_services = {}

def _load_services():
    if _services:
        return
    import psycopg2
    from src.services.retrieve import retrieve as _retrieve
    from src.services.embed import embed
    from src.services.extract import is_question, categorize
    from src.services.api_keys import validate_api_key
    from src.services.resolve import resolve
    _services["psycopg2"]          = psycopg2
    _services["retrieve"]          = _retrieve
    _services["embed"]             = embed
    _services["is_question"]       = is_question
    _services["categorize"]        = categorize
    _services["validate_api_key"]  = validate_api_key
    _services["resolve"]           = resolve

DEFAULT_USER = "sachit"
DEFAULT_IMPORTANCE = 0.5


def _get_conn():
    _load_services()
    return _services["psycopg2"].connect(os.getenv("DATABASE_URL"))


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
                        "type": "number",
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
                "required": ["content", "importance"],
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
                        "type": "number",
                        "description": (
                            "You MUST decide this. Re-evaluate importance after the update. (0.0–1.0)\n"
                            "0.9–1.0 — core identity, permanent preferences\n"
                            "0.7–0.8 — strong preferences, recurring patterns\n"
                            "0.5     — regular facts, project decisions\n"
                            "0.2–0.3 — transient context, one-off notes"
                        ),
                    },
                },
                "required": ["memory_id", "new_content", "importance"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    _load_services()
    retrieve          = _services["retrieve"]
    embed             = _services["embed"]
    is_question       = _services["is_question"]
    categorize        = _services["categorize"]
    validate_api_key  = _services["validate_api_key"]

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

        # Filter by can_read rules from the registration
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

        # Resolve agent identity from API key
        if api_key:
            agent = validate_api_key(api_key)
            if not agent:
                return [types.TextContent(type="text", text=json.dumps(
                    {"error": "Invalid or revoked API key."}))]
            agent_id = agent["agent_id"]
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

        conn = _get_conn()
        cur  = conn.cursor()

        resolution    = _services["resolve"](user_id, content, embedding, conn)
        action        = resolution["action"]
        final_content = resolution["content"]
        existing      = resolution["existing"]

        if action == "reinforce":
            cur.execute("""
                UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                WHERE id = %s RETURNING id
            """, (existing["id"],))
            memory_id = existing["id"]
            category  = existing["category"]

        elif action in ("replace", "merge"):
            new_embedding = embed(final_content)
            new_emb_str   = f"[{','.join(str(x) for x in new_embedding)}]"
            new_category  = categorize(final_content)
            try:
                cur.execute("""
                    UPDATE memories
                    SET content = %s, embedding = %s::vector, category = %s,
                        recall_count = recall_count + 1, last_accessed_at = NOW()
                    WHERE id = %s RETURNING id
                """, (final_content, new_emb_str, new_category, existing["id"]))
                memory_id = existing["id"]
                category  = new_category
            except Exception:
                conn.rollback()
                cur.execute("""
                    UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                    WHERE user_id = %s AND content = %s RETURNING id
                """, (user_id, final_content))
                memory_id = existing["id"]
                category  = existing["category"]

        else:  # "new"
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            cur.execute("""
                INSERT INTO memories (user_id, content, category, importance, embedding, agent_id, visibility)
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (user_id, content) DO UPDATE
                    SET recall_count = memories.recall_count + 1, last_accessed_at = NOW()
                RETURNING id
            """, (user_id, final_content, category, importance, embedding_str, agent_id, visibility))
            memory_id = cur.fetchone()[0]

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

        category  = categorize(new_content)
        embedding = embed(new_content)

        conn = _get_conn()
        cur  = conn.cursor()

        # Fetch the memory's owner so resolve() can scope the dedup query
        cur.execute("SELECT user_id FROM memories WHERE id = %s", (memory_id,))
        owner = cur.fetchone()
        if owner is None:
            cur.close()
            conn.close()
            return [types.TextContent(type="text", text=json.dumps(
                {"error": f"Memory {memory_id} not found."}))]
        user_id_owner = owner[0]

        # Check if new content clashes with a *different* row
        resolution = _services["resolve"](user_id_owner, new_content, embedding, conn)
        if resolution["action"] != "new" and resolution["existing"]["id"] != memory_id:
            # New content is a duplicate of another row — reinforce that row instead
            existing = resolution["existing"]
            cur.execute("""
                UPDATE memories SET recall_count = recall_count + 1, last_accessed_at = NOW()
                WHERE id = %s RETURNING id, content, category, importance
            """, (existing["id"],))
            row = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
            return [types.TextContent(type="text", text=json.dumps(
                {"updated": 1, "id": row[0], "content": row[1], "category": row[2],
                 "importance": row[3], "action": "reinforce_existing"}))]

        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        cur.execute("""
            UPDATE memories
            SET content          = %s,
                embedding        = %s::vector,
                category         = %s,
                importance       = %s,
                recall_count     = recall_count + 1,
                last_accessed_at = NOW()
            WHERE id = %s
            RETURNING id, content, category, importance
        """, (new_content, embedding_str, category, importance, memory_id))
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
        # Schedule subsequent runs every 24 hours
        timer = threading.Event()
        while not timer.wait(timeout=86400):
            run_decay()

    t = threading.Thread(target=loop, daemon=True, name="decay-scheduler")
    t.start()


async def main():
    _start_decay_scheduler()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
