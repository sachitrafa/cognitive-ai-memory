import hashlib
import json
import secrets
from dotenv import load_dotenv

from src.db.connection import get_backend, get_conn

load_dotenv()

KEY_PREFIX = "ym_"


def generate_api_key() -> str:
    """Generate a new random API key. Shown to user once — never stored in plaintext."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def _row_to_agent(row) -> dict:
    """Normalise a DB row to a plain dict, deserialising JSON arrays for SQLite."""
    d = dict(row)
    for field in ("can_read", "can_write"):
        val = d.get(field)
        if isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except (ValueError, TypeError):
                d[field] = []
        elif val is None:
            d[field] = []
    return d


def register_agent(
    agent_id: str,
    user_id: str,
    description: str = "",
    can_read: list[str] = None,
    can_write: list[str] = None,
) -> dict:
    """
    Register a new agent and return its API key.
    The key is returned once and never retrievable again.
    """
    if can_read is None:
        can_read = []
    if can_write is None:
        can_write = ["shared", "private"]

    api_key  = generate_api_key()
    key_hash = hash_key(api_key)
    backend  = get_backend()
    conn     = get_conn()
    cur      = conn.cursor()

    if backend == "postgres":
        cur.execute("""
            INSERT INTO agent_registrations (agent_id, user_id, api_key_hash, can_read, can_write, description)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_id) DO UPDATE
                SET api_key_hash = EXCLUDED.api_key_hash,
                    can_read     = EXCLUDED.can_read,
                    can_write    = EXCLUDED.can_write,
                    description  = EXCLUDED.description,
                    revoked_at   = NULL
            RETURNING id
        """, (agent_id, user_id, key_hash, can_read, can_write, description))
    else:
        # DuckDB and SQLite both use ? placeholders and JSON-encoded arrays
        cur.execute("""
            INSERT INTO agent_registrations (agent_id, user_id, api_key_hash, can_read, can_write, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (agent_id) DO UPDATE
                SET api_key_hash = excluded.api_key_hash,
                    can_read     = excluded.can_read,
                    can_write    = excluded.can_write,
                    description  = excluded.description,
                    revoked_at   = NULL
        """, (agent_id, user_id, key_hash, json.dumps(can_read), json.dumps(can_write), description))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "agent_id":  agent_id,
        "user_id":   user_id,
        "api_key":   api_key,
        "can_read":  can_read,
        "can_write": can_write,
        "warning":   "Save this API key — it will not be shown again.",
    }


def validate_api_key(api_key: str) -> dict | None:
    """
    Validate an API key and return the agent registration if active.
    Returns None if key is invalid or revoked.
    """
    key_hash = hash_key(api_key)
    backend  = get_backend()
    conn     = get_conn()
    cur      = conn.cursor()

    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur.close()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT agent_id, user_id, can_read, can_write, description
            FROM agent_registrations
            WHERE api_key_hash = %s AND revoked_at IS NULL
        """, (key_hash,))
    else:
        cur.execute("""
            SELECT agent_id, user_id, can_read, can_write, description
            FROM agent_registrations
            WHERE api_key_hash = ? AND revoked_at IS NULL
        """, (key_hash,))

    row = cur.fetchone()
    cur.close()
    conn.close()

    return _row_to_agent(row) if row else None


def revoke_agent(agent_id: str, user_id: str) -> bool:
    """Revoke an agent's API key. Returns True if revoked, False if not found."""
    backend = get_backend()
    conn    = get_conn()
    cur     = conn.cursor()

    if backend == "postgres":
        cur.execute("""
            UPDATE agent_registrations
            SET revoked_at = NOW()
            WHERE agent_id = %s AND user_id = %s AND revoked_at IS NULL
            RETURNING id
        """, (agent_id, user_id))
        row = cur.fetchone()
    else:
        cur.execute("""
            SELECT id FROM agent_registrations
            WHERE agent_id = ? AND user_id = ? AND revoked_at IS NULL
        """, (agent_id, user_id))
        row = cur.fetchone()
        if row:
            cur.execute("""
                UPDATE agent_registrations SET revoked_at = CURRENT_TIMESTAMP
                WHERE agent_id = ? AND user_id = ?
            """, (agent_id, user_id))

    conn.commit()
    cur.close()
    conn.close()
    return row is not None


def list_agents(user_id: str) -> list[dict]:
    """List all active agents for a user."""
    backend = get_backend()
    conn    = get_conn()
    cur     = conn.cursor()

    if backend == "postgres":
        from psycopg2.extras import RealDictCursor
        cur.close()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT agent_id, description, can_read, can_write, created_at
            FROM agent_registrations
            WHERE user_id = %s AND revoked_at IS NULL
            ORDER BY created_at
        """, (user_id,))
    else:
        cur.execute("""
            SELECT agent_id, description, can_read, can_write, created_at
            FROM agent_registrations
            WHERE user_id = ? AND revoked_at IS NULL
            ORDER BY created_at
        """, (user_id,))

    if backend == "duckdb":
        from src.db.connection import duckdb_rows
        result = conn.execute("""
            SELECT agent_id, description, can_read, can_write, created_at
            FROM agent_registrations
            WHERE user_id = ? AND revoked_at IS NULL
            ORDER BY created_at
        """, [user_id])
        rows = duckdb_rows(result)
        cur.close()
        conn.close()
        return [_row_to_agent(r) for r in rows]

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [_row_to_agent(r) for r in rows]
