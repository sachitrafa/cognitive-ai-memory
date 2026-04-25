import os
import sys
from dotenv import load_dotenv
from src.db.connection import get_backend, get_conn

load_dotenv()


def migrate():
    backend = get_backend()

    schema_map = {
        "postgres": "schema.sql",
        "sqlite":   "sqlite_schema.sql",
        "duckdb":   "duckdb_schema.sql",
    }
    schema_path = os.path.join(os.path.dirname(__file__), schema_map[backend])

    with open(schema_path, "r") as f:
        schema = f.read()

    conn = get_conn()

    if backend == "sqlite":
        conn.executescript(schema)
    elif backend == "duckdb":
        for stmt in schema.split(";"):
            # Strip comment lines, keep SQL lines
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            sql = "\n".join(lines).strip()
            if sql:
                conn.execute(sql)
    else:
        cur = conn.cursor()
        cur.execute(schema)
        conn.commit()
        cur.close()

    # ── Post-schema FTS setup ─────────────────────────────────────────────
    if backend == "sqlite":
        # Backfill any rows that existed before the FTS table was created.
        # The INSERT OR IGNORE prevents double-indexing on a fresh DB.
        conn.executescript("""
            INSERT OR IGNORE INTO memories_fts(rowid, content)
            SELECT id, content FROM memories;
        """)

    elif backend == "duckdb":
        # Install the FTS extension once (no-op if already installed).
        try:
            conn.execute("INSTALL fts; LOAD fts;")
        except Exception as exc:
            print(f"DuckDB FTS extension unavailable — keyword search disabled: {exc}",
                  file=sys.stderr)

    conn.close()
    print(f"Migration complete ({backend}).", file=sys.stderr)

    # Bootstrap the graph backend (creates indexes for Neo4j, touches pickle for NetworkX)
    try:
        from src.graph import get_graph_backend
        get_graph_backend()
        print("Graph backend initialised.", file=sys.stderr)
    except Exception as exc:
        print(f"Graph backend init skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    migrate()
