CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id               SERIAL PRIMARY KEY,
    user_id          TEXT NOT NULL,
    content          TEXT NOT NULL,
    memory_type      TEXT NOT NULL DEFAULT 'trivial',
    importance       FLOAT NOT NULL DEFAULT 0.5,
    recall_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP DEFAULT NOW(),
    created_at       TIMESTAMP DEFAULT NOW(),
    category         TEXT NOT NULL DEFAULT 'fact',
    agent_id         TEXT NOT NULL DEFAULT 'user',
    visibility       TEXT NOT NULL DEFAULT 'shared',
    embedding        vector(768),
    UNIQUE (user_id, content)
);

-- Add agent_id and visibility to existing tables (safe to run multiple times)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='memories' AND column_name='agent_id') THEN
        ALTER TABLE memories ADD COLUMN agent_id TEXT NOT NULL DEFAULT 'user';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='memories' AND column_name='visibility') THEN
        ALTER TABLE memories ADD COLUMN visibility TEXT NOT NULL DEFAULT 'shared';
    END IF;
END $$;

-- ivfflat index is only useful at scale (10k+ rows).
-- For small datasets, PostgreSQL does an exact scan automatically.
-- Uncomment when you have enough data:
-- CREATE INDEX IF NOT EXISTS memories_embedding_idx
--     ON memories USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories (user_id);
CREATE INDEX IF NOT EXISTS memories_agent_id_idx ON memories (agent_id);

-- tsvector column for BM25-style full-text search
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='memories' AND column_name='content_tsv'
    ) THEN
        ALTER TABLE memories ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS memories_fts_idx ON memories USING GIN (content_tsv);

-- Agent registrations — API key auth for multi-agent systems
CREATE TABLE IF NOT EXISTS agent_registrations (
    id          SERIAL PRIMARY KEY,
    agent_id    TEXT UNIQUE NOT NULL,
    user_id     TEXT NOT NULL,
    api_key_hash TEXT UNIQUE NOT NULL,   -- SHA-256 hash, never store plaintext
    can_read    TEXT[] DEFAULT '{}',     -- agent_ids this agent can read from (empty = all shared)
    can_write   TEXT[] DEFAULT ARRAY['shared', 'private'],
    description TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT NOW(),
    revoked_at  TIMESTAMP               -- NULL = active
);

CREATE INDEX IF NOT EXISTS agent_reg_user_id_idx ON agent_registrations (user_id);
CREATE INDEX IF NOT EXISTS agent_reg_api_key_idx ON agent_registrations (api_key_hash);
