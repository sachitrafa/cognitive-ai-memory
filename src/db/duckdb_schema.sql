-- YourMemory DuckDB schema
-- DuckDB default backend: native FLOAT[768] vectors, array_cosine_similarity

CREATE SEQUENCE IF NOT EXISTS memories_id_seq;
CREATE TABLE IF NOT EXISTS memories (
    id               BIGINT    DEFAULT nextval('memories_id_seq') PRIMARY KEY,
    user_id          VARCHAR   NOT NULL,
    content          TEXT      NOT NULL,
    category         VARCHAR   NOT NULL DEFAULT 'fact',
    importance       FLOAT     NOT NULL DEFAULT 0.5,
    embedding        FLOAT[768],
    recall_count     INTEGER   NOT NULL DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_id         VARCHAR   DEFAULT 'user',
    visibility       VARCHAR   DEFAULT 'shared',
    UNIQUE(user_id, content)
);

CREATE SEQUENCE IF NOT EXISTS agent_registrations_id_seq;
CREATE TABLE IF NOT EXISTS agent_registrations (
    id           BIGINT    DEFAULT nextval('agent_registrations_id_seq') PRIMARY KEY,
    agent_id     VARCHAR   NOT NULL UNIQUE,
    user_id      VARCHAR   NOT NULL,
    api_key_hash VARCHAR   NOT NULL UNIQUE,
    can_read     VARCHAR   DEFAULT '[]',
    can_write    VARCHAR   DEFAULT '["shared","private"]',
    description  TEXT      DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used    TIMESTAMP,
    revoked_at   TIMESTAMP
);
