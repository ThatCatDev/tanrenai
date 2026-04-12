CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    vastai_api_key_enc BYTEA,
    idle_timeout_min INT NOT NULL DEFAULT 60,
    max_cost_per_hr REAL NOT NULL DEFAULT 1.0,
    preferred_gpu TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id),
    vastai_instance_id INT,
    status TEXT NOT NULL DEFAULT 'pending',
    gpu_name TEXT NOT NULL DEFAULT '',
    gpu_url TEXT NOT NULL DEFAULT '',
    ssh_host TEXT,
    ssh_port INT,
    cost_per_hr REAL NOT NULL DEFAULT 0,
    model_loaded TEXT NOT NULL DEFAULT '',
    provision_state TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    destroyed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_instances_user_id ON instances(user_id);
CREATE INDEX IF NOT EXISTS idx_instances_status ON instances(status);
