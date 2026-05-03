CREATE TABLE IF NOT EXISTS b2b_llm_exact_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    usage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_hit_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_b2b_llm_exact_cache_namespace_model
    ON b2b_llm_exact_cache (namespace, model);

CREATE INDEX IF NOT EXISTS idx_b2b_llm_exact_cache_last_hit_at
    ON b2b_llm_exact_cache (last_hit_at DESC);
