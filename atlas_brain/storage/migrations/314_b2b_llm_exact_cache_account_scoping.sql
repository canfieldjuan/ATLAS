-- Per-account scoping on b2b_llm_exact_cache (PR-D3, LLM Gateway MVP).
--
-- The cache_key is a SHA256 of the canonicalized request envelope
-- (provider/model/messages/...). Without scoping, customer A and
-- customer B making the same request would race on the same row.
-- Composite PK (cache_key, account_id) keeps each account in its
-- own cache namespace.
--
-- Atlas's existing pipeline writes the sentinel UUID via the
-- DEFAULT, so its cache continuity is preserved (existing rows
-- become cache_key=X, account_id=SENTINEL).

ALTER TABLE b2b_llm_exact_cache
    ADD COLUMN IF NOT EXISTS account_id UUID NOT NULL
    DEFAULT '00000000-0000-0000-0000-000000000000';

-- Drop the cache_key-only PK and replace with a composite. The
-- DROP CONSTRAINT pattern is non-destructive because the new PK
-- covers every existing row uniquely (sentinel account_id is the
-- same for all atlas-pipeline rows).
ALTER TABLE b2b_llm_exact_cache
    DROP CONSTRAINT IF EXISTS b2b_llm_exact_cache_pkey;

ALTER TABLE b2b_llm_exact_cache
    ADD CONSTRAINT b2b_llm_exact_cache_pkey
    PRIMARY KEY (cache_key, account_id);

-- Lookup pattern: WHERE cache_key = $1 AND account_id = $2.
-- The PK index covers this directly; no extra index needed.

-- Per-account analytics: rows by recency per account.
CREATE INDEX IF NOT EXISTS idx_b2b_llm_exact_cache_account_last_hit
    ON b2b_llm_exact_cache (account_id, last_hit_at DESC);
