-- Per-account scoping on reasoning_semantic_cache (PR-D3, LLM Gateway MVP).
--
-- pattern_sig was UNIQUE; that becomes UNIQUE(pattern_sig, account_id)
-- so each account has its own cache namespace. ON CONFLICT clauses
-- in the application code update from `(pattern_sig)` to
-- `(pattern_sig, account_id)`.
--
-- Atlas's existing rows get the sentinel via DEFAULT and keep
-- their cache continuity.

ALTER TABLE reasoning_semantic_cache
    ADD COLUMN IF NOT EXISTS account_id UUID NOT NULL
    DEFAULT '00000000-0000-0000-0000-000000000000';

-- Drop the pattern_sig-only UNIQUE and replace with composite.
-- The constraint name follows Postgres convention (table_col_key);
-- IF EXISTS guards against environments where it has been renamed.
ALTER TABLE reasoning_semantic_cache
    DROP CONSTRAINT IF EXISTS reasoning_semantic_cache_pattern_sig_key;

ALTER TABLE reasoning_semantic_cache
    ADD CONSTRAINT reasoning_semantic_cache_pattern_sig_account_key
    UNIQUE (pattern_sig, account_id);

-- The existing partial index idx_rsc_pattern stays; it remains
-- useful for invalidated_at IS NULL filtering.
CREATE INDEX IF NOT EXISTS idx_rsc_account_pattern
    ON reasoning_semantic_cache (account_id, pattern_sig)
    WHERE invalidated_at IS NULL;
