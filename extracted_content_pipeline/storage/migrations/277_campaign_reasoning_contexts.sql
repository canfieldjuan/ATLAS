-- Per-tenant reasoning contexts for the Content Ops campaign generator.
--
-- Backs the new `PostgresCampaignReasoningContextRepository`, the
-- DB-backed counterpart to PR #462's file-backed
-- `FileCampaignReasoningContextProvider`. Both implement
-- `CampaignReasoningProviderPort.read_campaign_reasoning_context`,
-- so the host can swap providers without changing the route mount
-- (PR #402 / PR #462) or the bundle's
-- `with_reasoning_context()` derivation.
--
-- Selectors are the heterogeneous keys the file-backed provider's
-- `_candidate_keys` already understands -- `target_id`, `id`,
-- `company`, `company_name`, `account`, `account_name`, `email`,
-- `contact_email`, `vendor`, `vendor_name`, plus their lowercase
-- variants. Persisted as a TEXT[] so the read path can match with
-- a single `selectors && $2::text[]` predicate (GIN-indexed) per
-- per-target lookup, mirroring the file-backed dict-of-keys index.
--
-- payload JSONB carries the full normalized
-- `CampaignReasoningContext` shape (anchor_examples,
-- witness_highlights, top_theses, account_signals, timing_windows,
-- proof_points, coverage_limits, canonical_reasoning,
-- scope_summary, delta_summary). Hosts populate it by writing
-- `campaign_reasoning_context_metadata(context)` (the
-- normalize-then-emit helper already used by the file-backed
-- loader) into the column.
--
-- target_mode is persisted and filtered by the DB read path.
-- Blank target_mode rows are treated as global fallback contexts
-- so legacy seed data can still serve multiple outputs, but a row
-- saved for one nonblank mode does not satisfy another mode when
-- selectors overlap.
--
-- updated_at drives the final tie-breaker when multiple rows match
-- the same selector priority and target-mode specificity.

CREATE TABLE IF NOT EXISTS campaign_reasoning_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL DEFAULT '',
    selectors TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaign_reasoning_contexts_selectors
    ON campaign_reasoning_contexts USING GIN (selectors);

CREATE INDEX IF NOT EXISTS idx_campaign_reasoning_contexts_account
    ON campaign_reasoning_contexts (account_id, target_mode, updated_at DESC);
