-- Sales briefs: per-opportunity pre-call/account/renewal briefs produced by the
-- AI Content Ops Sales Briefs generator. Parallel to reports (273) but tuned
-- for the sales surface: a punchy one-line headline plus ordered sections
-- (account context, signals, talking points, risks, next actions). Section
-- shape mirrors reports.sections so renderers can reuse the same component.
--
-- The status lifecycle mirrors b2b_campaigns / reports / landing_pages: drafts
-- start at 'draft', move through 'queued' / 'approved' / 'rejected' / 'expired'
-- as host workflows review them. No CHECK constraint on status: hosts may
-- extend with their own intermediate states (e.g., 'in_review',
-- 'ready_for_call') without a follow-on migration.

CREATE TABLE IF NOT EXISTS sales_briefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL,
    target_mode TEXT NOT NULL,
    brief_type TEXT NOT NULL,
    title TEXT NOT NULL,
    headline TEXT NOT NULL,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    reference_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_briefs_target
    ON sales_briefs (account_id, target_mode, target_id);

CREATE INDEX IF NOT EXISTS idx_sales_briefs_status
    ON sales_briefs (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sales_briefs_type
    ON sales_briefs (account_id, brief_type, created_at DESC);
