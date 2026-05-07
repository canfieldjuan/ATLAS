-- Reports: customer-facing structured report drafts produced by the AI Content
-- Ops Reports generator. Parallel to b2b_campaigns but with a richer typed
-- shape (sections + reference_ids) so renderers can consume the structured
-- output directly without parsing markdown.
--
-- The status lifecycle mirrors b2b_campaigns: drafts start at 'draft', move
-- through 'queued' / 'approved' / 'rejected' / 'expired' as host workflows
-- review them. No CHECK constraint on status: hosts may extend with their own
-- intermediate states (e.g., 'in_review') without a follow-on migration.

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL,
    target_mode TEXT NOT NULL,
    report_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    reference_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_target
    ON reports (account_id, target_mode, target_id);

CREATE INDEX IF NOT EXISTS idx_reports_status
    ON reports (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_reports_type
    ON reports (account_id, report_type, created_at DESC);
