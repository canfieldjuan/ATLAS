-- Migration 059: Intervention approvals and safety audit trail
-- Tracks approval requests for high-risk intervention pipeline stages.
-- Uses atlas_events for immutable audit logging.

CREATE TABLE IF NOT EXISTS intervention_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id UUID NOT NULL,
    stage TEXT NOT NULL,                    -- 'narrative_architect', 'playbook', 'simulation'
    entity_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, approved, rejected, expired
    requested_by TEXT NOT NULL,             -- api, mcp, autonomous
    reviewed_by TEXT,                       -- user ID or 'auto' for auto-approval
    review_notes TEXT,
    safety_checks JSONB NOT NULL DEFAULT '{}',  -- content_filter, risk_level, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ                 -- auto-expire unreviewed requests
);

CREATE INDEX IF NOT EXISTS idx_intervention_approvals_status
    ON intervention_approvals (status)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_intervention_approvals_pipeline
    ON intervention_approvals (pipeline_id);
CREATE INDEX IF NOT EXISTS idx_intervention_approvals_entity
    ON intervention_approvals (entity_name);
