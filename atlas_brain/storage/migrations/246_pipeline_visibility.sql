-- Pipeline visibility: artifact lifecycle, quarantines, events, reviews
-- Fixes the core data model gaps identified in GPT Pro review (2026-03-30)

-- 1. Artifact attempts: real lifecycle records for generation/build runs
CREATE TABLE IF NOT EXISTS artifact_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type TEXT NOT NULL,        -- blog_post, battle_card, churn_report, vendor_briefing, campaign
    artifact_id TEXT,                   -- UUID or slug of the artifact (NULL if never created)
    run_id TEXT,                        -- execution ID for traceability
    attempt_no INT NOT NULL DEFAULT 1,
    stage TEXT NOT NULL,                -- generation, quality_gate, persistence, deploy
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'rejected', 'skipped')),
    score INT,                          -- quality score at this stage
    threshold INT,                      -- threshold that was applied
    blocker_count INT DEFAULT 0,
    warning_count INT DEFAULT 0,
    blocking_issues JSONB DEFAULT '[]'::jsonb,
    warnings JSONB DEFAULT '[]'::jsonb,
    feedback_summary TEXT,              -- retry feedback sent to LLM
    failure_step TEXT,                  -- which check failed
    error_message TEXT,                 -- human-readable error
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artifact_attempts_type_status
    ON artifact_attempts (artifact_type, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifact_attempts_run
    ON artifact_attempts (run_id);
CREATE INDEX IF NOT EXISTS idx_artifact_attempts_artifact
    ON artifact_attempts (artifact_type, artifact_id);

-- 2. Enrichment quarantines: first-class quarantine decisions
CREATE TABLE IF NOT EXISTS enrichment_quarantines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID,                     -- b2b_reviews.id
    vendor_name TEXT,
    source TEXT,                        -- g2, reddit, capterra, etc.
    reason_code TEXT NOT NULL,          -- vendor_absent_noisy_source, thin_social_context, etc.
    severity TEXT NOT NULL DEFAULT 'warning'
        CHECK (severity IN ('info', 'warning', 'error')),
    actionable BOOLEAN NOT NULL DEFAULT FALSE,
    summary TEXT,
    evidence JSONB DEFAULT '{}'::jsonb, -- snapshot of what triggered quarantine
    run_id TEXT,
    quarantined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ,
    released_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_enrichment_quarantines_vendor
    ON enrichment_quarantines (vendor_name, reason_code);
CREATE INDEX IF NOT EXISTS idx_enrichment_quarantines_reason
    ON enrichment_quarantines (reason_code, quarantined_at DESC);
CREATE INDEX IF NOT EXISTS idx_enrichment_quarantines_unreleased
    ON enrichment_quarantines (released_at) WHERE released_at IS NULL;

-- 3. Pipeline visibility events: immutable event ledger
CREATE TABLE IF NOT EXISTS pipeline_visibility_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id TEXT,
    stage TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info'
        CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    actionable BOOLEAN NOT NULL DEFAULT FALSE,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    artifact_type TEXT,
    reason_code TEXT,
    rule_code TEXT,
    decision TEXT,
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_id TEXT,
    summary TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    fingerprint TEXT,
    source_table TEXT,
    source_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pve_stage_type
    ON pipeline_visibility_events (stage, event_type, severity, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_pve_entity
    ON pipeline_visibility_events (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_pve_reason
    ON pipeline_visibility_events (reason_code) WHERE reason_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pve_run
    ON pipeline_visibility_events (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pve_fingerprint
    ON pipeline_visibility_events (fingerprint) WHERE fingerprint IS NOT NULL;

-- 4. Pipeline visibility reviews: separate review state from events
CREATE TABLE IF NOT EXISTS pipeline_visibility_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'acknowledged', 'resolved', 'ignored', 'accepted_risk')),
    assignee TEXT,
    latest_event_id UUID REFERENCES pipeline_visibility_events(id),
    occurrence_count INT NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolution_code TEXT,
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pvr_status
    ON pipeline_visibility_reviews (status, last_seen_at DESC);
