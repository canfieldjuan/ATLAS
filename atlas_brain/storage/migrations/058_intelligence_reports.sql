-- On-demand intelligence reports generated from pressure baselines,
-- news enrichment, and graph context via intelligence skill prompts.
CREATE TABLE IF NOT EXISTS intelligence_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_name         TEXT NOT NULL,
    entity_type         TEXT NOT NULL DEFAULT 'company',
    report_type         TEXT NOT NULL DEFAULT 'full',
    time_window_days    INT NOT NULL DEFAULT 7,
    report_text         TEXT,
    structured_data     JSONB NOT NULL DEFAULT '{}',
    pressure_snapshot   JSONB NOT NULL DEFAULT '{}',
    source_article_ids  UUID[] NOT NULL DEFAULT '{}',
    requested_by        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intelligence_reports_entity
    ON intelligence_reports (entity_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_intelligence_reports_created
    ON intelligence_reports (created_at DESC);
