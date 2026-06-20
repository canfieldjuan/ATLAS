-- Support bounded retention purges for stored FAQ deflection report artifacts.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_content_ops_deflection_reports_created_at
    ON content_ops_deflection_reports (created_at);
