-- GIN index on campaign_audit_log.metadata for persona dedup queries
-- Supports: metadata->>'company' = $1 and metadata->>'prospect_id' lookups
-- used by prospect_matching._fetch_already_matched_prospect_ids()

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_campaign_audit_metadata
    ON campaign_audit_log USING gin (metadata jsonb_path_ops);
