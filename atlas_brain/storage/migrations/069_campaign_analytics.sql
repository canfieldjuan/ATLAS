-- Campaign funnel analytics: materialized view for aggregated engagement metrics.
-- Refreshed periodically by the campaign_analytics_refresh task.

CREATE MATERIALIZED VIEW IF NOT EXISTS campaign_funnel_stats AS
SELECT
    DATE_TRUNC('week', bc.created_at) AS week,
    bc.company_name,
    bc.vendor_name,
    bc.channel,
    COALESCE(cs.partner_id, '00000000-0000-0000-0000-000000000000'::uuid) AS partner_id,
    COUNT(*) FILTER (WHERE bc.status IN ('sent','approved','queued','cancelled','expired','draft')) AS total,
    COUNT(*) FILTER (WHERE bc.status = 'sent') AS sent,
    COUNT(*) FILTER (WHERE bc.opened_at IS NOT NULL) AS opened,
    COUNT(*) FILTER (WHERE bc.clicked_at IS NOT NULL) AS clicked,
    COUNT(*) FILTER (WHERE cs.status = 'replied') AS replied,
    COUNT(*) FILTER (WHERE cs.status = 'bounced') AS bounced,
    COUNT(*) FILTER (WHERE cs.status = 'unsubscribed') AS unsubscribed,
    COUNT(*) FILTER (WHERE cs.status = 'completed') AS completed,
    AVG(EXTRACT(EPOCH FROM (bc.opened_at - bc.sent_at))/3600)
        FILTER (WHERE bc.opened_at IS NOT NULL AND bc.sent_at IS NOT NULL) AS avg_hours_to_open,
    AVG(EXTRACT(EPOCH FROM (bc.clicked_at - bc.sent_at))/3600)
        FILTER (WHERE bc.clicked_at IS NOT NULL AND bc.sent_at IS NOT NULL) AS avg_hours_to_click
FROM b2b_campaigns bc
LEFT JOIN campaign_sequences cs ON cs.id = bc.sequence_id
GROUP BY 1, 2, 3, 4, 5;

CREATE UNIQUE INDEX IF NOT EXISTS idx_funnel_stats_key
    ON campaign_funnel_stats (week, company_name, vendor_name, channel, partner_id);
