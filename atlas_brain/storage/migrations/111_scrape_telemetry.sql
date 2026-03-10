-- Phase 1 Sprint 1: Enhanced scrape telemetry
--
-- Adds CAPTCHA solve metrics and block type classification to b2b_scrape_log.
-- Closes the "proxy IP, CAPTCHA solve time" telemetry gap.

ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS captcha_attempts INT NOT NULL DEFAULT 0;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS captcha_types TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS captcha_solve_ms INT;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS block_type TEXT;
-- block_type values: NULL (no block), 'captcha', 'ip_ban', 'rate_limit', 'waf', 'unknown'
