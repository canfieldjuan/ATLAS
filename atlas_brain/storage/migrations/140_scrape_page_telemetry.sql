-- Per-page scrape telemetry and run-level enrichment
-- Migration 140
--
-- Adds stop_reason and date-range diagnostics to run-level scrape log.
-- Creates b2b_scrape_page_logs for page-level drill-down (conditional persistence).

-- Run-level enrichment
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS stop_reason TEXT;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS oldest_review DATE;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS newest_review DATE;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS date_dropped INT NOT NULL DEFAULT 0;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS duplicate_pages INT NOT NULL DEFAULT 0;
ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS has_page_logs BOOLEAN NOT NULL DEFAULT false;

-- Per-page telemetry (conditionally persisted: failures, debug, anomalies)
CREATE TABLE IF NOT EXISTS b2b_scrape_page_logs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID NOT NULL REFERENCES b2b_scrape_log(id) ON DELETE CASCADE,

    -- Request
    page                INT NOT NULL,
    url                 TEXT NOT NULL,
    requested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Response
    status_code         INT,
    final_url           TEXT,
    response_bytes      INT,
    duration_ms         INT,

    -- Extraction
    review_nodes_found  INT NOT NULL DEFAULT 0,
    reviews_parsed      INT NOT NULL DEFAULT 0,
    missing_date        INT NOT NULL DEFAULT 0,
    missing_rating      INT NOT NULL DEFAULT 0,
    missing_body        INT NOT NULL DEFAULT 0,
    missing_author      INT NOT NULL DEFAULT 0,

    -- Date range on this page
    oldest_review       DATE,
    newest_review       DATE,

    -- Pagination
    next_page_found     BOOLEAN NOT NULL DEFAULT false,
    next_page_url       TEXT,
    content_hash        TEXT,       -- SHA-256 prefix for duplicate page detection

    -- Dedup
    duplicate_reviews   INT NOT NULL DEFAULT 0,

    -- Stop
    stop_reason         TEXT,       -- date_cutoff, duplicate_page, blocked_or_throttled, etc.

    -- Errors on this page
    errors              JSONB NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_scrape_page_logs_run
    ON b2b_scrape_page_logs(run_id, page);
CREATE INDEX IF NOT EXISTS idx_scrape_page_logs_stop
    ON b2b_scrape_page_logs(stop_reason) WHERE stop_reason IS NOT NULL;
