-- Universal scraper fixes: add partial_success status + dedupe index.
-- Must run on databases that already applied migration 136.

-- 1. Replace the status check constraint to allow partial_success.
--    DROP + ADD is safe because no rows should have invalid status values.
ALTER TABLE universal_scrape_jobs
    DROP CONSTRAINT IF EXISTS universal_scrape_jobs_status_check;

ALTER TABLE universal_scrape_jobs
    ADD CONSTRAINT universal_scrape_jobs_status_check
    CHECK (status IN ('pending', 'running', 'completed', 'partial_success', 'failed', 'cancelled'));

-- 2. Deduplicate result rows: prevent same (job, url, page) from being inserted twice.
CREATE UNIQUE INDEX IF NOT EXISTS idx_usr_dedupe
    ON universal_scrape_results(job_id, target_url, page_number);
