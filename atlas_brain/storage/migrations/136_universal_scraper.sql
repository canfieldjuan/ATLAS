-- Universal web scraper: data-agnostic LLM-powered extraction.
-- Job tracking and result storage.

CREATE TABLE IF NOT EXISTS universal_scrape_jobs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name              TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'running', 'completed', 'partial_success', 'failed', 'cancelled')),
    config            JSONB NOT NULL,
    total_targets     INT NOT NULL DEFAULT 0,
    completed_targets INT NOT NULL DEFAULT 0,
    failed_targets    INT NOT NULL DEFAULT 0,
    total_records     INT NOT NULL DEFAULT 0,
    error             TEXT,
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usj_status  ON universal_scrape_jobs(status);
CREATE INDEX IF NOT EXISTS idx_usj_created ON universal_scrape_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS universal_scrape_results (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id           UUID NOT NULL REFERENCES universal_scrape_jobs(id) ON DELETE CASCADE,
    target_url       TEXT NOT NULL,
    page_number      INT NOT NULL DEFAULT 1,
    page_title       TEXT,
    extracted_data   JSONB NOT NULL,
    item_count       INT NOT NULL DEFAULT 0,
    raw_llm_response TEXT,
    duration_ms      INT,
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usr_job ON universal_scrape_results(job_id);
CREATE INDEX IF NOT EXISTS idx_usr_url ON universal_scrape_results(target_url);

-- Prevent duplicate result rows for the same page in the same job
CREATE UNIQUE INDEX IF NOT EXISTS idx_usr_dedupe
    ON universal_scrape_results(job_id, target_url, page_number);
