-- Promote scrape checkpoint state into first-class target columns
-- Migration 230

ALTER TABLE b2b_scrape_targets
    ADD COLUMN IF NOT EXISTS last_scrape_runtime_mode TEXT,
    ADD COLUMN IF NOT EXISTS last_scrape_stop_reason TEXT,
    ADD COLUMN IF NOT EXISTS last_scrape_oldest_review DATE,
    ADD COLUMN IF NOT EXISTS last_scrape_newest_review DATE,
    ADD COLUMN IF NOT EXISTS last_scrape_date_cutoff DATE,
    ADD COLUMN IF NOT EXISTS last_scrape_pages_scraped INT,
    ADD COLUMN IF NOT EXISTS last_scrape_reviews_found INT,
    ADD COLUMN IF NOT EXISTS last_scrape_reviews_filtered INT,
    ADD COLUMN IF NOT EXISTS last_scrape_date_dropped INT,
    ADD COLUMN IF NOT EXISTS last_scrape_duration_ms INT,
    ADD COLUMN IF NOT EXISTS last_scrape_resume_page INT;

DO $$
BEGIN
    ALTER TABLE b2b_scrape_targets
        ADD CONSTRAINT chk_last_scrape_runtime_mode
        CHECK (
            last_scrape_runtime_mode IS NULL
            OR last_scrape_runtime_mode IN ('initial', 'incremental')
        );
EXCEPTION WHEN duplicate_object THEN
    NULL;
END
$$;

UPDATE b2b_scrape_targets
SET
    last_scrape_runtime_mode = COALESCE(
        last_scrape_runtime_mode,
        NULLIF(metadata->'scrape_state'->>'runtime_mode', '')
    ),
    last_scrape_stop_reason = COALESCE(
        last_scrape_stop_reason,
        NULLIF(metadata->'scrape_state'->>'stop_reason', '')
    ),
    last_scrape_oldest_review = COALESCE(
        last_scrape_oldest_review,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'oldest_review', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'oldest_review')::date
            ELSE NULL
        END
    ),
    last_scrape_newest_review = COALESCE(
        last_scrape_newest_review,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'newest_review', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'newest_review')::date
            ELSE NULL
        END
    ),
    last_scrape_date_cutoff = COALESCE(
        last_scrape_date_cutoff,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'date_cutoff_used', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'date_cutoff_used')::date
            ELSE NULL
        END
    ),
    last_scrape_pages_scraped = COALESCE(
        last_scrape_pages_scraped,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'pages_scraped', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'pages_scraped')::int
            ELSE NULL
        END
    ),
    last_scrape_reviews_found = COALESCE(
        last_scrape_reviews_found,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'reviews_found', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'reviews_found')::int
            ELSE NULL
        END
    ),
    last_scrape_reviews_filtered = COALESCE(
        last_scrape_reviews_filtered,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'reviews_filtered', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'reviews_filtered')::int
            ELSE NULL
        END
    ),
    last_scrape_date_dropped = COALESCE(
        last_scrape_date_dropped,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'date_dropped', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'date_dropped')::int
            ELSE NULL
        END
    ),
    last_scrape_duration_ms = COALESCE(
        last_scrape_duration_ms,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'duration_ms', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'duration_ms')::int
            ELSE NULL
        END
    ),
    last_scrape_resume_page = COALESCE(
        last_scrape_resume_page,
        CASE
            WHEN NULLIF(metadata->'scrape_state'->>'resume_page', '') IS NOT NULL
            THEN (metadata->'scrape_state'->>'resume_page')::int
            ELSE NULL
        END
    )
WHERE metadata ? 'scrape_state';
