-- Track cross-source duplicate count per scrape run.
--
-- Used by the pre-scrape coverage gate in b2b_scrape_intake to decide
-- whether a target's recent runs have produced enough cross-source
-- duplicates to justify skipping the next paid scrape.
--
-- Old rows default to 0 -- the gate refuses to fire until enough fresh
-- post-deploy runs accumulate, which is the conservative cold-start.

ALTER TABLE b2b_scrape_log
    ADD COLUMN IF NOT EXISTS cross_source_duplicates INT NOT NULL DEFAULT 0;
