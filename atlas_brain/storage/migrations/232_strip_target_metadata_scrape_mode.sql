-- Remove legacy persisted runtime-mode override from scrape target metadata
-- Migration 232

UPDATE b2b_scrape_targets
SET metadata = metadata - 'scrape_mode'
WHERE metadata ? 'scrape_mode';
