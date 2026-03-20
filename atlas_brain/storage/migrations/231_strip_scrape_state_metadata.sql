-- Remove duplicated runtime checkpoint state from scrape target metadata
-- Migration 231

UPDATE b2b_scrape_targets
SET metadata = metadata - 'scrape_state'
WHERE metadata ? 'scrape_state';
