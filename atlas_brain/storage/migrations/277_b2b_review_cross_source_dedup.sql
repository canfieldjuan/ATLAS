-- Cross-source review dedup foundation for B2B review ingest.
--
-- Keep raw source rows for provenance, but allow ingest to:
--   * link duplicate rows to a canonical review
--   * skip enrichment on duplicate rows
--   * query vendor-wide duplicate candidates efficiently

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS cross_source_content_hash TEXT,
    ADD COLUMN IF NOT EXISTS cross_source_identity_key TEXT,
    ADD COLUMN IF NOT EXISTS duplicate_of_review_id UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS duplicate_reason TEXT,
    ADD COLUMN IF NOT EXISTS deduped_at TIMESTAMPTZ;

UPDATE b2b_reviews
SET cross_source_content_hash = raw_metadata->>'review_content_hash'
WHERE cross_source_content_hash IS NULL
  AND raw_metadata ? 'review_content_hash'
  AND raw_metadata->>'review_content_hash' IS NOT NULL
  AND raw_metadata->>'review_content_hash' != '';

UPDATE b2b_reviews
SET cross_source_identity_key = lower(trim(vendor_name))
    || '|'
    || regexp_replace(lower(coalesce(reviewer_name, '')), '[^a-z0-9]+', '', 'g')
    || '|'
    || to_char(reviewed_at AT TIME ZONE 'UTC', 'YYYY-MM-DD')
    || '|'
    || CASE
        WHEN rating IS NOT NULL THEN trim(to_char(rating, 'FM999999990.0'))
        ELSE ''
    END
WHERE cross_source_identity_key IS NULL
  AND reviewed_at IS NOT NULL
  AND coalesce(trim(reviewer_name), '') <> '';

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_cross_source_hash
    ON b2b_reviews (vendor_name, cross_source_content_hash)
    WHERE cross_source_content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_cross_source_identity
    ON b2b_reviews (vendor_name, cross_source_identity_key)
    WHERE cross_source_identity_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_duplicate_of
    ON b2b_reviews (duplicate_of_review_id)
    WHERE duplicate_of_review_id IS NOT NULL;
