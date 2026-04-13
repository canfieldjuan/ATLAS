-- Canonical review-to-vendor associations.
--
-- Keep one stored source review row per source item while preserving the
-- vendor graph needed by downstream B2B intelligence.

CREATE TABLE IF NOT EXISTS b2b_review_vendor_mentions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id       UUID NOT NULL REFERENCES b2b_reviews(id) ON DELETE CASCADE,
    vendor_name     TEXT NOT NULL,
    is_primary      BOOLEAN NOT NULL DEFAULT FALSE,
    match_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_review_vendor_mentions_unique
    ON b2b_review_vendor_mentions (review_id, vendor_name);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_review_vendor_mentions_primary
    ON b2b_review_vendor_mentions (review_id)
    WHERE is_primary;

CREATE INDEX IF NOT EXISTS idx_b2b_review_vendor_mentions_vendor
    ON b2b_review_vendor_mentions (vendor_name, review_id);

INSERT INTO b2b_review_vendor_mentions (review_id, vendor_name, is_primary, match_metadata)
SELECT r.id,
       r.vendor_name,
       TRUE,
       jsonb_build_object(
           'source', 'migration_297_primary_vendor',
           'imported_at', to_char(COALESCE(r.imported_at, NOW()) AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
       )
FROM b2b_reviews r
WHERE COALESCE(BTRIM(r.vendor_name), '') <> ''
ON CONFLICT (review_id, vendor_name) DO UPDATE
SET is_primary = EXCLUDED.is_primary,
    match_metadata = COALESCE(b2b_review_vendor_mentions.match_metadata, '{}'::jsonb) || EXCLUDED.match_metadata,
    updated_at = NOW();
