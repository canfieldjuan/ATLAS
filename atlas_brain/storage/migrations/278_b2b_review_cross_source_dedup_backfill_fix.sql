-- Fix cross-source review dedup backfill values for already-migrated databases.
--
-- Migration 277 introduced the columns and a first backfill, but historical
-- reviewer keys need to lowercase before stripping punctuation so uppercase
-- letters are preserved, and existing rows without raw_metadata hashes still
-- need content hashes computed from review text.

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
WHERE reviewed_at IS NOT NULL
  AND coalesce(trim(reviewer_name), '') <> '';
