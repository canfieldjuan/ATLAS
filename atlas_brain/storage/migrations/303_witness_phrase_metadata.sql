-- Phase 5: surface phrase metadata + per-review pain_confidence on witness
-- rows so the API can render confidence on the UI without re-querying the
-- enrichment row for every witness.
--
-- All five columns are nullable because:
--   - v1 enrichments have no phrase_metadata (legacy rows pre-Phase 1a)
--   - synthesized spans (event, competitor_pressure) carry no phrase tags
--   - the backfill (scripts/backfill_witness_phrase_metadata.py) leaves
--     un-tagged rows at NULL rather than stamping a fake default
--
-- phrase_polarity values: 'negative' | 'mixed' | 'positive' | 'unclear'
-- phrase_subject  values: 'subject_vendor' | 'self' | 'alternative' |
--                         'third_party' | 'unclear'
-- phrase_role     values: 'primary_driver' | 'supporting_context' |
--                         'passing_mention' | 'unclear'
-- phrase_verbatim is a tri-state bool: true / false / null (unknown)
-- pain_confidence values: 'strong' | 'weak' | 'none'
--
-- We do NOT install CHECK constraints. Phase 1a writes phrase_metadata
-- as a JSONB list and the prompt is the source-of-truth for valid enum
-- values; a CHECK constraint here would create a tight coupling to
-- LLM-side enum stability and could reject valid (if unrecognized)
-- values. Read-side code already gates on known values; unknown values
-- are simply ignored.

ALTER TABLE b2b_vendor_witnesses
    ADD COLUMN IF NOT EXISTS phrase_polarity text NULL,
    ADD COLUMN IF NOT EXISTS phrase_subject  text NULL,
    ADD COLUMN IF NOT EXISTS phrase_role     text NULL,
    ADD COLUMN IF NOT EXISTS phrase_verbatim boolean NULL,
    ADD COLUMN IF NOT EXISTS pain_confidence text NULL;

-- Partial index: the API filters on rows that have phrase tags so it can
-- decide whether to render the confidence banner. Once the backfill
-- drains v2-eligible rows the index is small.
CREATE INDEX IF NOT EXISTS idx_b2b_vendor_witnesses_pain_confidence
    ON b2b_vendor_witnesses (pain_confidence)
    WHERE pain_confidence IS NOT NULL;
