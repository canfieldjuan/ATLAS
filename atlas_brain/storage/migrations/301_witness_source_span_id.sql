-- Persist source_span_id on witnesses so the API can do exact span matching
-- instead of fuzzy text overlap against enrichment evidence_spans.
ALTER TABLE b2b_vendor_witnesses
    ADD COLUMN IF NOT EXISTS source_span_id text;
