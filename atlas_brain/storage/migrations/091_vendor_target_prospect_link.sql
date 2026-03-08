-- Migration 091: Link vendor_targets to prospects for enrichment tracking
--
-- prospect_id: the prospect record backing this vendor target contact
-- contact_enriched_at: timestamp for staleness checks (NULL = manually set)

ALTER TABLE vendor_targets
    ADD COLUMN IF NOT EXISTS prospect_id UUID REFERENCES prospects(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS contact_enriched_at TIMESTAMPTZ;
