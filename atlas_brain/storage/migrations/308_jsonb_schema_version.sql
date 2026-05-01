-- Schema versioning for the email-campaign JSONB blobs.
--
-- Pairs with atlas_brain/schemas/campaigns.py. Each model carries a
-- schema_version field defaulted to 1; the column on the table mirrors that
-- so future migrations can fan out reads/writes by version without parsing
-- the JSON. Defaults to 1 so existing rows are tagged as v1 implicitly.

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS schema_version INT NOT NULL DEFAULT 1;

ALTER TABLE campaign_sequences
    ADD COLUMN IF NOT EXISTS schema_version INT NOT NULL DEFAULT 1;

ALTER TABLE b2b_vendor_briefings
    ADD COLUMN IF NOT EXISTS schema_version INT NOT NULL DEFAULT 1;
