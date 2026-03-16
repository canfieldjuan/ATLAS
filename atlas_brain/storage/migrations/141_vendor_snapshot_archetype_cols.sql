-- Add archetype tracking to vendor snapshots for temporal archetype history.
-- Enables queries like "what was this vendor's archetype 4 weeks ago?"
ALTER TABLE b2b_vendor_snapshots
    ADD COLUMN IF NOT EXISTS archetype            TEXT,
    ADD COLUMN IF NOT EXISTS archetype_confidence  NUMERIC(4,3);
