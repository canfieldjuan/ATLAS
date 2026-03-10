-- Phase 3 Sprint 4: Cross-Vendor Trend Correlation
--
-- Adds indexes to support cross-vendor correlation queries:
-- concurrent change events, vendor pair time-series, market-wide trends.

-- Index for concurrent event detection (GROUP BY event_date, event_type)
CREATE INDEX IF NOT EXISTS idx_bce_date_type
    ON b2b_change_events (event_date, event_type);

-- Index for efficient snapshot cross-join by date
CREATE INDEX IF NOT EXISTS idx_bvs_date_vendor
    ON b2b_vendor_snapshots (snapshot_date, vendor_name);

-- Index for displacement edge time-series correlation
CREATE INDEX IF NOT EXISTS idx_displacement_edges_date_pair
    ON b2b_displacement_edges (computed_date, from_vendor, to_vendor);
