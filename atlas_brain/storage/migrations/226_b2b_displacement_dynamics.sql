-- Displacement dynamics: canonical per-vendor-pair competitive displacement
-- objects consumed by battle cards (competitive_reframes, migration_proof),
-- challenger briefs, and vendor-vs-vendor comparisons.
-- Keyed by (from_vendor, to_vendor) pair + date + schema version.

CREATE TABLE IF NOT EXISTS b2b_displacement_dynamics (
    from_vendor          text        NOT NULL,
    to_vendor            text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    dynamics             jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (from_vendor, to_vendor, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_displacement_dynamics_from
    ON b2b_displacement_dynamics (from_vendor);
CREATE INDEX IF NOT EXISTS idx_b2b_displacement_dynamics_to
    ON b2b_displacement_dynamics (to_vendor);
CREATE INDEX IF NOT EXISTS idx_b2b_displacement_dynamics_date
    ON b2b_displacement_dynamics (as_of_date DESC);
