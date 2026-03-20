-- Category dynamics: canonical per-category market regime and structural
-- dynamics objects consumed by market landscapes, churn reports, and blog content.
-- Keyed by category + date + schema version.

CREATE TABLE IF NOT EXISTS b2b_category_dynamics (
    category             text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    dynamics             jsonb       NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (category, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_category_dynamics_category
    ON b2b_category_dynamics (category);
CREATE INDEX IF NOT EXISTS idx_b2b_category_dynamics_date
    ON b2b_category_dynamics (as_of_date DESC);
