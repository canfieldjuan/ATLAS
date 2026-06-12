-- Tenant-scoped calibration library for Content Ops review.
--
-- The extracted calibration library (slice 5a) owns the deterministic anchor
-- types and selection. This host table stores curated calibration examples per
-- account so the marketer verify flow can read a tenant's anchors server-side
-- instead of the connector resending them on every call. Only teachable anchors
-- are stored (excerpt + reasoning required), since a non-teachable anchor cannot
-- illustrate a failure mode to the editor.

CREATE TABLE IF NOT EXISTS content_ops_calibration_library (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id        UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    example_id        TEXT NOT NULL,
    label             TEXT NOT NULL,
    excerpt           TEXT NOT NULL,
    reasoning         TEXT NOT NULL,
    source            TEXT NOT NULL DEFAULT 'curated',
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at       TIMESTAMPTZ,
    CONSTRAINT chk_content_ops_calibration_library_example_id
        CHECK (btrim(example_id) <> ''),
    CONSTRAINT chk_content_ops_calibration_library_excerpt
        CHECK (btrim(excerpt) <> ''),
    CONSTRAINT chk_content_ops_calibration_library_reasoning
        CHECK (btrim(reasoning) <> ''),
    CONSTRAINT chk_content_ops_calibration_library_label
        CHECK (
            label IN (
                'approved', 'rejected', 'borderline', 'known_defect',
                'good_voice', 'voice_drift', 'overclaim', 'weak_persuasion',
                'strong_persuasion'
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_content_ops_calibration_library_account_active
    ON content_ops_calibration_library (account_id, updated_at DESC)
    WHERE archived_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_content_ops_calibration_library_account_example_id_active
    ON content_ops_calibration_library (account_id, lower(btrim(example_id)))
    WHERE archived_at IS NULL;
