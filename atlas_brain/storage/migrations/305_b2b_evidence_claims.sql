-- Phase 9: EvidenceClaim contract shadow table.
--
-- One row per (artifact x witness x claim_type x targets) tuple. Captures
-- the result of validate_claim() at synthesis / intelligence build time so
-- consumers can ask "what is the best validated claim of type T about
-- vendor V" without re-running validation. Also doubles as the audit log
-- the rollout sequence uses to flip consumers off legacy witness picks.
--
-- See:
--   docs/progress/evidence_claim_contract_plan_2026-04-25.md
--   atlas_brain/services/b2b/evidence_claim.py            (validator)
--   atlas_brain/services/b2b/evidence_claim_repository.py (writer + selector)
--
-- artifact_type discriminator avoids partial-index gymnastics. Both
-- 'synthesis' and 'intelligence' rows share one uniqueness key so replays
-- are idempotent regardless of which artifact owns them.
--
-- Generated rank columns (grounding_rank, pain_confidence_rank) are stored
-- so the partial select_best_claim index can ORDER BY without a CASE
-- expression, which Postgres won't compose with index ordering.
--
-- source_excerpt_fingerprint is computed at write time as a hash of
-- (source_review_id, normalized excerpt_text). Used for cross-claim-type
-- dedup when a single phrase legitimately validates for multiple
-- claim_types.

CREATE TABLE IF NOT EXISTS b2b_evidence_claims (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Provenance: every claim is tied to exactly one source artifact.
    artifact_type text NOT NULL CHECK (artifact_type IN ('synthesis', 'intelligence')),
    artifact_id uuid NOT NULL,

    -- Convenience nullable cross-references kept for join readability.
    -- One of these mirrors artifact_id depending on artifact_type.
    synthesis_id uuid NULL,
    intelligence_id uuid NULL,

    vendor_name text NOT NULL,
    as_of_date date NULL,
    analysis_window_days integer NULL,

    claim_schema_version text NOT NULL DEFAULT 'v1',
    claim_type text NOT NULL,
    target_entity text NOT NULL,
    secondary_target text NULL,

    -- Witness anchor. NULL for synthesized spans that have no witness row.
    witness_id text NULL,
    witness_hash text NULL,
    source_review_id uuid NULL,
    source_span_id text NULL,

    -- Top-level ranking columns. Duplicated from claim_payload so the
    -- partial index below can do an ordered top-N scan without
    -- materializing JSONB.
    salience_score numeric NOT NULL DEFAULT 0,
    grounding_status text NULL,
    pain_confidence text NULL,

    -- Generated ranking columns. Lower is better for both: 0 = best.
    grounding_rank smallint GENERATED ALWAYS AS (
        CASE WHEN grounding_status = 'grounded' THEN 0 ELSE 1 END
    ) STORED,
    pain_confidence_rank smallint GENERATED ALWAYS AS (
        CASE pain_confidence
            WHEN 'strong' THEN 0
            WHEN 'weak'   THEN 1
            ELSE 2
        END
    ) STORED,

    -- Stable fingerprint for cross-claim-type dedup. Always use this
    -- column name end-to-end. Note that witness_hash means something
    -- different (it includes witness_type and signal_tags) and is not
    -- a substitute.
    source_excerpt_fingerprint text NULL,

    status text NOT NULL,
    rejection_reason text NULL,
    supporting_fields jsonb NOT NULL DEFAULT '[]'::jsonb,
    claim_payload jsonb NOT NULL DEFAULT '{}'::jsonb,

    -- validated_at is overwritten on each replay; created_at preserves
    -- first-seen so the audit can answer "when did we first observe
    -- this claim".
    validated_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Idempotency on replay across BOTH artifact types. A given
-- (artifact_type, artifact_id, witness_id, claim_type, target_entity,
-- secondary_target) produces exactly one row whose status /
-- rejection_reason / supporting_fields reflect the latest validation
-- result. Replays update validated_at; created_at is preserved.
--
-- COALESCE wraps witness_id and secondary_target so NULL and '' compare
-- equal in the index. Without COALESCE, two NULL rows would not collide
-- (NULLs are distinct in B-tree unique indexes).
CREATE UNIQUE INDEX IF NOT EXISTS uq_b2b_evidence_claims_replay
    ON b2b_evidence_claims (
        artifact_type,
        artifact_id,
        COALESCE(witness_id, ''),
        claim_type,
        target_entity,
        COALESCE(secondary_target, '')
    );

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_vendor_created
    ON b2b_evidence_claims (vendor_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_synthesis
    ON b2b_evidence_claims (synthesis_id) WHERE synthesis_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_intelligence
    ON b2b_evidence_claims (intelligence_id) WHERE intelligence_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_status
    ON b2b_evidence_claims (status, claim_type);

-- Partial index covering the select_best_claim hot path. The function
-- signature filters by (vendor_name, claim_type, target_entity, as_of_date,
-- analysis_window_days) and orders by (salience_score DESC, grounding_rank,
-- pain_confidence_rank, witness_id). All filter and order columns are in
-- the index, so Postgres can do an index-backed ordered top-N scan with
-- heap fetches only for the LIMIT-N rows it returns.
CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_select_best
    ON b2b_evidence_claims (
        vendor_name,
        claim_type,
        target_entity,
        as_of_date,
        analysis_window_days,
        salience_score DESC,
        grounding_rank,
        pain_confidence_rank,
        witness_id
    )
    WHERE status = 'valid';

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_dedup
    ON b2b_evidence_claims (source_excerpt_fingerprint)
    WHERE status = 'valid' AND source_excerpt_fingerprint IS NOT NULL;
