# Evidence Claim Contract Plan

Date: 2026-04-25

## Decision

Introduce a validated evidence-claim layer between witness selection and
downstream report rendering.

Do not patch individual consumers first. The Monday.com canary showed that the
same weak evidence can leak through multiple downstream surfaces because each
consumer currently interprets witness packs or quote arrays independently.

The durable fix is:

1. keep raw extraction and legacy phrase arrays as compatibility inputs
2. keep witness packets as candidate evidence
3. add validated evidence claims in shadow mode
4. move downstream consumers to claim queries one surface at a time
5. only remove legacy paths after coverage and audit gates are met

## Codebase Validation

No `EvidenceClaim` contract exists today.

Closest existing layers:

- `atlas_brain/services/b2b/enrichment_contract.py`
  - gates enrichment-level pain category and phrase quality
  - does not model claim type, target entity, or report-safe usage
- `atlas_brain/autonomous/tasks/_b2b_phrase_metadata.py`
  - reads phrase metadata
  - does not validate a phrase as a claim for a specific surface
- `atlas_brain/autonomous/tasks/_b2b_witnesses.py`
  - builds candidate spans and witness packs
  - does not produce a report-safe claim object
- `scripts/audit_witness_field_propagation.py`
  - verifies field propagation
  - does not verify semantic claim integrity

The current layers are:

1. raw enrichment JSONB
2. candidate evidence spans
3. selected witness packs

The missing layer is:

4. validated claims with explicit claim type, target entity, validity, and
   rejection reason

## Canary Failures This Must Prevent

Monday.com scoped synthesis exposed four failure modes.

1. Competitor attribution leak
   - Review context: "Before monday.com, we were using HubSpot, which ... does
     not provide a very good user interface..."
   - Current output: tagged as Monday.com `ux` pain with
     `subject_vendor`, `negative`, `primary_driver`, `strong`
   - Correct behavior: reject pain claim about Monday.com; the negative phrase
     targets HubSpot.

2. Positive quote used as pain evidence
   - Scorecard/deep-dive evidence uses a positive G2 quote about SEO/CMS
     workflow visibility.
   - Report claim discusses churn/UX friction.
   - Correct behavior: positive/use-case quotes cannot support a pain claim.

3. Passing mention used as named-account anchor
   - Witness has `phrase_role=passing_mention` and `pain_confidence=none`.
   - Battle card still uses it as an anchor.
   - Correct behavior: passing mentions are not report-safe anchors.

4. Suppressed section still rendered
   - Battle card `evidence_conclusions.suppressed_sections` includes
     `executive_summary`.
   - `executive_summary` is still populated and persisted.
   - Correct behavior: suppression must be enforced before render/persist.
   - Ownership note: this is a *render-time* policy, not a claim-validity
     question. The EvidenceClaim contract does not own suppressed-section
     enforcement directly. Each consumer that migrates to the claim contract
     must also gate its render output on `evidence_conclusions.suppressed_sections`.
     The semantic audit (defined below) tests for both: claim-level
     attribution and consumer-level suppression enforcement, but the fixes
     live in different code paths.

## Claim Taxonomy

Start with a closed enum. Do not let consumers invent string claim names.

Proposed `ClaimType`:

- `pain_claim_about_vendor`
- `counterevidence_about_vendor`
- `displacement_proof_to_competitor`
- `displacement_proof_from_competitor`
- `named_account_anchor`
- `pricing_urgency_claim`
- `feature_gap_claim`
- `support_failure_claim`
- `timing_pressure_claim`
- `adoption_or_onboarding_claim`
- `reliability_claim`
- `integration_or_workflow_claim`

Claim types should be added only when there is:

- a validation gate
- at least one downstream consumer
- an acceptance fixture

## Claim Validation API

New module:

`atlas_brain/services/b2b/evidence_claim.py`

Core API:

```python
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

class ClaimType(StrEnum):
    PAIN_CLAIM_ABOUT_VENDOR = "pain_claim_about_vendor"
    COUNTEREVIDENCE_ABOUT_VENDOR = "counterevidence_about_vendor"
    DISPLACEMENT_PROOF_TO_COMPETITOR = "displacement_proof_to_competitor"
    DISPLACEMENT_PROOF_FROM_COMPETITOR = "displacement_proof_from_competitor"
    NAMED_ACCOUNT_ANCHOR = "named_account_anchor"
    PRICING_URGENCY_CLAIM = "pricing_urgency_claim"
    FEATURE_GAP_CLAIM = "feature_gap_claim"
    SUPPORT_FAILURE_CLAIM = "support_failure_claim"
    TIMING_PRESSURE_CLAIM = "timing_pressure_claim"
    ADOPTION_OR_ONBOARDING_CLAIM = "adoption_or_onboarding_claim"
    RELIABILITY_CLAIM = "reliability_claim"
    INTEGRATION_OR_WORKFLOW_CLAIM = "integration_or_workflow_claim"

class ClaimValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    CANNOT_VALIDATE = "cannot_validate"

@dataclass(frozen=True)
class ClaimValidation:
    claim_type: ClaimType
    status: ClaimValidationStatus
    rejection_reason: str | None
    supporting_fields: tuple[str, ...]
    target_entity: str
    source_witness_id: str | None

def validate_claim(
    *,
    claim_type: ClaimType,
    witness: dict[str, Any],
    target_entity: str,
    secondary_target: str | None = None,
    source_review: dict[str, Any] | None = None,
) -> ClaimValidation:
    ...
```

`secondary_target` is required for claim types that name a second entity:

- `displacement_proof_to_competitor`: secondary_target = the competitor the
  vendor is losing share to
- `displacement_proof_from_competitor`: secondary_target = the competitor the
  vendor is winning share from
- `feature_gap_claim` (when comparing to a named competitor)

For other claim types `secondary_target` should be `None` and the validator
must reject claims that pass an unexpected secondary_target.

`cannot_validate` is not the same as `invalid`.

- `invalid`: metadata exists and proves the witness should not support the
  claim.
- `cannot_validate`: required metadata is absent, usually v3-backed or
  synthesized evidence. This is tracked separately in audit output.

### Best-Evidence Selection API

Validation answers "is this witness valid for this claim?" — but consumers
need "what is the best witness for this claim type for this vendor?" That
is a separate API that reads the shadow table.

The selection API is split from `validate_claim()` so the validation
layer stays a pure deterministic function (no DB pool, no I/O) and the
DB-bound query layer lives in a repository. This matches the rest of
the codebase, which is async-first and uses asyncpg pools.

Validation function (pure, sync, callable from tests without a pool):

```python
def validate_claim(
    *,
    claim_type: ClaimType,
    witness: dict[str, Any],
    target_entity: str,
    secondary_target: str | None = None,
    source_review: dict[str, Any] | None = None,
) -> ClaimValidation:
    ...  # deterministic gates + antecedent regex; no DB
```

Selection API (async, takes pool, lives in
`atlas_brain/services/b2b/evidence_claim_repository.py`):

```python
async def select_best_claim(
    pool,
    *,
    claim_type: ClaimType,
    target_entity: str,
    secondary_target: str | None = None,
    vendor_name: str,
    as_of_date: date,
    analysis_window_days: int,
    limit: int = 1,
) -> list[ClaimSelection]:
    """Return the highest-quality validated claims for the given query.

    Reads only `b2b_evidence_claims` rows where status = VALID. Ranks by
    top-level columns so the partial index covers the ORDER BY:
      1. salience_score DESC
      2. grounding_status = 'grounded' first (boolean lift)
      3. pain_confidence priority: strong > weak > none/null
      4. witness_id ASC for stable tie-break

    Dedups across multi-claim-per-witness rows by
    source_excerpt_fingerprint so a single phrase that validated for
    several claim types is not returned twice for the same call.

    Returns empty list (not None) when no valid claim exists. Consumer
    decides whether to fall back to legacy witness picks or render a
    "no quote available" placeholder.
    """
```

`ClaimSelection` is a thin frozen dataclass wrapping a row's
`claim_payload` + the joined witness data needed to render. Keep the
shape stable so consumers don't reach back into `b2b_vendor_witnesses`.

The repository module also owns the writer:

```python
async def upsert_claim(pool, claim: PersistedClaim) -> None:
    """Idempotent INSERT ... ON CONFLICT ... DO UPDATE on the
    (artifact_type, artifact_id, witness_id, claim_type, target_entity,
    secondary_target) replay key."""
```

Validation lives in the service module; persistence lives in the
repository. Consumers depend on the repository's async API. The split
keeps unit tests for `validate_claim()` pool-free and lets repository
tests exercise the real DB schema.

This API is consumed by the migrated consumers (battle_cards, vendor_briefing,
etc). It must be specified BEFORE consumer migration starts (rollout step 8),
even though shadow-mode capture (rollout steps 1-7) does not require it.

## Validation Gates

Baseline gates that apply to every report-safe claim:

- `grounding_status == "grounded"` unless the claim type explicitly permits
  non-verbatim theme evidence
- `phrase_role != "passing_mention"`
- `pain_confidence != "none"` for pain-like claims
- target entity can be resolved
- witness has stable `witness_id`

Pain claim gates:

- `phrase_subject == "subject_vendor"`
- `phrase_polarity in {"negative", "mixed"}`
- `phrase_role == "primary_driver"` for headline or anchor usage
- `pain_confidence in {"strong", "weak"}`
- `pain_category` is present
- generic categories are disallowed for headline claims unless explicitly
  requested

Counterevidence gates:

- `phrase_subject == "subject_vendor"`
- `phrase_polarity == "positive"`
- `grounding_status == "grounded"`
- `phrase_role != "passing_mention"`

Displacement-to-competitor gates:

- competitor or alternative is present
- evidence type indicates active evaluation, explicit switch, migration, or
  named alternative with reason
- reverse-flow evidence is rejected
- neutral mention is rejected
- subject attribution cannot contradict target vendor

Named-account anchor gates:

- reviewer company is present and passes existing company-signal exclusion
  rules
- `phrase_role != "passing_mention"`
- evidence is grounded
- claim is not based only on a title or summary fragment

Timing-pressure gates:

- specific time anchor or urgency indicator is present
- witness is grounded
- timing signal is about the target vendor or target account, not a generic
  third-party market observation

### Multi-claim-per-witness handling

A single witness can be valid for multiple claim types simultaneously. A
phrase like "support is broken and we missed our renewal deadline" supports
`pain_claim_about_vendor` (pain), `support_failure_claim`, and
`timing_pressure_claim` at once. The shadow table allows this — one row per
(witness × claim_type) tuple.

Consumer-side deduplication is the responsibility of `select_best_claim()`:

- when querying a SINGLE claim_type, the claim row is the natural unit
- when a consumer assembles a mixed-claim list (e.g. a battle card showing
  both `pain_claim_about_vendor` and `support_failure_claim` quotes),
  dedup by `(source_review_id, excerpt_text_fingerprint)` so a single phrase
  isn't shown twice under different claim labels
- the witness_hash field provides a stable fingerprint for this purpose

Validation never collapses claim types — a phrase that legitimately
supports three claim types should produce three claim rows with three
different rejection reasons (or three VALIDs). Audit metrics need the
breakdown to surface "which claim type rejects most often."

## Source-Window Attribution

Phrase tags alone are not enough. Validation must read the source window around
the phrase when available.

Required inputs:

- `source_span_id`
- `review_text`
- `summary`
- `start_char`
- `end_char`
- `excerpt_text`

The validator should recover a bounded source window and reject obvious
antecedent traps:

- "Before [subject vendor], we used [competitor], which had [negative trait]"
- "Unlike [subject vendor], [competitor] is [negative trait]"
- "[competitor] is cheaper/better/worse" when the claim targets subject vendor
- self-built/in-house alternatives where the cost or praise belongs to self

This should start deterministic and conservative. If the deterministic window
cannot decide, return `cannot_validate`, not `valid`.

### Antecedent Pattern Set (v1)

Exact patterns the v1 deterministic validator checks against the source
window before validating a pain claim. All matches are case-insensitive,
applied to the sentence containing the phrase plus the immediately
preceding sentence. `\[VENDOR\]` is a placeholder for the target vendor
name (with normalized aliases).

Each pattern requires BOTH a temporal/comparative marker AND a competitor
or self-transition phrase. A bare `before [VENDOR]` is not an antecedent
trap on its own — "Before Monday introduced this feature, our team
struggled" is a valid temporal statement about the subject vendor. The
trap fires only when the source window is talking about a *different*
entity (a competitor or the reviewer's own org).

Reject (return `invalid` with `rejection_reason='antecedent_trap'`) when
the phrase target is `subject_vendor` AND the source window matches one
of these full patterns:

```
\b(before|prior to|previously|originally|initially|formerly)\s+\[VENDOR\][^.]{0,80}\bwe\s+(used|had|were on|were using|relied on|ran)\b
\b(before|prior to|previously|originally|initially|formerly)\s+\[VENDOR\][^.]{0,80}\bour\s+(team|company|org|stack)\b
\bunlike\s+\[VENDOR\][^.]{0,80}\b\[COMPETITOR\]\b
\b(switched|moved|migrated|graduated|upgraded)\s+(from|away from)\s+\[COMPETITOR\][^.]{0,80}\bto\s+\[VENDOR\]\b
\bwe\s+(used to use|used to be on|came from|moved off)\s+\[COMPETITOR\][^.]{0,80}
\b\[COMPETITOR\]\s+(was our|were our|used to be)\b
```

`\[COMPETITOR\]` matches any vendor name from the canonical-aliases table
that is NOT the subject vendor. Each pattern fires when the source window
contains `\[VENDOR\]` AND ONE OF:

- a self-transition phrase (`we used`, `we had`, `our team`, `our company`,
  `our stack`) — the negative trait may belong to the reviewer's prior
  setup or a homegrown system, not necessarily a named competitor.
  Patterns 1-2 cover this case.
- a `\[COMPETITOR\]` token — the negative trait belongs to a named
  competitor. Patterns 3-6 cover this case.

The trap fires only when these markers co-occur with `\[VENDOR\]` inside
the same source window. A bare `before [VENDOR]` is not a trap on its
own; "Before Monday introduced this feature, our team struggled" matches
no pattern (no transition or competitor marker after the temporal
marker) and falls through to the per-phrase gates as a legitimate
temporal statement about the subject vendor.

When the phrase target is a competitor (used by the displacement claim
validator), the patterns swap roles: the trap fires when the negative
trait belongs to the subject_vendor but the displacement claim is about
the competitor.

Anything not matching deterministic patterns falls through to the existing
phrase-level subject/polarity gates. If those gates pass but the validator
detects ambiguous attribution (multiple vendor names within the source
window with no clear antecedent marker AND no matching transition pattern),
return `cannot_validate` with `rejection_reason='ambiguous_attribution'`.
Do NOT return `valid` in ambiguous cases — the claim contract's whole
point is to be safer than the per-phrase gates.

LLM-grade attribution checks (anything beyond the pattern set above) are
explicitly out of scope for v1. The audit will surface `cannot_validate`
rates per source so we can identify which review platforms produce the
most ambiguous attributions and decide whether v2 needs an LLM pass.

## Shadow Table

New migration:

`atlas_brain/storage/migrations/3xx_b2b_evidence_claims.sql`

Proposed schema:

```sql
CREATE TABLE IF NOT EXISTS b2b_evidence_claims (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Provenance: every claim is tied to exactly one source artifact.
    -- artifact_type discriminates (synthesis | intelligence) so uniqueness
    -- and joins can be expressed without partial-index gymnastics.
    artifact_type text NOT NULL CHECK (artifact_type IN ('synthesis', 'intelligence')),
    artifact_id uuid NOT NULL,
    -- Convenience nullable cross-references kept for join readability.
    synthesis_id uuid NULL,
    intelligence_id uuid NULL,
    vendor_name text NOT NULL,
    as_of_date date NULL,
    analysis_window_days integer NULL,
    claim_schema_version text NOT NULL DEFAULT 'v1',
    claim_type text NOT NULL,
    target_entity text NOT NULL,
    secondary_target text NULL,
    witness_id text NULL,
    witness_hash text NULL,
    source_review_id uuid NULL,
    source_span_id text NULL,
    -- Top-level ranking columns. Duplicated from claim_payload so the
    -- select_best_claim query plan can ORDER BY without materialising
    -- JSONB and so the partial index below actually accelerates ranking.
    salience_score numeric NOT NULL DEFAULT 0,
    grounding_status text NULL,
    pain_confidence text NULL,
    -- Stable fingerprint for cross-claim-type dedup in select_best_claim.
    -- Computed at write time as a normalized hash of source_review_id +
    -- excerpt_text. Lets a phrase that legitimately validates for three
    -- claim types be deduped to one row when a consumer assembles a mixed
    -- list.
    source_excerpt_fingerprint text NULL,
    status text NOT NULL,
    rejection_reason text NULL,
    supporting_fields jsonb NOT NULL DEFAULT '[]'::jsonb,
    claim_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    validated_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Idempotency on replay across BOTH artifact types. A given
-- (artifact_type, artifact_id, witness_id, claim_type, target_entity,
-- secondary_target) produces exactly one row whose status / rejection_reason
-- / supporting_fields reflect the latest validation result. Replays update
-- validated_at; created_at preserves the first-seen timestamp.
--
-- The COALESCE wraps secondary_target so NULL and '' compare equal in the
-- index. witness_id is COALESCED for synthesized spans that have no
-- witness anchor.
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

-- Partial index covering the select_best_claim hot path. Includes the
-- ranking columns so Postgres can use an index-only scan for the
-- "valid claims for this vendor + claim_type, ordered by salience" query.
CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_select_best
    ON b2b_evidence_claims (
        vendor_name,
        claim_type,
        target_entity,
        salience_score DESC,
        grounding_status,
        pain_confidence
    )
    WHERE status = 'valid';

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_claims_dedup
    ON b2b_evidence_claims (source_excerpt_fingerprint)
    WHERE status = 'valid' AND source_excerpt_fingerprint IS NOT NULL;
```

Shadow rows are upserted on replay. The unique index now covers BOTH
`synthesis` and `intelligence` artifact types via the `(artifact_type,
artifact_id, ...)` key, eliminating the earlier partial-index gymnastics
and giving intelligence-tied rows the same idempotency guarantees as
synthesis-tied rows.

The `idx_b2b_evidence_claims_select_best` index covers the hot path query
(valid claims for a given vendor + claim_type, ordered by salience +
grounding + pain_confidence). Top-level columns make this a pure-index
operation; ranking does not have to materialise JSONB.

`source_excerpt_fingerprint` powers cross-claim-type dedup in
`select_best_claim()` without forcing consumers to compute it themselves.

## Versioning

Add claim schema version to persisted artifacts:

- `b2b_reasoning_synthesis.synthesis.claim_schema_version`
- `b2b_intelligence.intelligence_data.claim_schema_version`

Do not add a required top-level database column until the JSONB versioning
proves useful. If report-list filtering needs it later, add a generated or
materialized column in a separate migration.

Cache invalidation:

- any artifact with missing or older `claim_schema_version` is stale for
  claim-aware consumers
- old rows remain readable with legacy behavior
- resynthesis or scheduled maintenance can roll rows forward

## Performance Budget

The shadow-mode validator runs synchronously inside synthesis. Cardinality
and latency must be bounded:

- Per-vendor synthesis input: typically 12 selected witnesses, occasionally
  up to 24 candidate witnesses considered.
- Per-witness validation: each witness is evaluated against the claim types
  it could plausibly support (not all 12). Most witnesses match 1-3
  applicable claim types based on their `signal_type` and `pain_category`.
  Worst case: 24 witnesses × 4 applicable types = 96 validations per synthesis.
- Per-validation cost: deterministic Python checks against fields already on
  the witness dict, plus regex match against the source-window patterns.
  Target: <2ms per validation, <200ms total per vendor synthesis.
- Memory: each ClaimValidation is a frozen dataclass; no payload duplication
  beyond the supporting_fields tuple.

Budget caps to enforce:

- if validator wall time exceeds 1 second per vendor synthesis, log a
  warning and surface in audit
- if any single validate_claim() call exceeds 50ms, log a warning with the
  offending claim_type + witness_id

These are observability gates, not hard timeouts — the validator should
never block synthesis from completing.

## API Compatibility (existing quote_grade field)

The current `EvidenceWitnessDetail.quote_grade: boolean` field
(`atlas_brain/api/b2b_evidence.py:272`, `atlas-churn-ui/src/api/client.ts`)
is computed as `(grounding_status = 'grounded')`. It pre-dates the
EvidenceClaim contract.

Decision: keep `quote_grade` as a derived boolean field for backwards
compatibility. Do NOT remove it during shadow mode or initial consumer
migration. The field's semantics (grounded enough to render as a quote)
are claim-type-agnostic and remain useful even after consumers migrate.

When a consumer renders a claim-driven quote, it should additionally check
the claim's `status == VALID` and `claim_payload.grounding_status == 'grounded'`.
The combination is stricter than `quote_grade` alone but the boolean
remains a safe legacy fallback for unmigrated consumers.

Future consideration (out of scope for v1): expose a typed
`claim_validations: list[ClaimValidation]` array on `EvidenceWitnessDetail`
so the UI can show per-claim-type validity. Defer until after at least one
consumer has migrated and we know the actual access pattern.

## Integration Points

### Tier 1 Prompt

File:

`atlas_brain/skills/digest/b2b_churn_extraction_tier1.md`

Current prompt already has subject rules and four examples. It still needs the
Monday/HubSpot antecedent trap.

Add an example where the review target is Monday.com:

- "Before monday.com, we used HubSpot, which did not provide a good UI for
  non-sales users. Monday.com gave tutors a simpler front end."
- HubSpot negative phrase: `subject=alternative`
- Monday positive phrase: `subject=subject_vendor`, `polarity=positive`
- any pain claim about Monday from HubSpot's poor UI must be rejected

This prompt change is isolated and independently valuable, but should be
validated on a small tricky-review set before broad re-enrichment.

### Witness Synthesis Hook

File:

`atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py`

Hook location:

- after `build_vendor_witness_artifacts(...)` has produced witness pack
- before persisted report consumers build from the packet
- before or alongside `_persist_packet_artifacts(...)`

Behavior:

- generate candidate claim rows for every selected witness
- write to `b2b_evidence_claims`
- add `claim_schema_version` to synthesis payload
- do not alter witness selection in shadow mode

### Semantic Audit

File:

`atlas_brain/services/reasoning_delivery_audit.py`

Add:

`summarize_claim_validation(...)`

Metrics — aggregate:

- total claims by type
- valid / invalid / cannot_validate counts (overall and per claim_type)
- rejection reasons by type (top 10 reasons per claim_type)
- invalid claim examples with witness_id and excerpt preview
- cannot_validate share by source schema version
- consumer readiness by report type

Metrics — diagnostic breakdowns (required for canary triage):

- per-vendor: valid / invalid / cannot_validate per claim_type per vendor.
  Surfaces single-vendor regressions during canary or after schema bumps.
- per-source: same breakdown grouped by source platform (G2, Capterra,
  PeerSpot, etc). Different platforms produce different attribution
  quality; this slice is needed to identify which source pipelines need
  Tier 1 prompt or grounding-rule tuning.
- per-pain_category: rejection patterns differ across pain_categories
  (pricing claims fail differently from UX claims). Needed to prioritize
  which gates to tune first if a category shows systemic invalid rate.
- per-rejection_reason: top reasons broken out by claim_type so we know
  whether the antecedent-trap pattern set is rejecting too aggressively
  vs. the polarity gate vs. role gate.

Script:

`scripts/audit_evidence_claims.py`

### Product Priority

The UIs are the primary product. Reports (battle cards, briefings, challenger
briefs, scorecards) are secondary surfaces sold separately or bundled as
upper-management material. Migration order reflects this: the API and the
React UI migrate first so customer-facing surfaces dogfood the contract
immediately; reports follow as server-side consumers of the same API; MCP
is last.

### Cost Guardrail

UI-first migration also gates LLM spend on contract validation. Tier 1
(API + UI) is pure code + DB-read work — no LLM calls, no synthesis
re-runs, no token budget. We can iterate on claim gates and rejection
reasons in the browser at zero per-iteration cost.

Tier 2 (server-side reports) is where money starts: re-generating reports
to consume validated claims may trigger synthesis re-runs (Sonnet 4.5 on
the current Phase 8b config) or downstream cache invalidations. Those costs
should not be incurred until the UI A/B has confirmed the contract gates
are correct.

The rule: we do not burn token budget on report regeneration until the
browser experience validates that the contract is producing the right
selections. If the UI A/B reveals a gate is too strict or too lenient, we
tune the gate first — at zero LLM cost — before any server-side consumer
migration touches synthesis.

### First Consumer Migration

The first migration target is the API endpoint that backs the UI:

File:

`atlas_brain/api/b2b_evidence.py`

Reason:

- the UI is the primary product surface; the API is the only path between
  the contract and what customers see
- list_witnesses + get_witness are the smallest surface that exercises the
  contract end-to-end (HTTP -> service -> shadow table read -> UI)
- introducing a `?claim_type=<type>` filter on list_witnesses gives the UI
  a targeted query that the existing per-witness endpoint never had
- old clients that don't pass claim_type fall back to the existing
  list_witnesses behavior unchanged

API migration rule:

- new optional query parameter `claim_type` on `GET /witnesses`. When set,
  return only witnesses where a validated claim of that type exists for
  the target_entity in `b2b_evidence_claims`.
- new optional response field `claim_validations: list[ClaimValidationDTO]`
  on `GET /witnesses/{id}`. Returns every claim row for that witness,
  enabling the UI to show "this witness validates as a pain claim about
  Monday but invalid as a counterevidence claim because polarity=negative."
- keep `quote_grade: bool` as backwards-compat (see API Compatibility
  section above).

UI migration rule (immediately after API):

- `atlas-churn-ui/src/api/client.ts`: extend `EvidenceWitnessDetail` with
  `claim_validations` array; add `listWitnesses(vendor, {claim_type?})`
  helper.
- `atlas-churn-ui/src/components/EvidenceDrawer.tsx`: surface
  per-claim-type validity banners (in addition to the existing
  pain_confidence and quote_grade banners). When a witness has both
  "valid pain claim" and "invalid counterevidence" entries, the drawer
  shows both with rejection reasons.
- `atlas-churn-ui/src/lib/reportViewModels.ts` and
  `atlas-churn-ui/src/types/reportViewModels.ts`: add
  `ClaimValidationViewModel`; map for the report pages that already
  consume `ReasoningWitnessViewModel`.
- pages that already consume EvidenceDrawer (Opportunities,
  VendorDetail, etc.): no change required — drawer enhancement is
  additive.

### Later Consumer Migration Order

Tier 1 — primary product surface (migrate first, in this order):

1. `atlas_brain/api/b2b_evidence.py` — claim_type filter, claim_validations
   response field. The contract becomes externally visible here.
2. `atlas-churn-ui/src/api/client.ts` + `EvidenceDrawer.tsx` +
   `reportViewModels.ts` — UI consumes the new API shape. Customer-facing
   experience picks up the contract here.

Tier 2 — secondary product surfaces (server-rendered reports, consume the
same service layer the API uses):

3. `b2b_battle_cards.py` — already has `anchor_examples` and
   `witness_highlights`; narrowest server-side surface for hand inspection.
4. `b2b_vendor_briefing.py` + `atlas_brain/templates/email/vendor_briefing.py`
   — email exports of the same data the UI shows.
5. `b2b_challenger_brief.py`
6. `b2b_churn_reports.py` (scorecard + deep dive evidence subsections)

Tier 3 — content generation (claim-aware fallback when select_best_claim is
stable; fall back to legacy otherwise):

7. `b2b_campaign_generation.py`
8. `b2b_blog_post_generation.py`

Tier 4 — broad intelligence_data surface (passthrough; consume validated
claims when present, else legacy. Most of these surfaces are upper-management
exports rather than primary UI surfaces):

9. `b2b_intelligence.report_type` rows not already covered by Tier 2's
   `b2b_churn_reports.py` migration: `accounts_in_motion`,
   `weekly_churn_feed`, `displacement_report`, `category_overview`,
   `churn_alert`, `watchlist_alert`, `tenant_report`.

   Note on `vendor_scorecard`: the **scorecard rendering code** lives in
   `b2b_churn_reports.py` (Tier 2, full migration). The
   `vendor_scorecard` entry in this Tier 4 list refers to the
   *persisted* `b2b_intelligence` rows of `report_type='vendor_scorecard'`
   that pre-date the Tier 2 migration. Tier 4 means "for legacy persisted
   rows of these types, add a claim-aware fallback path so consumers can
   read either old or new schema." It does NOT mean re-migrating the
   live rendering code, which is owned by Tier 2.

Tier 5 — MCP (final):

10. `atlas_brain/mcp/b2b_churn_server.py` — 60+ tools. Add `claim_type`
    parameter to `get_witness`, `list_witnesses`, and the 7 report-type
    fetch tools. MCP migrates last because it is a developer-tooling
    surface, not a primary product surface; consistency with the API
    matters more than priority.

## Acceptance Tests

Create fixtures before changing consumers.

### Fixture Location and Shape

Fixtures live at `tests/fixtures/evidence_claims/<name>.json` and follow
this shape so every fixture round-trips through the validate_claim API:

```json
{
  "name": "antecedent_trap_monday_hubspot",
  "review": {
    "id": "...",
    "vendor_name": "Monday.com",
    "summary": "...",
    "review_text": "Before monday.com, we used HubSpot, which did not provide..."
  },
  "witness": {
    "witness_id": "witness:...",
    "excerpt_text": "did not provide a good UI for non-sales users",
    "phrase_subject": "subject_vendor",
    "phrase_polarity": "negative",
    "phrase_role": "primary_driver",
    "pain_confidence": "strong",
    "grounding_status": "grounded",
    "pain_category": "ux"
  },
  "expected": {
    "pain_claim_about_vendor": {
      "status": "invalid",
      "rejection_reason": "antecedent_trap"
    },
    "displacement_proof_to_competitor": {
      "status": "cannot_validate",
      "rejection_reason": "secondary_target_not_provided"
    }
  }
}
```

Each fixture asserts the expected ClaimValidation for every claim type the
fixture is intended to cover. Tests iterate the dict and call
`validate_claim()` per claim_type; the test fails if any claim's status or
rejection_reason diverges. A `tests/test_evidence_claim_fixtures.py` runner
loads every JSON in the directory and parametrizes the assertions.

### Required Fixtures

1. Competitor antecedent trap
   - review target: Monday.com
   - text: "Before monday.com, we used HubSpot, which did not provide a good
     UI for non-sales users."
   - expected: `pain_claim_about_vendor` for Monday is invalid

2. Positive pricing as pain
   - text: "Good value at $500 per seat."
   - expected: pricing pain claim is invalid
   - counterevidence claim can be valid

3. Passing mention
   - phrase metadata: `role=passing_mention`
   - expected: named-account anchor is invalid

4. V3-backed witness
   - missing phrase metadata
   - expected: status is `cannot_validate`, not `invalid`

5. Synthesized span
   - no source phrase tags
   - expected: status is `cannot_validate`

6. Report-safe pain quote
   - grounded, subject vendor, negative, primary driver, pain confidence weak
     or strong
   - expected: pain claim is valid

7. Shadow mode non-regression
   - existing report output is unchanged
   - `b2b_evidence_claims` rows are written

8. Suppressed-section enforcement
   - `evidence_conclusions.suppressed_sections` includes `executive_summary`
   - claim-aware battle card does not persist a live executive summary

## Impact Analysis

This is additive until consumer migration.

No current behavior changes in shadow mode:

- existing witness selection remains unchanged
- report rendering remains unchanged
- old JSONB rows remain readable
- v3 and synthesized rows become `cannot_validate`, not invalid

Compatibility with phrase schema cleanup:

- keep the six legacy arrays as `list[str]`
- keep `phrase_metadata` parallel
- do not introduce dicts into legacy arrays
- do not delete keyword fallbacks until claim coverage is proven

Risks:

- false negatives if deterministic source-window attribution is too strict
- high `cannot_validate` rate while old v3 rows remain active
- audit-row retention growth: even with upsert idempotency, rows for
  invalid/cannot_validate claims accumulate over time as new witnesses
  arrive. Need a retention policy decision (Open Decisions #1).
- consumer confusion if `invalid` and `cannot_validate` are conflated

Mitigations:

- shadow mode first
- daily audit
- vendor canary set before fanout
- separate retention policy for `b2b_evidence_claims`
- no consumer reads until validation rates are known

## Rollout Sequence

1. Update Tier 1 prompt with the antecedent trap example.
2. Add `EvidenceClaim` design fixtures (per the Fixture Location section).
3. Add `evidence_claim.py` contract and unit tests.
4. Add shadow table migration.
5. Write claim rows from synthesis in shadow mode.
6. Add claim validation audit and daily summary.
7. Run shadow capture for **at least two consecutive nightly synthesis
   cycles spanning a Monday batch run.** Two daily cycles cover the
   per-vendor synthesis path; the Monday batch covers weekly-batch-bound
   consumers (`challenger_brief`, `weekly_churn_feed`) so the capture set
   includes their actual claim distribution before any consumer migrates.

   --- UI-first migration starts here. Tier 1 is browser-iterable at $0/iteration. ---

8. Migrate `atlas_brain/api/b2b_evidence.py` to expose validated claims
   (claim_type filter on list_witnesses + claim_validations field on
   detail). Add behind a feature flag; default off so unmigrated clients
   are unaffected.
9. Migrate `atlas-churn-ui/src/api/client.ts` types and helpers to consume
   the new API shape. Migrate `EvidenceDrawer.tsx` to surface per-claim
   validity banners. Migrate `reportViewModels.ts` for report pages.
10. Hand-audit the canary vendor set in the browser. Iterate on claim
    gates while the UI is the only consumer; LLM cost stays $0 because
    no synthesis re-runs are required to tune deterministic gates.

    --- UI-acceptance gate. Do NOT proceed until 5 of 7 canary vendors
        pass. Server-side migration begins only after browser is clean. ---

11. Migrate `b2b_battle_cards.py` to consume validated claims via
    `select_best_claim()`. Behind feature flag.
12. Hand-audit battle cards on the same canary set.
13. Migrate `b2b_vendor_briefing.py` + email template.
14. Migrate `b2b_challenger_brief.py`.
15. Migrate `b2b_churn_reports.py` (scorecard + deep dive evidence).
16. Migrate Tier 3 content generators (campaign, blog) when
    `select_best_claim()` is stable across all migrated reports.
17. Migrate the remaining `b2b_intelligence` report-type surfaces per the
    Tier 4 enumeration (passthrough fallback for legacy persisted rows).
18. Migrate the MCP server tools per the Tier 5 enumeration.

### Canary Vendor Set

The hand-audit at rollout step 10 (browser-side, immediately after the UI
migration at step 9) must cover every observed failure mode and the
realistic mix of pool compositions. Use this exact set:

1. **Pipedrive** — clean all-v4 baseline (post-Step-B winner). Confirms the
   contract doesn't reject legitimate claims when the data is good.
2. **Salesforce** — high-volume all-v4. Stress-tests cardinality of the
   validator and the select_best_claim ranking when many valid claims
   compete.
3. **ClickUp** — current STALE_PACK pattern (mixed v3/v4 with v3 dominance
   for legitimate review-level signals). Confirms cannot_validate is
   distinguished from invalid for v3-backed witnesses.
4. **Mailchimp** — second STALE_PACK pattern, different vendor category.
   Cross-checks #3.
5. **Monday.com** — the original canary failure source. Must reject the
   "Before Monday, we used HubSpot" antecedent trap. Must NOT also reject
   legitimate Monday pain claims that share a review with a competitor
   mention.
6. **Notion** — high-rating positive testimonial vendor. Confirms positive
   quotes are accepted by counterevidence_about_vendor and rejected by
   pain_claim_about_vendor.
7. **Shopify** — highest review volume in the dataset (~540 v4 reviews).
   Volume stress test; confirms the per-vendor budget holds.

For each canary vendor, document:

- valid / invalid / cannot_validate counts per claim_type
- rejection reason distribution
- **browser-side comparison**: open the EvidenceDrawer for the same
  vendor with and without the `claim_type` filter; compare what the UI
  surfaces for each Phase 6 banner type. The customer-facing experience
  is the primary acceptance signal.
- compare select_best_claim output to the legacy battle-card witness pick
  (server-side report comparison, secondary signal).
- record any disagreement and classify: contract correct / contract
  too-strict / contract too-lenient / legacy was wrong

Record findings in `docs/progress/evidence_claim_canary_audit_<date>.md`.
Do NOT proceed to Tier 2 (server-side reports) until at least 5 of 7
canary vendors show "contract correct or contract correctly stricter
than legacy" *in the browser*. Anything worse means tightening or
relaxing gates first. Server-side report migration is gated on UI
acceptance, not the other way around.

## Deletion Gates

Do not remove legacy evidence paths until all are true:

1. active witness packs have high v4 phrase metadata coverage
2. claim validation has acceptable valid / invalid / cannot_validate rates
3. battle cards are claim-backed without quality regressions
4. scorecard/deep-dive evidence no longer uses generic `quotable_phrases` for
   pain claims
5. semantic audit catches the Monday/HubSpot fixture
6. propagation audit still reports zero fillable missing fields
7. consumer inventory shows no critical direct reads of raw quote arrays for
   report-safe evidence

## Open Decisions

Decisions resolved by this revision (no longer open):

- ~~Append-only vs upsert~~: resolved via the unique index on
  `(artifact_type, artifact_id, witness_id, claim_type, target_entity, secondary_target)` —
  upsert. Covers both synthesis-tied and intelligence-tied rows uniformly.
- ~~Claims directly vs `best_evidence_for_claim` helper~~: resolved via
  the `select_best_claim()` API in the Best-Evidence Selection section.
  Consumers use the helper, never raw claim rows.

Still open:

1. Should claim rows be persisted for invalid/cannot_validate claims
   forever, or retained for a shorter audit window? Lean: keep
   indefinitely during shadow + canary; revisit after canary results.
2. Should claim validation run only on selected witness packs, or also on
   all candidate spans? Lean: selected packs only in v1 (cardinality
   reason); revisit if canary surfaces high `cannot_validate` rate
   suggesting we're dropping good candidates pre-validation.
3. What feature flags should gate the migration? The first migrating
   consumer is the API + UI (Tier 1), not battle cards. Suggested two-flag
   structure:
     - `ATLAS_B2B_EVIDENCE_API_USE_CLAIMS=true` (Tier 1 — API exposes
       `claim_type` filter and `claim_validations` field; UI consumes them)
     - `ATLAS_B2B_BATTLE_CARDS_USE_CLAIMS=true` (Tier 2 — server-side
       battle-card rendering switches to `select_best_claim()`)
   Both default off. Per-vendor allowlist overrides during canary. Adding
   a flag per migrating consumer is fine; one flag covering everything
   would force lockstep migration.
4. Does the audit need a per-tenant breakdown for B2B SaaS multi-tenant
   reports (P5/P6 tiers), or is per-vendor sufficient? Defer to first
   tenant-tier report migration.

## Current Recommendation

Proceed in this order:

1. Tier 1 prompt example update
2. Contract fixtures
3. `evidence_claim.py` validator + `evidence_claim_repository.py` (split
   per the Best-Evidence Selection API section)
4. Shadow table migration
5. Synthesis hook (shadow mode capture)
6. Semantic audit
7. API migration (`b2b_evidence.py` — claim_type filter and
   claim_validations field, behind feature flag)
8. UI migration (`atlas-churn-ui` — client.ts types, EvidenceDrawer
   banners, reportViewModels)
9. Browser canary on the named vendor set
10. Battle-card migration (Tier 2; gated on browser-canary acceptance)

Do not start broad consumer migration until the claim enum and validation
status semantics are fixed in tests, and do not start Tier 2 (server-side
reports) until the browser canary at step 9 passes the 5-of-7 threshold.
