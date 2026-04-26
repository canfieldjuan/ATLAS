# ProductClaim contract plan — 2026-04-26

## Operating rule

UI first. Reports inherit.

The dashboard / UI is the truth layer for Churn Signals. Reports are
downstream renderings of the same validated objects. A claim that
cannot be rendered safely in the UI -- with evidence, denominator,
confidence, readiness state, and suppression -- must not appear in a
report either.

Debug rule: when a bad claim shows up anywhere, do not patch the
report first. Walk the trace from raw review -> enrichment -> metric
aggregation -> evidence/witness packet -> reasoning synthesis -> UI
card -> report section, and find the FIRST point the claim became
renderable. Fix it there.

## Decision

Introduce a single shared envelope, `ProductClaim`, that every UI card
and every report section consumes. The envelope carries the fields the
operator listed in the framing change, plus the gate fields that
distinguish "render in UI detail view" from "publish in customer-facing
report":

```
claim_id, claim_type, claim_scope, claim_text, target_entity,
supporting_count, direct_evidence_count, witness_count,
contradiction_count, denominator, sample_size,
confidence, evidence_posture,
render_allowed, report_allowed, suppression_reason,
evidence_links, contradicting_links,
as_of_date, analysis_window_days, schema_version
```

UI consumers read `render_allowed`. Report consumers read
`report_allowed`, which is strictly tighter. The two gates are derived
deterministically from the underlying counts + posture, NOT set
independently per consumer -- so a report can never accidentally
publish a claim the UI suppressed.

## Codebase validation

Current first unsafe render points (named by the operator):

- `atlas-churn-ui/src/pages/VendorDetail.tsx` -- 1768 lines. Renders
  rate cards (`decision_maker_churn_rate`, recommend ratios, active
  evaluation counts) and reasoning overlays before the data has a
  denominator or readiness check. Confirmed: zero hits for
  `render_allowed`, `evidence_posture`, `confidence` keys in the file.
- `atlas-churn-ui/src/pages/EvidenceExplorer.tsx` -- 1659 lines.
  Drawer view is grounded by witness-detail fields; list/vault/trace
  views surface raw rows without the same gating.
- `atlas-churn-ui/src/pages/Opportunities.tsx` -- 2163 lines. Target
  account rows render without a shared readiness/suppression contract.
- `atlas-churn-ui/src/pages/Challengers.tsx` -- 465 lines. Builds
  challenger proof CLIENT-SIDE from alternatives/targets data; no
  validated displacement claim source.
- `atlas_brain/api/b2b_evidence.py` -- 940 lines. Witness-detail
  endpoints are claim-safe (this is what Phase 9 EvidenceClaim
  validates). Vault/trace/list surfaces are not consistently
  claim-safe.

The contract this doc proposes is the missing intermediate layer
between these surfaces and the underlying enrichment / synthesis data.

## Claim taxonomy

Each `ProductClaim` carries a `claim_scope` discriminator. The scope
determines what `target_entity` means and which UI surface consumes it.

```
ClaimScope:
  WITNESS          target_entity = vendor_name; one row per witness x claim_type
  VENDOR           target_entity = vendor_name; vendor-level theme
  ACCOUNT          target_entity = company_name; opportunity / target account
  COMPETITOR_PAIR  target_entity = subject_vendor; secondary_target = competitor
  ALERT            target_entity = vendor_name; change-event / watchlist trigger
```

Initial `claim_type` set per scope (extensible):

```
WITNESS:
  -- backed by b2b_evidence_claims rows, one per witness x claim_type
  pain_claim_about_vendor, counterevidence_about_vendor,
  named_account_anchor, displacement_proof_to_competitor,
  displacement_proof_from_competitor, pricing_urgency_claim,
  feature_gap_claim, support_failure_claim, timing_pressure_claim,
  adoption_or_onboarding_claim, reliability_claim,
  integration_or_workflow_claim

VENDOR:
  weakness_theme       -- aggregated negative signal
  strength_theme       -- aggregated positive signal
  churn_pressure       -- weighted score with denominator
  trend_direction      -- direction + magnitude over the window
  source_distribution  -- per-source coverage / skew
  evidence_posture     -- vendor-level grounding quality

ACCOUNT (Opportunities):
  active_evaluation    -- buyer is evaluating alternatives
  displacement_intent  -- buyer signals switching off subject vendor
  expansion_intent     -- buyer signals expanding usage
  weak_signal_only     -- mention without supporting evidence

COMPETITOR_PAIR (Challenger):
  direct_displacement  -- A->B switch with explicit signal
  incumbent_weakness   -- B is weak on a category A is strong on
  buyer_preference     -- B preferred over A in head-to-head reviews
  category_overlap     -- A and B compete on >= N pain categories

ALERT (Watchlist):
  new_pressure_spike   -- vendor crossed a pressure threshold
  pattern_change       -- aggregate signal shifted direction
  new_displacement     -- new direct displacement edge appeared
  evidence_decay       -- supporting evidence aged past threshold
```

Each `(claim_scope, claim_type)` pair has its own validation gate.
WITNESS scope reuses Phase 9's `validate_claim()` directly. VENDOR /
ACCOUNT / COMPETITOR_PAIR / ALERT scopes get their own validators in
later patches.

## Envelope schema

```python
@dataclass(frozen=True)
class ProductClaim:
    claim_id: str               # stable hash of (scope, type, target, secondary, as_of)
    claim_scope: ClaimScope
    claim_type: str             # see taxonomy
    claim_text: str             # short human-readable headline
    target_entity: str          # vendor_name | company_name | vendor_name (for pair)
    secondary_target: str | None  # competitor name for pair scope; None elsewhere

    # Numerator / denominator / context
    supporting_count: int       # rows that back this claim (numerator)
    direct_evidence_count: int  # subset of supporting_count that is verbatim quote-grade
    witness_count: int          # distinct witnesses (de-duped reviewers)
    contradiction_count: int    # rows pointing the OPPOSITE direction
    denominator: int | None     # population for rate-based claims
    sample_size: int | None     # total rows considered for selection (incl. discards)

    # Quality flags
    confidence: ConfidenceLabel       # high | medium | low
    evidence_posture: EvidencePosture # usable | weak | unverified | contradictory | insufficient

    # Render gates (computed; never set independently)
    render_allowed: bool
    report_allowed: bool
    suppression_reason: str | None

    # Provenance
    evidence_links: tuple[str, ...]      # backing witness_ids / claim_ids
    contradicting_links: tuple[str, ...] # contradiction witness_ids / claim_ids

    as_of_date: date
    analysis_window_days: int
    schema_version: str = "v1"
```

`claim_id` is a deterministic hash so the same logical claim across
re-runs produces the same id, which lets the React side cache and
diff cleanly.

## Render vs report gate semantics

```
render_allowed   = not suppressed AND
                   evidence_posture in {usable, weak, contradictory}
report_allowed   = render_allowed AND
                   evidence_posture in {usable} AND
                   confidence in {high, medium}
```

Reasoning:

- `weak` posture is renderable in detail views (the user can see
  "monitor only" / "insufficient evidence" labels) but not publishable
  to a customer-facing report.
- `contradictory` posture is renderable WITH the contradiction surfaced
  next to the claim (so the operator sees the conflict) but never
  publishable as confirmed.
- `unverified` and `insufficient` are blocked from both render and
  report; they remain queryable for audit only.

Suppression reasons are enumerated, not free-form, so the audit can
roll up causes:

```
SuppressionReason:
  insufficient_supporting_count
  contradictory_evidence
  unverified_evidence
  denominator_unknown
  sample_size_below_threshold
  weak_evidence_only
  passing_mention_only
  consumer_filter_applied  # consumer-side override (e.g. headline_safe)
```

## Relationship to Phase 9 EvidenceClaim

`b2b_evidence_claims` is the WITNESS-scope substrate. Every higher-scope
ProductClaim aggregates from a set of evidence_claims:

```
VENDOR.weakness_theme[Asana, pricing] aggregates from:
  evidence_claims.where(
    vendor_name = Asana
    AND claim_type IN (pain_claim_about_vendor, pricing_urgency_claim)
    AND status = 'valid'
    AND pain_category = 'pricing'
  )
```

The numerator/denominator/posture computation happens at aggregation
time. Higher scopes do NOT re-validate witness rows; they trust the
EvidenceClaim status field and apply scope-specific rules.

This means Phase 9 is not blocked by this contract -- the witness
substrate is already correct. What's missing is the aggregation layer
that turns `b2b_evidence_claims` into the higher-scope rows the React
pages need.

## Surface mapping

```
VendorDetail.tsx        -> VENDOR scope + WITNESS-scope drilldown
                           consumes: weakness_theme, strength_theme,
                             churn_pressure, trend_direction, source_distribution,
                             evidence_posture (vendor-level), pain_claim_about_vendor
                             (witness drilldown).

EvidenceExplorer.tsx    -> WITNESS scope + VENDOR scope rollups
                           consumes: every WITNESS claim_type, plus VENDOR
                             rollups for the trace/vault summary views.

Opportunities.tsx       -> ACCOUNT scope (with WITNESS drilldown)
                           consumes: active_evaluation, displacement_intent,
                             expansion_intent, weak_signal_only;
                             drilldowns to named_account_anchor witnesses.

Challengers.tsx         -> COMPETITOR_PAIR scope
                           consumes: direct_displacement, incumbent_weakness,
                             buyer_preference, category_overlap.
                           Replaces client-side inference today.

Watchlists.tsx /        -> ALERT scope
IncidentAlerts.tsx         consumes: new_pressure_spike, pattern_change,
                             new_displacement, evidence_decay.
```

Each surface lands its own migration patch (operator's order: Vendor
Workspace -> Evidence UI -> Opportunities -> Challenger -> Alerts).

## Open decisions

1. **Persistence model.** Two options:
   - (A) One unified table `b2b_product_claims` with `claim_scope`
     discriminator. Single repository, single API surface, JSONB
     `claim_payload` per scope.
   - (B) Per-scope tables (`b2b_vendor_claims`, `b2b_account_claims`,
     etc.) with shared protocol but typed columns.
   Recommend (A) for v1: easier to evolve, single repository pattern
   matches Phase 9, scope-specific indexes can still be partial. Open
   to switching to (B) at the canary review if ranking queries need
   typed columns the JSONB can't satisfy.

2. **Witness scope storage.** Should `b2b_evidence_claims` be wrapped
   into `b2b_product_claims` (one substrate)? Or stay separate and the
   ProductClaim layer reads from it via JOIN?
   Recommend: keep `b2b_evidence_claims` separate. Wrapping risks
   destabilizing Phase 9's mid-soak. The aggregation layer joins.

3. **claim_id generation.** Hash of what?
   Recommend: `sha256(claim_scope || claim_type || target_entity ||
   secondary_target || as_of_date || analysis_window_days)`. Stable
   across re-runs of the same claim; varies on different days /
   windows / vendors.

4. **Where does aggregation run?** Inside the existing synthesis cycle
   (after `b2b_reasoning_synthesis` row is persisted), or as a separate
   nightly task?
   Recommend: separate task. Keeps the synthesis hot path lean and
   lets the aggregation re-run independently when gate logic changes.

5. **Headline-safe filter from Phase 9.** Where does the
   "overall_dissatisfaction not headline-safe" filter live?
   Recommend: as a `consumer_filter_applied` suppression reason at
   the WITNESS->VENDOR aggregation step. Battle cards / briefings then
   see `report_allowed=false` instead of needing to re-implement the
   generic-pain check.

## Patch sequence

Operator's order (tracked here for the implementation log):

1. **Patch 1 (this commit):** ProductClaim envelope + computed gate
   helper + tests pinning the shape.
2. Vendor Workspace denominator-aware: extend `VENDOR` scope claims;
   suppress unsafe rate cards in `VendorDetail.tsx`.
3. Evidence UI list/vault/trace: surface the same grounding /
   confidence / posture as the drawer.
4. Opportunities readiness/suppression: `ACCOUNT` scope claims with
   suppression on missing fields.
5. Challenger displacement claims: replace client-side inference with
   `COMPETITOR_PAIR` scope claims sourced from validated
   `displacement_proof_to/from_competitor` evidence.
6. Reports inherit: each report module reads `report_allowed=true`
   ProductClaims for its scope.

No report-side changes until step 6.

## Acceptance criteria for Patch 1

- [x] Envelope dataclass with all fields the operator named, plus
      `claim_scope` discriminator and the gate helpers.
- [x] Enums for `ClaimScope`, `EvidencePosture`, `ConfidenceLabel`,
      `SuppressionReason` (closed sets, no free-form strings).
- [x] `decide_render_gates()` pure function that derives
      `render_allowed`, `report_allowed`, `suppression_reason` from
      the envelope inputs. No I/O.
- [x] Tests pinning gate semantics for every (posture, confidence)
      combination so a future patch can't loosen the report gate by
      accident.

Out of scope for Patch 1:

- DB migration. Deferred until at least one consumer needs it
  (Patch 2 = Vendor Workspace).
- Repository / async DB layer. Deferred for the same reason.
- Aggregation logic (witness -> vendor / account / pair). Deferred.
- React-side migration. Deferred.

## Status

Patch 1 is the contract definition + gate semantics, nothing else.
The contract is the proving ground; if the gate semantics are wrong,
every downstream patch inherits the wrong behavior. Get them right
before persisting or wiring.
