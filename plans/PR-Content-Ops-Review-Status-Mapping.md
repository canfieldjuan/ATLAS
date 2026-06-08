# PR - Content-Ops Review Status Mapping

## Why this slice exists

Issue #1353 identifies the next wiring gap after the tenant claim registry:
`ReviewDecision` and generated-asset lifecycle statuses still speak different
vocabularies. The deterministic review service now returns accountable review
decisions, but the generated-asset review routes still only accept arbitrary
host status strings. Without an explicit mapping, future callers can either
invent ad hoc status values or accidentally treat a blocked verdict as a normal
review update.

This slice keeps the MCP transport out and closes only the lifecycle vocabulary
seam. The generated-assets API remains backward-compatible for existing
host-defined statuses, while a caller that has a review verdict can pass that
decision directly and get the correct lifecycle status.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Add deterministic review-decision status mapping to the existing generated-asset
review routes:

1. Add a small mapping from `ReviewDecision` values to existing lifecycle
   status strings.
2. Let single and batch generated-asset review payloads provide
   `review_decision` or `decision` instead of `status`.
3. Preserve current behavior when callers provide `status`, including custom
   host-defined statuses.
4. Prove every decision branch, decoded string input, unknown-decision
   rejection, and current custom-status behavior with focused tests.

### Review Contract

- Acceptance criteria:
  - [ ] `APPROVED` and `APPROVED_WITH_EXCEPTION` map to `approved`.
  - [ ] `REVISION_REQUIRED` maps to `rejected`.
  - [ ] `BLOCKED` and `ESCALATED` map to `queued`, never `approved`.
  - [ ] Plain decoded decision strings behave the same as enum values.
  - [ ] Unknown decisions fail with a 400 instead of silently storing them.
  - [ ] Existing explicit `status` payloads keep working unchanged.
  - [ ] No MCP transport, LLM, DB migration, or claim extraction is introduced.
- Affected surfaces: API, generated-asset lifecycle status updates, CI coverage.
- Risk areas: backcompat, decoded input robustness, silent approval, maintainability.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/api/generated_assets.py`
- `tests/test_extracted_content_asset_api.py`
- `plans/PR-Content-Ops-Review-Status-Mapping.md`

## Mechanism

The generated-assets API will keep accepting explicit `status` values exactly
as it does today. If `status` is absent, it will look for `review_decision` and
then `decision`, normalize the value by equality against `ReviewDecision`, and
write the mapped lifecycle status.

The mapping is intentionally conservative:

| Review decision | Generated-asset status |
|---|---|
| `approved` | `approved` |
| `approved_with_exception` | `approved` |
| `revision_required` | `rejected` |
| `blocked` | `queued` |
| `escalated` | `queued` |

`blocked` means the review itself is incomplete or untrustworthy, so the asset
stays in the queue rather than becoming approved or content-rejected.
`escalated` is also a queue state because it needs human handling.

## Intentional

- This does not make generated-asset review routes run the review workflow yet.
  It only gives service/tool callers a safe vocabulary bridge once they have a
  verdict.
- Explicit `status` still wins over `review_decision` to preserve existing host
  callers and operational scripts.
- The existing generated-assets status vocabulary remains host-extensible; this
  slice adds a safe path for review decisions, not a global enum lock.
- Cross-layer router callers are unaffected because the router factory
  signature is unchanged and explicit `status` payload behavior is preserved;
  focused host generated-assets and hosted-workflow tests cover those callers.

## Deferred

- `PR-Content-Ops-Quality-Gate-Coverage-Rows`: map deterministic quality-gate
  and brand-voice findings into Content-PR coverage rows.
- `PR-Content-Ops-Tenant-Binding-Bridge`: reconcile MCP/OAuth tenant binding
  with the `TenantScope` used by host services.
- `PR-Marketer-Verification-MCP`: expose verify-only marketer tools after the
  service, registry, status mapping, tenant binding, and coverage-row wiring are
  in place.
- Parked hardening: none expected unless implementation surfaces non-blocking
  generated-asset status audit gaps.

## Verification

- Focused generated-asset API pytest command -- 110 passed.
- Focused host generated-assets API pytest command -- 16 passed.
- Focused hosted-workflow API pytest command -- 3 passed.
- Extracted pipeline CI enrollment audit command -- 155 matching tests are
  enrolled.
- Extracted package guardrail commands -- validation, Atlas reasoning import
  ban, standalone audit, and ASCII policy passed.
- Local PR review command with a prepared PR body file -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/generated_assets.py` | 51 |
| `tests/test_extracted_content_asset_api.py` | 119 |
| `plans/PR-Content-Ops-Review-Status-Mapping.md` | 119 |
| **Total** | **289** |
