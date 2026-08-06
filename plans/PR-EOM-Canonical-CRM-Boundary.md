# PR-EOM-Canonical-CRM-Boundary

## Why this slice exists

Website issue canfieldjuan/Effingham_Office_Maids_Website#105 locks the
decision that Atlas must be the canonical mutation boundary for EOM
lead/customer lifecycle writes before the broader CRM UI work continues. Current
`origin/main` already funnels EOM public intake, estimate/first-clean booking,
lost/reopen, and office conversion through dedicated EOM service methods, and
generic `update_contact` blocks EOM type/stage ownership transitions. The
remaining immediate bypass this slice can close safely is the operator-facing
generic CRM MCP create tool: when the CRM MCP default context is EOM, a bare
`create_contact(full_name=...)` request stamps `business_context_id` from that
default and delegates to the shared provider create path.

Direct backend/provider callers are not safe to block in this PR. The shared
provider path is still used by live/scheduled EOM writers such as
`atlas_brain/autonomous/tasks/email_backfill.py`, and by live operator scripts
that need a named migration/deprecation arc before the provider can become the
single canonical enforcement point.

### Problem-derived contract

- Root cause: the generic CRM MCP create tool has no admission check after
  resolving its effective `business_context_id`, so a default-EOM or
  explicit-EOM MCP request can ask the generic manual create surface to mint an
  EOM lead/customer row outside the public lead ingress or funnel transition
  contract.
- Correct fix must touch/change: add a fail-closed check in
  `atlas_brain/mcp/crm_server.py` after the effective context is resolved and
  before `_provider().create_contact(...)` is called; return a normal
  unsuccessful MCP response with the EOM ingress/funnel guidance; keep the
  shared `DatabaseCRMProvider.create_contact` behavior compatible for existing
  backend/import callers; add focused MCP tests for default-EOM and
  explicit-EOM rejection, a non-EOM default-stamp test, and provider regression
  tests proving direct backend EOM creates still do not cross-match foreign
  tenants and still reach their existing insert path.
- Must not change: public lead intake response/email behavior; EOM inbound
  resolver SQL; estimate/first-clean booking; lost/reopen; office conversion
  handoff; NocoDB/database privilege policy; non-EOM CRM creation; existing
  provider EOM contact reuse/claim behavior; live scheduled/import EOM writers;
  customer-visible website/UI copy; tracker code; and unrelated open Atlas PRs.

## Scope (this PR)

Ownership lane: eom-crm-canonical-boundary
Slice phase: production hardening

1. Reject EOM contact creation from the generic CRM MCP `create_contact` tool
   when the effective context is the EOM business context.
2. Preserve the shared provider create path for existing backend/import jobs
   until the follow-up deprecation/migration slice routes or retires each
   legacy EOM writer.

### Review Contract

- Acceptance criteria:
  - [ ] CRM MCP `create_contact(full_name=...)` with a default EOM context
        returns the EOM ingress/funnel error and does not call the provider;
        settled by
        `tests/test_crm_read_scoping.py::test_create_contact_default_eom_guard_message`.
  - [ ] CRM MCP `create_contact(..., business_context_id='effingham_maids')`
        with no default also returns the EOM ingress/funnel error and does not
        call the provider; settled by
        `tests/test_crm_read_scoping.py::test_create_contact_explicit_eom_guard_message`.
  - [ ] A non-EOM CRM MCP default context still stamps provider creates;
        settled by
        `tests/test_crm_read_scoping.py::test_create_contact_non_eom_default_stamps`.
  - [ ] Direct provider EOM calls remain backward compatible for current
        backend/import writers: they do not cross-match a foreign tenant and
        still reach the existing insert path after an EOM miss; settled by
        `tests/test_leads_intake.py::test_create_contact_does_not_match_foreign_context_after_eom_miss`
        and
        `tests/test_crm_read_scoping.py::test_create_contact_non_merging_mode_admits_fresh_eom_backend_miss`.
  - [ ] Existing EOM same-tenant matches and claimable NULL-context matches
        still return without creating a fresh row; settled by existing
        `test_create_contact_dedupe_claims_null_context_contact`,
        `test_create_contact_dedupe_same_tenant_match_reused`,
        `test_provider_prefers_same_tenant_over_null_context`, and
        `test_create_contact_non_merging_mode_returns_same_tenant_without_writes`.
  - [ ] Non-EOM fresh provider creation still reaches the DB insert path;
        settled by
        `tests/test_crm_read_scoping.py::test_create_contact_non_eom_miss_still_inserts`.
- Reachability proof: the real `atlas_brain.mcp.crm_server.create_contact`
  tool function is invoked in unit tests with default-EOM and explicit-EOM
  effective contexts, and the returned JSON is the observable unsuccessful MCP
  response. This PR adds no new HTTP route or product UI.
- Affected surfaces: CRM MCP `create_contact`, focused provider/MCP tests, and
  this plan.
- Risk areas: EOM lifecycle bypass from generic operator tooling, default-tenant
  MCP behavior, live backend/import writer compatibility, legacy claim
  compatibility, non-EOM CRM backwards compatibility, and error-message
  surfacing.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: generic CRM MCP create admission for the EOM business
  context.
- Replaced-path behaviors: an MCP create request whose effective context is
  `effingham_maids` no longer calls the provider. Explicit non-EOM MCP creates
  and non-EOM default-stamped MCP creates keep delegating to the provider. Direct
  provider callers keep their prior EOM behavior in this PR.
- Guard-relevant fields: effective `business_context_id` only.
- Caller x input shape: CRM MCP `create_contact` with omitted context plus a
  configured default, and CRM MCP `create_contact` with explicit
  `business_context_id`.

#### Guard closure declaration

- EOM context membership: **CLOSED / DERIVED**. Membership is the single
  canonical `EOM_BUSINESS_CONTEXT_ID` value imported from
  `atlas_brain.services.eom_lead_ingress` at the MCP decision point; unlisted
  context strings are outside this EOM-only guard and continue to provider
  creation because rejecting other tenants would be a cross-tenant product
  behavior change.
- Guard input field set: **CLOSED / ENUMERATED**. The guard decision reads only
  the effective `business_context_id` produced by
  `business_context_id or _default_context()` in the CRM MCP create tool.
  `contact_type`, `source`, tags, names, phone/email, and notes do not affect
  this admission decision; for the EOM context they are rejected through the
  same single guard, and for non-EOM or absent contexts they keep the previous
  provider behavior.
- Caller/input-shape inventory: **CLOSED / DERIVED** for this PR's changed
  surface. Membership is the `create_contact` tool signature in
  `atlas_brain/mcp/crm_server.py`; no backend provider callsites are included in
  this slice. Backend/import EOM provider callers are intentionally outside the
  set and keep existing behavior until the deferred deprecation/migration arc
  routes or disables them one by one.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values:
  `MCPConfig.crm_default_business_context` code default is `None`. A
  repo-owned deployed assignment for
  `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT` could not be determined from the
  searched repository paths; the actual Render/local process environment value
  would settle deployed default-EOM reachability.
- Explicit value probe: explicit
  `business_context_id='effingham_maids'` is rejected by CRM MCP create before
  provider delegation; explicit non-EOM context still delegates.
- Absent value probe: with `crm_default_business_context=None` and no explicit
  context, CRM MCP create keeps legacy unscoped provider delegation.
- Default-session/default-context probe: when
  `crm_default_business_context='effingham_maids'` in tests, CRM MCP
  `create_contact(full_name=...)` rejects before provider delegation.
- Side-effect ordering: the MCP guard returns before `_provider().create_contact`
  and therefore before any provider insert, claim, update, enqueue, or external
  side effect can occur from the rejected generic EOM request.

### Files touched

- `atlas_brain/mcp/crm_server.py`
- `plans/PR-EOM-Canonical-CRM-Boundary.md`
- `tests/test_crm_read_scoping.py`
- `tests/test_leads_intake.py`

## Mechanism

CRM MCP `create_contact` already resolves an effective tenant by applying an
explicit `business_context_id` first and then `_default_context()`. This PR adds
one guard immediately after that effective context is built and before provider
delegation. If the effective context equals the canonical EOM business context,
the tool returns a normal unsuccessful JSON response that tells the caller to
use the EOM ingress or funnel transition service.

The shared `DatabaseCRMProvider.create_contact` path is intentionally left
compatible in this PR because it has existing backend/import EOM callers that
would need coordinated route/disable work before a provider-level rejection is
safe.

## Intentional

- This PR intentionally guards the generic CRM MCP create tool only. It does not
  block direct provider callers, existing EOM row reuse, claimable legacy rows,
  or same-row enrichment that prior EOM ingress/read-scoping slices deliberately
  preserved.
- This PR intentionally does not add a broad source/provenance/audit framework
  for every historical ingestion script. That is the larger #105 Slice 0+
  contract and needs its own entry-path inventory.

## Deferred

Parking predicate: this slice parks EOM canonical-boundary hardening only when
the concern belongs to backend/import provider callers outside CRM MCP
`create_contact`, or to the broader canonical service/provenance/audit contract
that requires a named follow-up entry-path inventory. It does not park defects
where CRM MCP `create_contact` can still create an EOM contact after this
change.

- Full #105 canonical contract: one explicit Atlas create/update service for
  CRM UI lead/customer adds and edits, stable identity contract, provenance and
  audit fields for all six known entry paths, and tests proving those entry
  paths use that service.
- Legacy script deprecation/update arc: `scripts/import_calendar_contacts.py`,
  `scripts/import_eom_customers_live.py`,
  `scripts/sync_eom_portal_customers.py`, and live scheduled EOM writers such
  as `atlas_brain/autonomous/tasks/email_backfill.py` need named dispositions
  (retire, dry-run only, or route through the canonical service) before the
  provider can safely become the canonical rejection point.

Parked hardening: none against the predicate above.

## Verification

- Passed: `python -m py_compile atlas_brain/mcp/crm_server.py tests/test_leads_intake.py tests/test_crm_read_scoping.py`.
- Passed: focused provider/MCP guard set — 10 passed:
  `python -m pytest tests/test_crm_read_scoping.py::test_create_contact_non_eom_default_stamps tests/test_crm_read_scoping.py::test_create_contact_default_eom_guard_message tests/test_crm_read_scoping.py::test_create_contact_explicit_eom_guard_message tests/test_crm_read_scoping.py::test_create_contact_explicit_context_wins tests/test_crm_read_scoping.py::test_create_contact_non_merging_mode_admits_fresh_eom_backend_miss tests/test_crm_read_scoping.py::test_create_contact_non_eom_miss_still_inserts tests/test_leads_intake.py::test_create_contact_does_not_match_foreign_context_after_eom_miss tests/test_leads_intake.py::test_create_contact_dedupe_claims_null_context_contact tests/test_leads_intake.py::test_create_contact_dedupe_same_tenant_match_reused tests/test_leads_intake.py::test_provider_prefers_same_tenant_over_null_context -q --tb=short -rfE`.
- Passed: `python -m pytest tests/test_leads_intake.py tests/test_crm_read_scoping.py -q --tb=short -rfE` — 119 passed, 1 torch/pynvml warning.
- Passed: `git diff --check`.
- Passed: `python scripts/check_deployed_config_probing.py --base origin/main`.
- Previous adjacent observation: `python -m pytest tests/test_eom_lead_ingress.py tests/test_mcp_servers.py -q --tb=short -rfE` failed 6 tests in Email/Twilio/Calendar MCP expectations (`TestEmailMCPTools.test_send_email_provider_error`, `TestIMAPEmailProvider.test_list_messages_calls_executor`, `TestIMAPEmailProvider.test_get_message_calls_executor`, `TestTwilioMCPTools.test_make_call_twilio_error`, `TestCalendarMCPTools.test_list_calendars`, `TestCalendarMCPTools.test_list_events`). CRM MCP subset passed before this review-comment patch; this PR does not touch those modules.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/crm_server.py` | 15 |
| `plans/PR-EOM-Canonical-CRM-Boundary.md` | 229 |
| `tests/test_crm_read_scoping.py` | 84 |
| `tests/test_leads_intake.py` | 12 |
| **Total** | **340** |
