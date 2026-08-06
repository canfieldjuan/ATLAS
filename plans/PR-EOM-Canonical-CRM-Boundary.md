# PR-EOM-Canonical-CRM-Boundary

## Why this slice exists

Website issue canfieldjuan/Effingham_Office_Maids_Website#105 locks the
decision that Atlas must be the canonical mutation boundary for EOM
lead/customer lifecycle writes before the broader CRM UI work continues. Current
`origin/main` already funnels EOM public intake, estimate/first-clean booking,
lost/reopen, and office conversion through dedicated EOM service methods, and
generic `update_contact` blocks EOM type/stage ownership transitions. The
remaining small bypass is fresh generic creation: a generic CRM caller can still
ask `DatabaseCRMProvider.create_contact` to insert a new
`business_context_id='effingham_maids'` contact directly when no compatible
match exists. That mints EOM leads/customers without the explicit EOM ingress or
funnel transition contract.

### Problem-derived contract

- Root cause: the generic CRM creation path has no final admission check for a
  fresh EOM-owned contact after same-tenant and claimable-legacy dedupe miss.
  Generic updates already reject EOM ownership/type/stage transitions, and EOM
  inbound creation has its own atomic resolver, but a default-EOM MCP/create
  caller or another generic backend caller can still fall through to the raw
  `INSERT INTO contacts (...) business_context_id='effingham_maids' ...` path.
- Correct fix must touch/change: add a provider-level fail-closed guard before
  the fresh insert path in `atlas_brain/services/crm_provider.py`; keep existing
  EOM same-tenant reuse, claimable-legacy claim, and protected same-row
  enrichment semantics intact; expose the provider `ValueError` through the CRM
  MCP create tool instead of flattening it to "Internal error"; and add focused
  provider/MCP tests proving fresh EOM creation is rejected while non-EOM
  creation and existing EOM reuse still work.
- Must not change: public lead intake response/email behavior; EOM inbound
  resolver SQL; estimate/first-clean booking; lost/reopen; office conversion
  handoff; NocoDB/database privilege policy; non-EOM CRM creation; existing EOM
  contact reuse; claimable legacy contact claiming; customer-visible website/UI
  copy; tracker code; and unrelated open Atlas PRs.

## Scope (this PR)

Ownership lane: eom-crm-canonical-boundary
Slice phase: production hardening

1. Reject new EOM contact inserts from the generic `create_contact` /
   `find_or_create_contact` provider path when dedupe/claim does not find a
   compatible existing row.
2. Return that provider rejection as an actionable MCP create error and prove
   the boundary with focused provider and MCP regression tests.

### Review Contract

- Acceptance criteria:
  - [ ] A generic provider create with
        `business_context_id='effingham_maids'` and no compatible existing row
        raises a `ValueError` before opening the DB insert path; settled by
        `tests/test_leads_intake.py::test_create_contact_rejects_fresh_eom_contact_after_foreign_miss`
        and `tests/test_crm_read_scoping.py::test_create_contact_non_merging_mode_rejects_fresh_eom_default_miss`.
  - [ ] Existing EOM same-tenant matches and claimable NULL-context matches
        still return without creating a fresh row; settled by existing
        `test_create_contact_dedupe_claims_null_context_contact`,
        `test_create_contact_dedupe_same_tenant_match_reused`,
        `test_provider_prefers_same_tenant_over_null_context`, and
        `test_create_contact_non_merging_mode_returns_same_tenant_without_writes`.
  - [ ] Non-EOM fresh creation still reaches the DB insert path; settled by a
        focused provider regression in `tests/test_crm_read_scoping.py`.
  - [ ] The MCP create tool returns the guard message for the default EOM
        context instead of a generic internal error; settled by
        `tests/test_crm_read_scoping.py`.
- Reachability proof: the real `atlas_brain.mcp.crm_server.create_contact`
  tool is invoked with the default EOM context and the returned JSON exposes the
  provider guard as an observable error. The provider-level path is unit-tested
  directly because this PR adds no new HTTP route or product UI.
- Affected surfaces: `DatabaseCRMProvider.create_contact`,
  `DatabaseCRMProvider.find_or_create_contact`, CRM MCP `create_contact`, and
  focused provider/MCP tests.
- Risk areas: EOM lifecycle bypass, default-tenant MCP behavior, legacy claim
  compatibility, non-EOM CRM backwards compatibility, and error-message
  surfacing.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: generic provider/MCP fresh-contact admission for the EOM
  business context.
- Replaced-path behaviors: a dedupe miss for
  `business_context_id='effingham_maids'` no longer falls through to the raw
  `INSERT INTO contacts` path. Existing EOM matches, claimable legacy matches,
  and non-EOM fresh creates keep their prior behavior.
- Guard-relevant fields: `business_context_id`, `contact_type`, `lead_stage`,
  `merge_existing`, and the provider's compatible-match result.
- Caller x input shape: MCP/default-EOM create requests and generic backend
  provider calls carrying the EOM business context.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no config values change.
- Explicit value probe: explicit non-EOM `business_context_id` still admits a
  fresh provider insert in tests.
- Absent value probe: default MCP context configured to EOM rejects a fresh
  create and returns the provider error.
- Default-session/default-context probe: CRM MCP `create_contact(full_name=...)`
  uses `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT` as before, but that default no
  longer authorizes a fresh EOM row.
- Side-effect ordering: provider dedupe/claim resolution runs first; only after
  no compatible existing row remains does the EOM fresh-insert guard reject
  before the DB pool insert path is reached.

### Files touched

- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Canonical-CRM-Boundary.md`
- `tests/test_crm_read_scoping.py`
- `tests/test_leads_intake.py`

## Mechanism

`create_contact` already determines the requested business context, searches
same-tenant/legacy matches, claim-checks legacy rows, and either returns an
existing record or falls through to the fresh insert path. This PR adds one
guard at that fall-through seam: if the requested context is
`effingham_maids`, reject with a `ValueError` that tells callers to use the EOM
ingress or funnel transition service. `find_or_create_contact` inherits the
same protection because it delegates to `create_contact`. The MCP create tool
already passes the deployment default context into provider data; it now lets
provider `ValueError`s reach the caller as a normal unsuccessful tool response.

## Intentional

- This PR intentionally guards only fresh generic EOM creation. It does not
  block returning existing EOM rows, claimable legacy rows, or same-row
  enrichment that prior EOM ingress/read-scoping slices deliberately preserved.
- This PR intentionally does not add a broad source/provenance/audit framework
  for every historical ingestion script. That is the larger #105 Slice 0+
  contract and needs its own entry-path inventory.

## Deferred

- Full #105 canonical contract: one explicit Atlas create/update service for
  CRM UI lead/customer adds and edits, stable identity contract, provenance and
  audit fields for all six known entry paths, and tests proving those entry
  paths use that service.
- Legacy script deprecation/update arc: `scripts/import_calendar_contacts.py`,
  `scripts/import_eom_customers_live.py`, and
  `scripts/sync_eom_portal_customers.py` still need a named operator path
  (retire, dry-run only, or route through the canonical service) before they are
  safe as live mutation tools.

Parked hardening: none.

## Verification

- Passed: `python -m py_compile atlas_brain/services/crm_provider.py atlas_brain/mcp/crm_server.py tests/test_leads_intake.py tests/test_crm_read_scoping.py`.
- Passed: focused provider/MCP guard set — 9 passed:
  `python -m pytest tests/test_leads_intake.py::test_create_contact_rejects_fresh_eom_contact_after_foreign_miss tests/test_leads_intake.py::test_create_contact_dedupe_claims_null_context_contact tests/test_leads_intake.py::test_create_contact_dedupe_same_tenant_match_reused tests/test_leads_intake.py::test_provider_prefers_same_tenant_over_null_context tests/test_crm_read_scoping.py::test_create_contact_default_stamps tests/test_crm_read_scoping.py::test_create_contact_default_eom_guard_message tests/test_crm_read_scoping.py::test_create_contact_non_merging_mode_returns_same_tenant_without_writes tests/test_crm_read_scoping.py::test_create_contact_non_merging_mode_rejects_fresh_eom_default_miss tests/test_crm_read_scoping.py::test_create_contact_non_eom_miss_still_inserts -q --tb=short -rfE`.
- Passed: `python -m pytest tests/test_leads_intake.py tests/test_crm_read_scoping.py -q --tb=short -rfE` — 118 passed, 1 torch/pynvml warning.
- Passed: `python -m pytest tests/test_eom_lead_ingress.py tests/test_mcp_servers.py::TestCRMMCPTools -q --tb=short -rfE` — 28 passed.
- Passed: `python -m pytest tests/test_eom_lead_conversion.py -q --tb=short -rfE` — 173 passed, 1 torch/pynvml warning.
- Observed unrelated adjacent failures: `python -m pytest tests/test_eom_lead_ingress.py tests/test_mcp_servers.py -q --tb=short -rfE` fails 6 tests in Email/Twilio/Calendar MCP expectations (`TestEmailMCPTools.test_send_email_provider_error`, `TestIMAPEmailProvider.test_list_messages_calls_executor`, `TestIMAPEmailProvider.test_get_message_calls_executor`, `TestTwilioMCPTools.test_make_call_twilio_error`, `TestCalendarMCPTools.test_list_calendars`, `TestCalendarMCPTools.test_list_events`). CRM MCP subset passed; this PR does not touch those modules.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/crm_server.py` | 2 |
| `atlas_brain/services/crm_provider.py` | 9 |
| `plans/PR-EOM-Canonical-CRM-Boundary.md` | 176 |
| `tests/test_crm_read_scoping.py` | 73 |
| `tests/test_leads_intake.py` | 18 |
| **Total** | **278** |
