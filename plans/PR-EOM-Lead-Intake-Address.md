# PR-EOM-Lead-Intake-Address

## Why this slice exists

The operator (Juan) asked to add an optional "address" field to the public lead
forms so a submitted service address is captured in the CRM, without breaking
existing behavior. The website forms + `script.js` were updated to send an
optional `address` (Effingham_Office_Maids_Website #140, merged). The Atlas
intake endpoint dropped it: `LeadIntakeRequest` had no `address` field and
`_process_lead_intake` hard-coded `address=None` in the CRM write. This slice
exposes and persists it.

### Problem-derived contract

- Root cause: The intake endpoint cannot capture a submitted address — the
  request model has no `address` field, and the ingress call passes
  `address=None` unconditionally, even though the CRM write path and the
  `contacts.address` column already support it.
- Correct fix must touch/change: Add an optional `address` field to
  `LeadIntakeRequest`; pass the submitted address (stripped, blank->None) to
  `resolve_or_create_eom_inbound_lead_and_log_interaction`; record it on the
  interaction metadata (`submitted_address`) so it survives even when an
  existing contact is resolved read-only; add tests proving the passthrough and
  the blank->None collapse.
- Must not change: No DB schema/migration (`contacts.address` already exists);
  do not overwrite an existing contact's address (preserve-existing stays); do
  not make address required; do not change the email-or-phone admission rule,
  tenant/source stamping, honeypot, throttle, or acknowledgement-email behavior;
  do not touch the website (separate merged PR), the time tracker, or unrelated
  lanes.

## Scope (this PR)

Ownership lane: eom-crm/lead-intake-address
Slice phase: Vertical slice

1. Accept an optional `address` on the intake model and forward it to the CRM
   create + interaction metadata (was hardcoded `None`).
2. Add regression tests: address forwarded to the CRM contact and recorded on
   the interaction; blank/whitespace address collapses to `None`.

### Review Contract

- Acceptance criteria:
  - `LeadIntakeRequest` accepts optional `address` (default `""`, max_length
    300) — settled by the model in `atlas_brain/api/leads.py` and
    `tests/test_leads_intake.py::test_address_forwarded_to_crm_and_recorded_on_interaction`.
  - A submitted address reaches the CRM create as `address=` and the interaction
    metadata as `submitted_address` — settled by the same test (asserts
    `find_or_create_contact.call_args.kwargs["address"]` and
    `log_interaction.call_args.kwargs["metadata"]["submitted_address"]`).
  - Blank/whitespace address collapses to `None` on create — settled by
    `tests/test_leads_intake.py::test_blank_address_collapses_to_none_on_create`.
  - Base payload (no address) still passes `address=None` — settled by the added
    assertion in `test_contact_stamped_with_eom_tenant_and_web_source`.
  - No behavior change to email-or-phone admission, honeypot, throttle, or ack
    email — settled by the unchanged existing tests still green (49 total).
- Reachability proof: `POST /api/v1/leads/intake` with `{"address": "..."}` ->
  on new-lead create, `contacts.address` is written (atomic insert in
  `atlas_brain/services/crm_provider.py`, INSERT column `address`); a post-deploy
  smoke POST persists the address on the created lead.
- Affected surfaces: `atlas_brain/api/leads.py` (`LeadIntakeRequest`,
  `_process_lead_intake`); the intake interaction-metadata contract
  (`submitted_address` added).
- Risk areas: extra-field admission (model already ignores unknowns, so the
  website-first deploy was safe); blank/whitespace handling; not overwriting an
  existing contact's address.
- Reviewer rules triggered: R1 (input admission/normalization), R2, R5, R6
  (contract/metadata shape).

### Boundary-change enumeration

The intake model is an admission boundary; this adds one optional field and a
strip-normalize on it.

- Boundary path/seam: `LeadIntakeRequest` field admission and the
  `_process_lead_intake` address normalization (`payload.address.strip() or None`).
- Replaced-path behaviors: previously `address=None` unconditionally; now the
  stripped submitted address, or `None` when empty.
- Guard-relevant fields: `address` (optional, default `""`, max_length 300).
- Caller x input shape: absent address -> `""` -> `None` (unchanged effective
  behavior); present address -> stripped string; whitespace-only -> `None`.

### Deployed-config probing

N/A - no guard/config/env fallback change. No new config key; the change is a
data passthrough of an optional field. Model has no `extra='forbid'`, so unknown
fields were already ignored (the website shipped first safely).

### Files touched

- `atlas_brain/api/leads.py`
- `plans/PR-EOM-Lead-Intake-Address.md`
- `tests/test_leads_intake.py`

## Mechanism

The website form posts an optional `address` in the JSON intake body.
`LeadIntakeRequest` now declares `address` (optional). `_process_lead_intake`
strips it and passes `address=payload.address.strip() or None` to
`resolve_or_create_eom_inbound_lead_and_log_interaction`, which already threads
`address` into the CRM create (`contacts.address`, via the atomic EOM insert)
for new leads and leaves existing contacts unchanged. The stripped value is also
added to the interaction metadata as `submitted_address`, mirroring
`submitted_email`/`submitted_phone`, so a returning contact's newly-submitted
address is recorded on the interaction even when the contact row is not
overwritten.

## Intentional

- One combined `address` string (not structured street/city/state/zip): the
  atomic EOM insert only has an `address` column, so structured fields would
  require extending that insert. The operator chose a single field.
- No DB migration: `contacts.address` already exists.
- Existing-contact address is not overwritten (preserve-existing); the
  submission is preserved on the interaction record instead.

## Deferred

- Structured address (city/state/zip) if ever needed — would extend the atomic
  EOM insert and the model.

Parked hardening: none.

## Verification

- Pending before push: `pytest tests/test_leads_intake.py` (49 passed);
  `atlas_brain/api/leads.py` compiles clean via python -m py_compile. Post-deploy:
  smoke `POST` with an address -> `success: true` and `contacts.address`
  populated on the new lead.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/leads.py` | 4 |
| `plans/PR-EOM-Lead-Intake-Address.md` | 139 |
| `tests/test_leads_intake.py` | 26 |
| **Total** | **169** |
