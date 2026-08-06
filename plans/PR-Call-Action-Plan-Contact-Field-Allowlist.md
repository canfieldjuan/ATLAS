# PR-Call-Action-Plan-Contact-Field-Allowlist

## Why this slice exists

Closes the exploitable half of #2299. `POST /api/v1/comms/call-actions/{id}/approve-plan`
executes LLM-proposed actions from a call transcript. One of them,
`update_contact`, forwarded the plan's `params` dict verbatim into
`DatabaseCRMProvider.update_contact`, so the only constraint was that provider's
19-field allow-list -- which permits `business_context_id` (tenancy) and
`source` (provenance).

Untrusted transcript-derived input therefore reached a privileged write that
could re-tenant a contact or forge its provenance. The provider blocks EOM
ownership and lifecycle transitions (`crm_provider.py` EOM transition guards),
but those guards do not cover `source` at all, and do not apply to non-EOM
tenants -- a plan could re-tenant a `churnsignals` contact unimpeded.

This matters now because website#109 (Slice 0B) widens the canonical mutation
surface. Widening a boundary while an unauthenticated caller can reach a
privileged writer behind it undoes the point of the boundary, so #2299 was
recorded as a hard gate on #109. This slice removes that gate.

### Why this is the field allow-list and not the auth dependency

#2299 also recommended adding a CRM-operator auth dependency to the router.
Investigation showed that would break the operator's workflow: **all ten routes
on this router are ntfy notification action buttons.** The notification
publisher builds them at `call_actions.py`:

```
f"http, Send, {base}/api/v1/comms/call-actions/{tid}/send-{draft_type}, method=POST, clear=true; "
f"http, Discard, {base}/api/v1/comms/call-actions/{tid}/discard, method=POST, clear=true"
```

When the operator taps "Send" on a phone notification, ntfy's server issues that
POST. It carries no session cookie and no bearer token, so a session-auth
dependency would 401 every action button and silently kill the call-handling
workflow. The correct authentication design is a signed capability token bound
to (transcript, action, expiry) in the URL -- a real slice, tracked on #2299,
not a one-line dependency.

The field allow-list is the part that is both correct and safe to ship now: it
closes the privilege escalation with zero effect on the notification workflow.

### Problem-derived contract

- **Root cause:** the executor treats plan-proposed `params` as trusted, so the
  set of contact fields an LLM plan can write is defined by the provider's
  general-purpose update surface rather than by what a phone call can actually
  establish.
- **Correct fix must touch/change:** `_exec_update_contact` filters `params` to
  call-derived fields before calling the provider; dropped fields are logged
  rather than discarded silently; a payload consisting only of forbidden fields
  performs no write at all; tests probe both directions.
- **Must not change:** the ntfy action-button contract (no route signatures, no
  URLs, no auth behavior); the other executors; `DatabaseCRMProvider`; any
  route's response shape; the transcript pipeline.

## Scope (this PR)

Ownership lane: comms-call-actions-security
Slice phase: production hardening

1. Add `_PLAN_UPDATABLE_CONTACT_FIELDS` to `atlas_brain/api/comms/call_actions.py`
   and filter plan params through it in `_exec_update_contact`.
2. Log dropped fields at WARNING; return a status naming what was applied and
   what was dropped.
3. Skip the provider call entirely when nothing survives the filter.
4. Add `tests/test_call_action_plan_contact_fields.py`.

### Files touched

- `atlas_brain/api/comms/call_actions.py`
- `plans/PR-Call-Action-Plan-Contact-Field-Allowlist.md`
- `tests/test_call_action_plan_contact_fields.py`

### Review Contract

1. Call-derived fields still reach the provider unchanged --
   `tests/test_call_action_plan_contact_fields.py::test_call_derived_fields_reach_the_provider`.
2. Every field the allow-list claims to accept is actually accepted, so the
   constant cannot drift narrower than its documentation --
   `::test_every_allowed_field_is_actually_accepted`.
3. No forbidden field reaches the provider, parameterized over
   `business_context_id`, `source`, `source_ref`, `contact_type`, `status`,
   `lead_stage`, `lead_owner`, `next_follow_up_at`, `tags`, `id` --
   `::test_forbidden_field_never_reaches_the_provider`.
4. A payload of only forbidden fields performs no write **and raises
   `PlanActionSkipped`**, so `approve_plan` records it as `skipped` rather than
   counting it in `executed`, naming it in the CRM interaction summary,
   persisting the plan as executed, and listing it under "Completed" in the
   notification -- `::test_tenancy_only_payload_raises_skipped`,
   `::test_only_unknown_keys_raises_skipped_not_success`.
5. Dropped fields are logged, because silently discarding a privileged field is
   its own failure mode -- `::test_dropped_fields_are_logged_not_silent`.
6. Pre-existing skip behavior is unchanged --
   `::test_no_linked_contact_is_still_skipped`, `::test_empty_params_is_still_skipped`.

Affected surfaces: the `update_contact` plan action only. Risk areas: an
allow-list narrower than real call outcomes would silently stop applying
legitimate updates (covered by criteria 1-2).

- Reviewer rules triggered: R1, R2, R3, R5, R14.

R1 and R5 are the path triggers for `atlas_brain/api/**`. R1 (requirements
match): the change implements exactly the #2299 finding it cites and nothing
more -- no route, URL, or response shape moves. R5 (backward compatibility) is
the load-bearing one here: the ten routes on this router are ntfy action-button
targets, so the compatibility surface that must not move is the notification
contract, and criterion 6 plus the untouched route signatures are the evidence.
R3 (authz/privileged paths) applies because this is the privileged write
reachable from an unauthenticated route; R14 (guard-class) because the change is
an admission boundary, probed on both sides per criteria 1-2 versus 3-4; R2
because the failure-branch fixtures are the evidence.

### Boundary-change enumeration

- Boundary path/seam: plan-action admission for contact updates, in
  `_exec_update_contact`.
- Replaced-path behaviors: a plan proposing tenancy, provenance, or lifecycle
  fields previously had them forwarded to the provider; now they are dropped
  and logged. Plans proposing only call-derived fields behave identically.
- Guard-relevant fields: the ten members of `_PLAN_UPDATABLE_CONTACT_FIELDS`,
  and by exclusion every other contact column.
- Caller x input shape: `approve-plan` x {only-allowed, only-forbidden, mixed,
  empty, no-linked-contact}.

**Reachability proof:** the executor is reached from
`POST /api/v1/comms/call-actions/{transcript_id}/approve-plan` via
`_execute_plan_action`'s `update_contact` branch. The tests drive
`_exec_update_contact` directly with a recording provider and assert on what the
provider actually received, so the observable state is the provider call itself
rather than a return string.

**Guard-class closure declaration**

- **Member set:** `_PLAN_UPDATABLE_CONTACT_FIELDS` (ten call-derived fields).
- **Key space:** **OPEN.** `params` is producer-supplied JSON from an LLM plan,
  so the set of keys that can arrive is unbounded. Enumerating known-bad names
  therefore proves nothing about the space, which is why criterion 3 alone was
  insufficient evidence.
- **Membership:** **ENUMERATED**, not derived. The set is a literal frozenset
  read by a single `in` test, so it cannot drift with schema changes.
- **Out-of-set default:** **REJECT**, and the rejection is loud. Unknown keys
  are dropped, logged at WARNING, and -- when nothing survives -- the executor
  raises `PlanActionSkipped` so the action is not audited as executed. Safety
  rationale: a new `contacts` column must not become writable from a transcript
  merely by existing, and a refused write must not be indistinguishable from a
  performed one in the audit trail.
- **Property evidence:** `::test_arbitrary_unknown_keys_never_reach_the_provider`
  generates 120 keys spanning snake_case, dotted paths, dunders, whitespace
  padding, unicode, SQL-ish names, and near-misses of real members
  (`email_<hash>`, `business_context_id_<hash>`), asserting only enumerated
  members reach the provider. `::test_mixed_payload_keeps_only_allowlist_members`
  repeats it with every allow-list member present.
- **Both sides:** every member proven accepted (criterion 2); arbitrary
  non-members proven rejected (property tests); rejected-only payloads proven to
  raise rather than write (criterion 4); both pre-existing skip branches pinned.

### Producer vocabulary

`call_extraction.md` emits `customer_name`, `customer_phone`, `customer_email`,
and `address`, and `action_planning.md` gives `update_contact` **no parameter
schema**. A plan naming the extracted fields is therefore the likely shape, not
an edge case. Without `_PLAN_FIELD_ALIASES` the allow-list would have silently
rejected every legitimate update -- the false-negative side of the same guard,
and worse than the hole it closes because it fails quietly during normal use.

Aliases map producer names onto canonical contact fields only. They cannot
introduce a new writable field: the canonical name is still checked against the
frozenset, so `customer_source` or `customer_business_context_id` are rejected
exactly like their bare forms.

### Plan-level outcome

A plan whose only outcomes were skips is persisted as `skipped`, not
`executed`. Recording `executed` made a retry answer "Plan already executed"
while nothing had happened, and `fail_count` was computed as
`len(results) - ok_count`, billing every skip as a failure. Both are now
counted explicitly, and the notification title reads "Plan Not Executed" when
nothing ran.

## Mechanism

The executor now builds `allowed` by intersecting the plan's params with the
constant, computes `rejected` as the difference, logs the rejected keys, and
returns early without calling the provider when `allowed` is empty. The provider
receives only `allowed`.

The constant documents each exclusion with its reason inline, so the next reader
sees why `source` is absent rather than assuming it was an oversight.

## Intentional

- **Allow-list, not deny-list.** A deny-list silently admits any column added to
  `contacts` later. The set of things a phone call can establish is small and
  stable; the set of columns is neither.
- **Empty result performs no write.** Otherwise the executor would issue an
  update whose only effect is bumping `updated_at`, making a rejected tenancy
  rewrite look like a successful edit in the audit trail.
- **`tags` excluded.** It is free-form and drives downstream segmentation
  (`sync_eom_portal_customers.py` writes residential/commercial tags), so it is
  not a call outcome.
- **No auth dependency in this PR.** See "Why this is the field allow-list"
  above: it would break every ntfy action button. Tracked on #2299.

## Deferred

- **Authenticating the ten call-action routes** via signed capability tokens in
  the ntfy action URLs (#2299). This is the remaining half of that issue and
  needs a design, because naive session auth breaks the phone workflow.
- **Auditing the other executors** (`_exec_book`, `_exec_email`, `_exec_sms`,
  `_exec_callback`) for the same untrusted-params pattern. They consume
  `extracted_data` rather than `params` in most paths; `_exec_callback` does
  take `params`, but its effect is a reminder rather than a CRM write. Recorded
  on #2299 rather than widened into this slice.

Parking predicate: this slice parks *authentication design*; it closes the
privilege escalation reachable through the existing unauthenticated surface.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_call_action_plan_contact_fields.py -q
16 passed

$ python -m pytest tests/test_call_intelligence.py tests/test_call_action_plan_contact_fields.py -q
46 passed

$ python -m py_compile atlas_brain/api/comms/call_actions.py
(no output)
```

Eight unrelated collection errors exist in the suite
(`test_mcp_content_ops_*`, `test_invoicing_readonly_oauth`). Verified
pre-existing: identical with the change stashed and unstashed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/comms/call_actions.py` | 122 |
| `plans/PR-Call-Action-Plan-Contact-Field-Allowlist.md` | 224 |
| `tests/test_call_action_plan_contact_fields.py` | 294 |
| **Total** | **640** |
