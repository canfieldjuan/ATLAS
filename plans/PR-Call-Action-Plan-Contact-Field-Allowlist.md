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

### Why this exceeds the 400-LOC budget

The diff is roughly 955 lines, of which the plan doc and the test file are the
large majority; the executable change is confined to one module. It is over
budget because the slice is not separable without shipping something unsafe:

- The allow-list alone would have **silently broken every legitimate update**,
  because the producer emits `customer_email` and the list accepts `email`. The
  alias mapping is not a follow-up; without it the guard is a regression.
- The alias mapping alone would have **written `email = NULL` over existing CRM
  data**, because the producer emits null for un-mentioned fields. The null and
  value-shape handling is not a follow-up either.
- Filtering without the `PlanActionSkipped` outcome records a refused write as a
  completed action, so the audit trail asserts something false about a security
  control. That is not a defect worth shipping for a week.

Each of those was found by review *after* the preceding piece landed, which is
the evidence that they are one indivisible change rather than three. The tests
are the deliverable for a security guard: shipping the filter without proof that
forbidden fields are rejected and legitimate ones accepted is the exact failure
this slice exists to prevent.

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
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/test_call_action_plan_contact_fields.py`
- `tests/unit_gate_baseline.txt`

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

7. **Plan persistence reflects what happened.** A plan whose only outcomes were
   skips persists as `skipped`; a plan with any error persists as `executed` and
   stays non-retryable, because an errored action may already have taken effect
   -- `::test_approve_plan_persists_skipped_when_only_skips_occurred`,
   `::test_approve_plan_persists_executed_when_an_action_errored`.
8. **Result aggregation counts skips separately from failures.** `fail_count` is
   computed from `status == "error"` rather than `len(results) - ok_count`, so a
   skip is not billed as a failure in the operator log.
9. **CRM interaction summary names only succeeded actions.** `action_summary`
   filters on `status == "ok"`, so a skipped action does not appear as approved
   work on the contact timeline.
10. **The notification title mirrors the persisted terminal state**, not whether
    any action succeeded -- `::test_notification_title_matches_the_persisted_state`.
    Telling the operator "Not Executed" for a terminal, non-retryable plan
    invites a manual redo of a send that may already have gone out.
11. **Untrusted key names are bounded before reaching a log, the persisted
    result, or ntfy** -- `::test_control_characters_in_keys_cannot_forge_log_records`,
    `::test_rejected_key_rendering_is_length_bounded`.

Affected surfaces: the `update_contact` plan action; `approve_plan` persistence
and retry semantics; result aggregation; the CRM interaction summary; and the
plan-execution notification. Risk areas: an allow-list narrower than real call
outcomes would silently stop applying legitimate updates (criteria 1-2); a
terminal-state label that disagrees with what was persisted could prompt a
duplicate send (criterion 10).

- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R14.

R1 and R5 are the path triggers for `atlas_brain/api/**`. R1 (requirements
match): the change implements exactly the #2299 finding it cites and nothing
more -- no route, URL, or response shape moves. R5 (backward compatibility) is
the load-bearing one here: the ten routes on this router are ntfy action-button
targets, so the compatibility surface that must not move is the notification
contract, and criterion 6 plus the untouched route signatures are the evidence.
R6 (observability/reporting) and R8 (idempotency and retry) are triggered by the
plan-outcome changes: the persisted status feeds the idempotency guard, and the
notification and logs are the operator's only view of what happened. R3
(authz/privileged paths) applies because this is the privileged write reachable
from an unauthenticated route; R14 (guard-class) because the change is
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

### Value shape

Every allow-listed column is VARCHAR or TEXT. A producer-supplied dict, list,
or bool is not a value the column can hold: admitting one stores a stringified
object or raises at the driver mid-plan. Values must be strings and are bounded
per column, mirroring `migrations/035_contacts.sql`. Oversized values are
rejected rather than truncated, because truncation silently stores a corrupted
value.

### Null is not a value

`call_extraction.md` emits `null` for anything the caller did not mention, so a
plan that copies the extracted payload carries nulls for most fields. Admitting
them would write `email = NULL` over existing CRM data: a call that mentioned
only a phone number would erase the contact's email. Null and blank values are
ignored, and logged separately from rejected ones. **A call can teach us a
value; it cannot teach us that a value is absent.**

The alias mapping made this more likely rather than less, by accepting exactly
the extracted field names that carry the nulls.

### Untrusted key names in output sinks

Rejected key names are LLM-produced JSON keys that flow into a log record, the
persisted plan result, and the ntfy body. `_render_keys` strips non-printable
characters, bounds each name, and caps the count, so a transcript cannot yield
`"field\nERROR forged entry"` and forge a multiline log record.

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


## CI baseline changes in this PR

Two baselines move here. Both are recorded rather than silently swept, because
`--update-baseline` on either lane would have picked up unrelated files.

**1. `tests/unit_gate_baseline.txt` -- six entries REMOVED (the ratchet tightening).**

The unit gate failed with `regressions=0; newly-passing=6`. This PR introduced no
regressions; six nodes listed as known-failures now pass, and the gate requires a
known-failures list to shrink when that happens. All six are in
`tests/test_b2b_reviews_import.py` and are unrelated to this PR's subject -- they
were fixed on `main`; this is simply the PR that ran the gate afterwards.

Verified rather than trusted: the file was run three consecutive times locally,
`6 passed` each time, before removing the entries. Removing a baseline entry for
a *flaky* test would convert a steady red into an intermittent one, which is
worse than leaving it.

**2. `tests/maturity_sweep/baseline_atlas_brain_api.json` -- one entry raised,
`call_actions.py` 6 -> 40 (`INTERNAL_MOCK` 0 -> 10).**

This one IS caused by this PR, and it is accepted rather than fixed. The ten
findings are all this PR's tests using `monkeypatch.setattr` on first-party
targets. Disposition per target:

- `_exec_email`, `_notify_plan_executed` -- **must** be stubbed. Executing them
  in a test sends real email and fires real operator notifications. Not
  negotiable regardless of what the detector prefers.
- `get_call_transcript_repo`, `_get_transcript_or_404` -- no injection seam
  exists. `call_actions.py` contains no `Depends()` at all; the routes call
  `get_call_transcript_repo()` directly as a module-level import, so
  monkeypatching is the only way to test without a live database.
- `settings.alerts.ntfy_enabled` -- configuration, not logic.

The detector is right that this shape is worse than dependency injection. The
honest answer is that converting this module to DI is a real improvement and is
not a field-allow-list fix; doing it here would be exactly the scope creep this
arc has been trying to avoid. **Deferred: see ATLAS #2304.**

The entry was grafted rather than regenerated: a blanket `--update-baseline` on
this lane also rewrote `invoicing/auth.py`, `leads.py`, `ollama_compat.py`, and
`presence.py`, none of which this PR touches. Only the `call_actions.py` key
differs from the base revision.


## Round-N findings addressed

**Alias collision (R3/R14 BLOCKER).** `_PLAN_FIELD_ALIASES` is many-to-one, so a
plan carrying both `email` and `customer_email` resolved both to one column and
`allowed[canonical] = value` let the later JSON member win silently. The planner
is fed the existing CRM contact *and* the newly extracted call data, so the
collision is reachable exactly when a caller supplies updated details -- the case
where writing the wrong one is worst. Now resolved per column: identical values
are one statement made twice and are accepted; differing values drop the column
entirely, because nothing available here distinguishes the stale or hallucinated
value from the correct one and this writes to a live CRM row. Fail closed rather
than guess. Mutation-probed: restoring last-writer-wins fails two of the three
new tests.

**Executor outcome normalisation (R6/R13).** Four no-work returns in `_exec_email`
and `_exec_sms` were plain strings, so `approve_plan` recorded them as `ok`,
counted them in the CRM interaction summary, listed them under "Completed" in the
operator notification, and persisted the plan as terminal `executed`. They now
raise `PlanActionSkipped`. Nothing was sent on those paths, so the resulting
retryable `skipped` plan is correct.

The failed-send path is deliberately NOT a skip. `send_email` returns `bool`
(verified: `ResendEmailService.send_email(...) -> bool`), and `if not sent`
previously returned the string `"Email send failed"` -- reporting a failed send
as a completed action. It now raises `RuntimeError`, which records `error` and
keeps the plan `executed` and non-retryable. Making it a skip would mark the plan
retryable after the provider had already been called, which is how duplicate
customer email gets sent.

`_exec_sms` needs no equivalent change, checked rather than assumed:
`TelephonyProvider.send_sms(to_number, from_number, body, ...) -> SMSMessage`
returns an object and raises on failure, so failures already reach the generic
handler as `error`. (The `send_sms(..., message=...)` signatures elsewhere in
`atlas_brain/comms/real_services.py` belong to `SMSService`, a different
abstraction, and are not what this call site uses -- worth stating because the
mismatch looks like a bug until the provider type is resolved.)

**Producer-contract drift (R2/R14).** `PRODUCER_FIELDS` remains a literal, but is
no longer unbound: `test_producer_fields_match_the_extraction_prompt` reads
`atlas_brain/skills/call/call_extraction.md` and fails if a field named here is
absent there, and `test_every_producer_field_is_routed_by_the_alias_map_or_is_canonical`
asserts each one resolves through `_PLAN_FIELD_ALIASES` into
`_PLAN_UPDATABLE_CONTACT_FIELDS`. This does not unify the schemas -- that is a
real refactor across prompt, executor, and tests, and is **deferred to ATLAS
#2304**. It closes the part that actually bites: silent drift going unnoticed
while every test stays green.

## Verification

```
$ python -m pytest tests/test_call_action_plan_contact_fields.py -q
47 passed

$ python -m pytest tests/test_call_intelligence.py tests/test_call_action_plan_contact_fields.py -q
77 passed

$ python -m py_compile atlas_brain/api/comms/call_actions.py
(no output)
```

Eight unrelated collection errors exist in the suite
(`test_mcp_content_ops_*`, `test_invoicing_readonly_oauth`). Verified
pre-existing: identical with the change stashed and unstashed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/comms/call_actions.py` | 265 |
| `plans/PR-Call-Action-Plan-Contact-Field-Allowlist.md` | 419 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 4 |
| `tests/test_call_action_plan_contact_fields.py` | 664 |
| `tests/unit_gate_baseline.txt` | 6 |
| **Total** | **1358** |
