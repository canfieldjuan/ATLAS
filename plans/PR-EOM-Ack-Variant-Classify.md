# PR — EOM acknowledgement variant: classify + record (slice A1)

## Why this slice exists

Every website estimate submission currently receives the **same** acknowledgement
email. `format_request_acknowledgement`
(`atlas_brain/templates/email/request_acknowledgement.py`) never branches on
`service`; it only echoes the raw value into a "Your request: …" line. Rendering
residential vs commercial with the same client name differs by exactly one line.

That already reaches real commercial leads — a `commercial` / `one-time`
submission from `/contact` on 2026-08-04 received copy promising a walkthrough
"less than 20 minutes" that ends with a price and same-visit booking. Roughly
true for a single office; wrong for a multi-site prospect, and it is the first
thing EOM says to them. EOM is also starting multi-site work in two new counties
using local labor (EOM as subcontractor rather than cleaner), which makes
multi-site a distinct workflow rather than a bigger commercial one.

Tracking issue: #2320 (part of #2188). Website companion:
`Effingham_Office_Maids_Website` #141 / PR #143.

### Diff-budget overage — why this slice is indivisible

This slice exceeds the 400 LOC soft cap. The **runtime change is 59 added
lines** (11 in `leads.py`, 38 in `request_acknowledgement.py`, 10 re-exports).
The remainder is not product surface: the mandatory `plans/PR-*.md` doc that
AGENTS.md requires for any non-Markdown diff, and the test matrix the Review
Contract commits to.

Splitting was considered and rejected on each available seam:

- **Classifier without recording** leaves a pure function no caller reaches —
  dead code with no reachability proof, which R14/R2 would (correctly) reject.
- **Recording without the classifier** has nothing to record.
- **Either without the test matrix** removes the proof that makes the change
  reviewable at all: a classifier's whole risk is the inputs it does not expect,
  so all six form values, the full non-string class, both evidence records, the
  no-send path, and the byte-identical residential golden are the minimum
  evidence, not padding.

The seam that *does* exist is copy versus classification, and this slice is
already on the classification side of it — A2 and A3 carry the two new
templates separately. So the 400-line cap is exceeded by required artifacts
around a 59-line change, not by bundled scope.

### Problem-derived contract

- Root cause: template selection does not exist. There is one `ACK_SUBJECT` and
  one `ACK_TEMPLATE`, and the only per-submission variation is echoed text.
  Nothing derives *which* email a lead should receive from *what they asked for*.
- Correct fix must touch/change: a deterministic, server-side mapping from the
  submitted `service` value to an acknowledgement variant; the intake path that
  already holds that value (`atlas_brain/api/leads.py::_process_lead_intake`);
  and the two evidence records intake already writes — the `web_form`
  interaction metadata and the sent-email history metadata. It must not read a
  browser-supplied template name.
- Must not change: the residential email, which is the majority path (6 of 9 real
  submissions to date) and must render byte-identically; interaction dedupe; the
  per-identity daily cap (`MAX_DAILY_SUBMISSIONS`); the global hourly
  acknowledgement cap (`GLOBAL_ACK_HOURLY_CAP`); honeypot handling; Resend
  routing and sender identity; and failure isolation — a template or send failure
  must never fail the already-committed lead capture.

## Scope (this PR)

Ownership lane: eom-crm/lead-ack-variant
Slice phase: Vertical slice

1. Add a deterministic `classify_ack_variant(service)` mapping every value the
   website forms can submit to one of four variants, with `general` as an
   explicit total fallback.
2. Record the derived variant on the `web_form` interaction metadata and on the
   sent-email history metadata, alongside the raw submitted value.
3. Change no rendered email. Every variant still renders the single existing
   template; template selection lands in A2/A3.

### Review Contract

- Acceptance criteria:
  1. Every value the website forms can submit maps explicitly —
     `residential`/`deep`/`move` → `residential`, `commercial` →
     `commercial_single_site`, `multi-location-commercial` →
     `commercial_multi_site`, `other` → `general` — settled by
     `tests/test_ack_variant_classification.py::test_every_submitted_service_maps_explicitly`.
     `other` is an allowlisted form option, not an unknown, so it names its
     variant rather than falling through.
  2. Classification is total: unrecognised, empty, whitespace-only and
     non-string input resolve to `general` and never raise — settled by
     `::test_unrecognised_service_falls_back_to_general` and
     `::test_classifier_is_total_for_the_whole_non_string_class`, which covers
     truthy non-strings (`1`, `True`, list, dict, object, bytes) as well as the
     falsy ones.
  3. Classification reads only the submitted value; no browser-supplied template
     name is consulted — `classify_ack_variant` takes a single `service: str`
     (`atlas_brain/templates/email/request_acknowledgement.py`) and `leads.py`
     passes `payload.service`, with no template field on `LeadIntakeRequest`.
  4. Both evidence records carry the derived variant **and** the raw submitted
     `service` — settled by
     `::test_variant_recorded_on_interaction_and_email_history` (asserts the
     `metadata` kwarg on the CRM `log_interaction` and email-history `create`
     calls, parametrized over all seven inputs).
  5. The variant is recorded even when no acknowledgement is sent — settled by
     `::test_variant_recorded_on_interaction_even_when_no_email_is_sent`
     (phone-only lead; `provider.send` not awaited).
  6. No rendered email changes: the residential render is byte-identical to
     `origin/main` @ `40bb24553` — settled by
     `::test_residential_render_is_byte_identical_to_pre_change_production` and
     by the golden diff in Verification (same sha256 both sides).
  7. Interaction dedupe is unchanged: the dedupe key reads only
     `_INTERACTION_DEDUPE_ANCHOR_KEYS` (which does not contain `ack_variant`),
     `metadata["attribution"]`, and the normalized summary —
     `atlas_brain/services/crm_provider.py` `_interaction_anchor` /
     `_interaction_attribution_identity`.
  8. Caps, honeypot, Resend routing and failure isolation unchanged — settled by
     `tests/test_leads_intake.py` passing unmodified except the exact-shape
     email-history assertion, which gains the two new metadata keys (`service`
     and `ack_variant`).
- Reachability proof: the real entrypoint is
  `atlas_brain/api/leads.py::_process_lead_intake`, the same coroutine the
  `POST /api/v1/leads/intake` route awaits. It is exercised directly in
  `tests/test_ack_variant_classification.py`; the observable state is the
  `metadata` kwarg captured on the CRM `log_interaction` call and on the
  email-history `create` call, asserted per variant.
- Affected surfaces: `POST /api/v1/leads/intake`; `contact_interactions.metadata`
  and `sent_emails.metadata` evidence records; the
  `atlas_brain.templates.email` public export list.
- Risk areas: silently altering interaction dedupe by adding a metadata key;
  drifting residential copy while refactoring around it; exceeding the
  `sent_emails.template_type VARCHAR(32)` column; import cycles from a new
  module-level import in `leads.py`.
- Reviewer rules triggered: R1, R2, R5, R14. R1/R2/R5 from the path trigger
  `atlas_brain/api/**`. R14 is added deliberately: it is an advisory prose-only
  row rather than an auto-enforced path trigger, but this diff introduces a
  classifier, which that row names ("Guard, validator, cap, classifier, gate,
  sanitizer, denylist, parser admission rule, or safety checker changes →
  R14, R2"), so it is declared rather than left to the auto-check.

### Boundary-change enumeration

This diff adds a classifier, so the enumeration applies.

- Boundary path/seam: `classify_ack_variant`
  (`atlas_brain/templates/email/request_acknowledgement.py`), called once from
  `atlas_brain/api/leads.py::_process_lead_intake` before the CRM write.
- Replaced-path behaviors: no classification existed. Previously every submitted
  `service` took one undifferentiated path and the value was used only as echoed
  text. This slice adds a derived value beside it and still routes every variant
  to the same template, so no previously reachable output is replaced.
- Guard-relevant fields: `LeadIntakeRequest.service` (`str`, `max_length=120`,
  free-form from a `<select>`). Normalization is `strip().lower()`; the mapping
  is an exact-match dict, so no prefix, substring, or regex matching is involved.
- Caller × input shape: one caller (`_process_lead_intake`) × six allowlisted
  form values, plus empty, whitespace-only, unknown, mixed-case, padded, and the
  whole non-string class (truthy and falsy) — all enumerated in
  `tests/test_ack_variant_classification.py`.

#### Closure declaration — `_ACK_VARIANT_BY_SERVICE`

Per `docs/GUARD_CLASS_CLOSURE.md`, for the literal member set that controls the
routing decision:

1. **Is the set closed or open? — OPEN.** Membership is the set of `service`
   values the website `<select>` elements can submit, and those live in a
   different repository and deploy independently
   (`canfieldjuan/Effingham_Office_Maids_Website`: `contact.html`,
   `commercial-estimate.html`, `house-cleaning-estimate.html`,
   `house-cleaning-services/index.html`). A new `<option>` can ship there
   without any change here, so Atlas cannot enumerate membership
   authoritatively — the six values below are a **sample taken at a point in
   time**, not a closed universe.
2. **Where does membership come from? — ENUMERATED.** The six members
   (`residential`, `deep`, `move`, `commercial`, `multi-location-commercial`,
   `other`) are fixed text in this change, read once out of the four form files
   named above at website `origin/main`. Because they are fixed text, this code
   **cannot detect drift** when the website adds or renames an option; the
   out-of-set behaviour below is what absorbs that drift. `DERIVED` was
   rejected: the source of truth is HTML in another repo with no runtime
   contract Atlas can query, so recomputing per use is not available.
3. **What happens to an input outside the set? — it resolves to `general`, and
   `general` renders the existing template.** This covers *unlisted members of
   the class* (a new website option) as well as true non-members (garbage,
   non-strings), which is the drift case that matters. The direction is the safe
   side for two reasons: a lead whose service Atlas does not recognise still
   receives exactly the acknowledgement every lead receives today, so the
   failure mode is "no new behaviour" rather than wrong expectations; and
   routing an unknown value into commercial copy could promise a discovery call
   and a written proposal to someone who asked for a house cleaning. Intake also
   never fails on an unknown value — the classifier is total, so lead capture is
   never at risk from a form change.

### Deployed-config probing

N/A — no guard/config boundary change. `classify_ack_variant` reads no
environment variable, setting, or deployed config; its only input is the
submitted `service` value, and its fallback (`general`) is a literal in the
mapping rather than a config default. The one setting this path already consults,
`settings.email.enabled`, is untouched.

### Files touched

- `atlas_brain/api/leads.py`
- `atlas_brain/templates/email/__init__.py`
- `atlas_brain/templates/email/request_acknowledgement.py`
- `plans/PR-EOM-Ack-Variant-Classify.md`
- `tests/test_ack_variant_classification.py`
- `tests/test_eom_sent_email_tenant_scope.py`
- `tests/test_leads_intake.py`

## Mechanism

A module-level dict maps normalized (`strip().lower()`) service values to variant
constants; `classify_ack_variant` returns
`_ACK_VARIANT_BY_SERVICE.get(normalized, ACK_VARIANT_GENERAL)`, so the function is
total by construction rather than by branch coverage. `leads.py` classifies once,
before the CRM write, and passes the result into the metadata dict it already
builds plus the email-history metadata it already writes.

Deliberately **not** a column on `contacts`. A multi-location company is still a
commercial customer — "multi-site" describes the shape of the request, not a
third customer type — so this is a lead-time acknowledgement variant.
`contacts.contact_type` already means lifecycle (`customer` 410 rows / `lead` 297
rows); adding `customer_type` beside it would put two similarly-named columns on
unrelated axes. Whether residential/commercial ever becomes a permanent contact
attribute stays an open decision this slice does not force.

## Intentional

- `template_type` stays `"request_acknowledgement"`. Two reasons: it would be
  inaccurate, because this slice still renders that one template, so a
  per-variant value would misdescribe what was sent; and it would not fit —
  `sent_emails.template_type` is `VARCHAR(32)`
  (`atlas_brain/storage/migrations/016_sent_emails.sql:10`) while
  `request_acknowledgement:residential` is 35 characters. A2/A3 will use short
  per-variant values once the templates genuinely differ.
- Adding a metadata key cannot shift interaction dedupe — verified against
  `_interaction_anchor` (fixed allowlist, no `ack_variant`) and
  `_interaction_attribution_identity` (reads only `metadata["attribution"]`) in
  `atlas_brain/services/crm_provider.py`.
- The raw `service` value is kept alongside the derived variant, so a future
  mapping change can be reasoned about against what was actually submitted.
- `classify_ack_variant` is imported locally inside `_process_lead_intake`,
  matching that file's existing lazy-import style for `..templates.email`,
  `..config`, and `..storage.*`, rather than adding a module-level import.
- One existing assertion was updated rather than loosened:
  `test_successful_acknowledgement_records_tenant_history` pins the email-history
  metadata dict exactly, so the new key was added to it instead of relaxing it to
  a subset check.
- `tests/unit_gate_baseline.txt` is deliberately **left untouched**, even though
  the local pre-push unit-gate mirror reports entries as STALE and asks for the
  ratchet to shrink. That report is a known **false negative on this dev box**:
  the mirror runs in an env that is a superset of CI's lean unit-gate env, so
  `test_monthly_invoice_generation.py` `*_real_repo`/PDF tests pass here
  (postgres is live on localhost:5433; CI's unit gate has no DB). The
  `tests/security/*` entries that previously appeared stale were resolved
  separately on `main` by #2323 and are no longer in the baseline. Historically
  also affected: `test_monthly_invoice_generation.py` `*_real_repo`/PDF tests pass
  here (postgres is live on localhost:5433; CI's unit gate has no DB). Removing
  those entries would convert them into **CI regressions**. Prior CI evidence:
  FULL gate run 31233990183 reported `baseline=169; regressions=0;
  newly-passing=0`. Tooling fix tracked separately as #2324.

## Deferred

- A2 — single-site commercial template; adds the template and routes
  `commercial_single_site` to it.
- A3 — multi-site commercial template; discovery-call-first copy that must not
  promise an immediate price, that Mayra and Tina inspect every location, one
  identical checklist, or pre-confirmed serviceability. Named contact becomes
  Juan rather than Mayra and Tina.
- Per-variant `template_type` — blocked on A2/A3 (see Intentional); values must
  fit `VARCHAR(32)`.
- Duplicate-email semantics for a corrected service — a resubmission with a
  different service already yields a new dedupe key and therefore re-acknowledges.
  Intended; not redesigned here.
- UTC vs America/Chicago dedupe bucketing — `crm_provider.py` buckets on the UTC
  date, so two submissions either side of 7 p.m. Central fall in different
  buckets. Changing dedupe time semantics deserves its own tests and an atomic
  design; folding it in would turn three email templates into a rewrite of intake
  idempotency.

Parking predicate: this slice parks **copy and template-selection findings** —
anything about what an email *says* or which template it picks — because it
deliberately changes no rendered output. It does not park correctness findings
about classification, recording, dedupe, or caps.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_ack_variant_classification.py -q` — **32 passed**
- `python -m pytest tests/test_leads_intake.py tests/test_ack_variant_classification.py -q`
  — **82 passed**
- Golden render: all eight service values rendered before and after are
  **byte-identical**, sha256
  `23199591b0e5ea13e999db364005c376e1c5c70fd69cf4f66106e8eb6667f7bc` on both
  sides, captured against `origin/main` @ `40bb24553`.
- Suite failure diff against a clean `origin/main` worktree with the same `-k`
  selection (`lead or ack or intake or eom`): **16 failures on both sides, zero
  introduced**. Eight collection errors (`test_invoicing_readonly_oauth`,
  `test_mcp_content_ops_*`, and others) reproduce at baseline.
- `ruff check` on the five changed paths — clean. Four repository-wide findings
  (`invoice.py`, `vendor_briefing.py`) and one unused-import at
  `tests/test_leads_intake.py:37` are pre-existing: they reproduce at
  `origin/main` with these changes stashed, and none sit on a line this PR
  touches.
- `python -m py_compile` on the three changed runtime modules — OK.
- `git diff --check` — clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/leads.py` | 19 |
| `atlas_brain/templates/email/__init__.py` | 10 |
| `atlas_brain/templates/email/request_acknowledgement.py` | 42 |
| `plans/PR-EOM-Ack-Variant-Classify.md` | 318 |
| `tests/test_ack_variant_classification.py` | 283 |
| `tests/test_eom_sent_email_tenant_scope.py` | 6 |
| `tests/test_leads_intake.py` | 6 |
| **Total** | **684** |
