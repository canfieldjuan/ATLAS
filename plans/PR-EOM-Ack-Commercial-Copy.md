# PR-EOM-Ack-Commercial-Copy

## Why this slice exists

A1 (#2328, merged `8d67520f4`) classified every submission into an
acknowledgement variant and recorded it on both evidence records, but
deliberately changed **no** email — every variant still rendered the one
residential template. So the defect that motivated #2320 is still live in
production today: a `commercial` lead receives copy promising that Mayra and
Tina will do a walkthrough in "less than 20 minutes", give a price before they
leave, and that the lead can "schedule your first cleaning right then". Roughly
true for a house; wrong for a facility, and wrong in a way the recipient can
see. It is the first thing EOM says to a commercial prospect.

This slice is the copy change A1 built the plumbing for: `commercial` and
`multi-location-commercial` render their own templates.

Tracking issue: #2320 (part of #2188). Predecessor: #2328.

### Problem-derived contract

- Root cause: `format_request_acknowledgement` selected no template. There was
  one `ACK_TEMPLATE`, and the only per-submission variation was an echoed
  "Your request: …" line. A1 added the variant but did not consume it.
- Correct fix must touch/change: template selection inside
  `format_request_acknowledgement`, driven by the variant A1 already derives;
  two new template bodies carrying the commercial workflow (facility
  walk-through scheduled by Juan, written estimate after the walk-through, and
  for multi-site a discovery call before any numbers).
- Must not change: the residential email, which is frozen by operator decision
  (2026-08-09, "leave it as is") and must stay byte-identical; the `general`
  fallback, which covers the form's "Other" option and anything unrecognised
  and is therefore not known to be commercial; `ACK_SUBJECT`; `template_type`;
  the recorded `ack_variant`; interaction dedupe; the per-identity daily cap
  (`MAX_DAILY_SUBMISSIONS`); the global hourly cap (`GLOBAL_ACK_HOURLY_CAP`);
  honeypot handling; Resend routing and sender identity; and failure isolation
  — a template or send failure must never fail the already-committed capture.

## Scope (this PR)

Ownership lane: eom-crm/lead-ack-variant
Slice phase: Vertical slice

1. Add `COMMERCIAL_SINGLE_SITE_TEMPLATE` and `COMMERCIAL_MULTI_SITE_TEMPLATE`,
   both operator-authored.
2. Select the template inside `format_request_acknowledgement` from the variant
   it derives itself via `classify_ack_variant`, so the email sent and the
   `ack_variant` recorded cannot disagree.
3. Speak the submitted cadence in the commercial voice, dropping cadence values
   that cannot be spoken in a sentence.
4. Leave residential and `general` byte-identical.
5. Close the A1 CI-enrollment gap for this surface (see Intentional).

### Review Contract

- Acceptance criteria:
  1. `commercial` renders the single-site template and
     `multi-location-commercial` renders the multi-site template — settled by
     `tests/test_ack_commercial_templates.py::test_single_site_commercial_renders_the_single_site_template`
     and `::test_multi_site_commercial_renders_the_multi_site_template`.
  2. Neither commercial email carries the residential promises that were wrong
     for a business (`Mayra Canfield and Tina Gomez`, `less than 20 minutes`,
     `schedule your first cleaning right then`) — settled by
     `tests/test_ack_variant_classification.py::test_commercial_services_render_their_own_template`.
  3. The rendered variant always equals the recorded variant — settled by
     `::test_rendering_agrees_with_the_recorded_variant`. Guaranteed
     structurally, not by convention: `format_request_acknowledgement` calls
     `classify_ack_variant(service)` itself rather than accepting a
     caller-supplied variant that could drift from the one intake records.
  4. Every website frequency option renders well-formed English, including the
     `custom` option and a blank — settled by
     `::test_single_site_speaks_every_spoken_frequency`,
     `::test_multi_site_speaks_every_spoken_frequency`,
     `::test_unspoken_frequency_falls_back_to_cadence_free_wording` and
     `::test_commercial_copy_never_double_spaces_or_dangles_the_cadence`.
  5. Residential and `general` are unchanged — settled by
     `tests/test_ack_variant_classification.py::test_residential_and_general_still_render_the_original_template`
     and the byte-identical anchor
     `::test_residential_render_is_byte_identical_to_pre_change_production`,
     plus the golden diff in Verification.
  6. Operator copy guardrails hold in all three templates: no dollar figures,
     "estimate" never "quote" — settled by
     `::test_no_template_contains_a_dollar_figure` and
     `::test_no_template_says_quote`.
  7. "Within 24 hours" promises initial contact only, never walkthrough
     completion or estimate delivery — settled by
     `::test_only_approved_sentences_promise_24_hours` (an exact whitelist; see
     Mechanism for why it is not a keyword rule) and
     `::test_estimate_delivery_is_never_inside_the_24_hour_window`.
  8. The multi-site email makes none of the promises #2320 forbids — settled by
     `::test_multi_site_makes_none_of_the_promises_it_must_not_make`.
  9. The commercial body is what intake actually sends and stores, not merely
     what the renderer returns — settled by
     `::test_intake_sends_the_commercial_body_for_commercial_services`, which
     asserts the `body` kwarg handed to the email provider and that the stored
     copy equals the sent copy.
- Reachability proof: the entrypoint is
  `atlas_brain/api/leads.py::_process_lead_intake`, the coroutine
  `POST /api/v1/leads/intake` awaits. The observable state is the `body` kwarg
  captured on the email provider `send` call and on the email-history `create`
  call, asserted per variant.
- Affected surfaces: the acknowledgement email body for `commercial` and
  `multi-location-commercial` submissions (customer-visible); the
  `atlas_brain.templates.email.request_acknowledgement` module exports; the
  EOM lead-pipeline workflow trigger set.
- Risk areas: drifting residential copy while editing the module around it;
  a cadence value that renders broken English; the rendered variant diverging
  from the recorded one; copy guardrail regressions (dollar figures, "quote",
  over-promised turnaround).
- Reviewer rules triggered: R1, R2, R5, R14. R1/R2/R5 from the path trigger
  `atlas_brain/api/**`. R14 declared deliberately: the template selection is a
  router-classifier consuming a closed variant set.

### Boundary-change enumeration

The seam is template selection inside `format_request_acknowledgement`: one
unconditional `ACK_TEMPLATE` becomes a three-way branch on the derived variant.

- Replaced-path behaviour: previously every variant rendered `ACK_TEMPLATE`.
  Now `commercial` → `COMMERCIAL_SINGLE_SITE_TEMPLATE`,
  `multi-location-commercial` → `COMMERCIAL_MULTI_SITE_TEMPLATE`, and
  everything else (`residential`, `deep`, `move`, `other`, unrecognised, empty,
  non-string) → `ACK_TEMPLATE`, unchanged.
- Guard-relevant fields: `service` (selects the template, via
  `classify_ack_variant`) and `frequency` (selects the spoken cadence).
- Caller × input shape: one caller,
  `leads.py::_process_lead_intake`, passing `payload.service` and
  `payload.frequency`. Both are free-text server-side (`max_length` 120), so
  the branch is total over arbitrary strings and over the non-string class,
  inherited from A1's `classify_ack_variant`.

Closure declaration for the variant set consumed here:

1. **Is the set closed or open? — CLOSED.** The four `ACK_VARIANT_*` values are
   the complete output range of `classify_ack_variant`, which A1 proved total:
   every input, including non-strings, returns one of exactly four values.
2. **Where does membership come from? — ENUMERATED**, in
   `atlas_brain/templates/email/request_acknowledgement.py`, and pinned to
   literals by
   `tests/test_ack_variant_classification.py::test_exported_variant_constants_equal_the_contracted_literals`
   and `::test_implementation_mapping_equals_the_contract_oracle`.
3. **Out-of-set behaviour and its safety rationale — fall through to
   `ACK_TEMPLATE`.** The branch uses `if / elif / else` with `else` covering
   everything non-commercial, so a fifth variant added later cannot reach a
   commercial template by accident. It renders the copy those leads receive in
   production today, which is the safe direction: the failure mode of the
   `else` is "unchanged behaviour", never "a business-specific promise sent to
   someone whose request was not identified as commercial".

### Deployed-config probing

N/A - this slice reads no environment variable, feature flag or deployed
config. Template selection is a pure function of the submitted `service` and
`frequency`. The one pre-existing gate on this path, `settings.email.enabled`,
is untouched and still decides only whether any acknowledgement is sent.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/templates/email/request_acknowledgement.py`
- `plans/PR-EOM-Ack-Commercial-Copy.md`
- `tests/test_ack_commercial_templates.py`
- `tests/test_ack_variant_classification.py`

## Mechanism

`format_request_acknowledgement` derives the variant itself —
`classify_ack_variant(service)` — rather than taking one from its caller. Intake
computes `ack_variant` separately for the evidence records, so accepting a
caller-supplied variant would create two independent derivations that could
drift; deriving it here means the email sent and the variant recorded are the
same computation on the same input.

The cadence is echoed differently per variant. Residential keeps its raw
`"Your request: <service>, <frequency>."` line untouched. The commercial
variants speak it in a sentence — "You requested a weekly commercial cleaning."
— which requires a value that reads naturally after "a". `custom` is a real
website option but a placeholder for "we'll work it out on the call", and "a
custom commercial cleaning" does not parse, so it is dropped exactly like a
blank frequency via `_UNSPOKEN_FREQUENCIES`.

The "within 24 hours" guard is an **exact whitelist of approved sentences**,
not a keyword rule, and that is deliberate. The obvious keyword rule — no line
containing "24 hours" may also contain "estimate" — cannot distinguish
"requesting a free estimate … within 24 hours" (correct: it names what was
requested) from "we'll send the estimate within 24 hours" (a promise EOM cannot
keep). A guard that flags correct copy gets relaxed until it means nothing. The
whitelist instead fails on **any** edit to a 24-hour sentence and asks a human
to re-approve it, which is the review this actually needs.

## Intentional

- **The residential email is frozen.** Operator decision 2026-08-09: friendly
  echo-line labels are *not* extended to residential — "leave it as is". The
  byte-identical anchor from A1 therefore stops being a migration aid and
  becomes a permanent contract: any future diff to that copy is a regression.
- **`general` keeps the current copy.** `other` and unrecognised values are not
  known to be commercial, so pointing them at commercial copy would be a
  downgrade. A dedicated neutral "Other" email is copy work, not plumbing — the
  classifier already separates them.
- **`template_type` stays `"request_acknowledgement"` for every variant.** A1
  deferred "per-variant `template_type`"; this slice closes that as **won't
  do**. `ack_variant` already rides in `sent_emails.metadata` and is proven
  against real PostgreSQL, so a per-variant `template_type` would duplicate it
  while splitting the column into pre-change and post-change values and
  creating a backfill obligation for zero new information.
- **`ACK_SUBJECT` is shared across all variants** (operator-approved). One less
  moving part, and the subject is not what carries the workflow difference.
- **CI enrollment fixed for this surface.** `tests/test_ack_commercial_templates.py`
  is added to the gating `Run EOM lead pipeline checks` invocation and to both
  path-filter blocks. Separately,
  `atlas_brain/templates/email/request_acknowledgement.py` was **not** a path
  trigger at all — though both sibling template modules were — so a change to
  the acknowledgement copy alone did not fire the workflow. It is added now.
  This is the same class of gap Codex found on #2328, fixed proactively.

## Deferred

Parking predicate: deferred items are non-blocking because none of them changes
what a lead receives in this slice; each is either unreachable today or a
separate copy decision that needs operator input.

- A dedicated `general` / "Other" template. Needs operator-authored copy. Today
  those leads keep exactly the email they already get, so there is no
  regression to hold this slice for.
- UTC-vs-Central dedupe bucketing (inherited from A1). The dedupe key buckets on
  the UTC date, so 6:55pm and 7:05pm Central fall in different buckets.
  Unchanged here; changing dedupe time semantics deserves its own tests.
- A corrected service *inside the same variant* may still send a second email.
  Documented in #2320, unchanged.

Parked hardening: none.

## Verification

All counts re-run at this head.

- `python -m pytest tests/test_ack_commercial_templates.py -q` — **51 passed**
- `python -m pytest tests/test_ack_commercial_templates.py
  tests/test_ack_variant_classification.py -q` — **97 passed**
- `python -m pytest tests/test_ack_commercial_templates.py
  tests/test_ack_variant_classification.py tests/test_leads_intake.py
  tests/test_eom_sent_email_tenant_scope.py -q` — **181 passed, 1 skipped**
  (the skip is the PostgreSQL test, run separately below).
- **The PostgreSQL route test was RUN, not skipped** — a skipped test is not
  evidence. Pointing `ATLAS_MIGRATION_TEST_DATABASE_URL` at the local instance:
  **1 passed**. It builds a throwaway `atlas_eom_sent_email_<uuid>` schema and
  drops it in a `finally`; before/after snapshots taken around this run are
  identical — schema count **129 → 129**, `contacts`/`contact_interactions`/
  `sent_emails` **712/2821/12 → 712/2821/12**.
- Golden render diff against A1's captured baseline, over all eight
  (name, service, frequency) combinations: the six residential/general rows are
  **byte-identical** (per-row sha256 equal on both sides); only the two
  commercial rows differ, which is this slice's entire intended effect.
- **Negative probes — each guard was made to fail, then restored** (baseline and
  restored state both **97 passed**). A guard only shown to pass on good input
  is not evidence it bites:
  | Injected defect | Result |
  |---|---|
  | multi-site copy says "every location" | 1 failed |
  | dollar figure added to commercial copy | 2 failed |
  | "quote" used instead of "estimate" | 1 failed |
  | 24-hour promise attached to estimate delivery | 1 failed |
  | commercial copy names Mayra instead of Juan | 2 failed |
  | commercial silently falls back to `ACK_TEMPLATE` | 6 failed |
  | residential copy edited (the frozen path) | 2 failed |
- `ruff check` on the changed module and both changed test files — clean.
- `python -m py_compile` on the changed runtime module — OK.
- `git diff --check` — clean.
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml` re-parsed with
  `yaml.safe_load` after both edits — valid.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| `atlas_brain/templates/email/request_acknowledgement.py` | 130 |
| `plans/PR-EOM-Ack-Commercial-Copy.md` | 271 |
| `tests/test_ack_commercial_templates.py` | 306 |
| `tests/test_ack_variant_classification.py` | 45 |
| **Total** | **757** |
