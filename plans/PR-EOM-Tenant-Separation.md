# PR-EOM-Tenant-Separation

## Why this slice exists

Issue #2151 Phase 2 (operator-requested continuation after Phase 1 / PR #2153
deployed). Six CRM contact writers stamp no `business_context_id`, so EOM and
B2B (churnsignals) records mix in one `contacts` table, and historical rows
sit at NULL. Verified at HEAD 44ddd9964 (2026-07-23):

- `atlas_brain/tools/scheduling.py:427` (booking) — no stamp, though
  `context.id` is in scope (`:407` stamps the appointment).
- `atlas_brain/autonomous/tasks/gmail_digest.py:343` (web3forms lead) — no
  stamp.
- `atlas_brain/autonomous/tasks/email_backfill.py:198` — no stamp.
- `scripts/import_calendar_contacts.py:492` (contact_data) — no stamp.
- **B2B leak:** `atlas_brain/api/b2b_vendor_briefing.py:432` and
  `atlas_brain/autonomous/tasks/email_intake.py:613` write
  `contact_type='lead'` with NULL context into the shared table.
- No backfill script exists for existing NULL rows.

### Problem-derived contract

- Root cause: contact writes never declare tenant ownership, so tenancy is
  unknowable at read time and unfixable retroactively without provenance
  rules; the two B2B writers additionally leak software-sales leads into the
  same NULL pool as cleaning customers.
- Correct fix must touch/change: the six writer call sites (stamp
  `effingham_maids` for EOM flows via literal or `context.id`;
  `churnsignals` for the two B2B flows); a one-time, idempotent,
  dry-run-first backfill script classifying existing NULL rows by write
  provenance (source values are writer-unique; EOM appointments are
  tenant-stamped NOT NULL by schema `012:24` and give trustworthy linkage).
- Must not change: `crm_provider` (Phase 1 already made stamped dedupe
  tenant-safe: same-tenant page, then claimable-NULL page), read-path
  filtering/defaults (deferred to its own slice — see Deferred), schema,
  the lead-intake endpoint, any B2B briefing/campaign behavior beyond the
  stamp, Phase 3 customer import.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. Stamp `business_context_id` on all seven previously NULL-context contact
   writers: booking (`context.id`), gmail_digest web-lead, email_backfill,
   calendar import (`effingham_maids`); briefing gate + campaign reply +
   the extracted-package mirror gate
   (`extracted_competitive_intelligence/api/b2b_vendor_briefing.py`) as
   `churnsignals` — stops the B2B leak at every source.
2. New `scripts/backfill_business_context.py`: dry-run-by-default,
   `--apply` to write, idempotent (only touches rows still NULL). Two
   evidence tiers: tier 1 (default) = tenant-stamped appointment linkage
   (schema-trustworthy, `012:24` NOT NULL); tier 2 (opt-in
   `--classify-by-source`, operator-attested) = writer-unique source maps
   (`contacts.source` is free text also settable via the MCP crm tool, so
   it is never trusted automatically). Everything else stays NULL and is
   reported, never guessed. Count and UPDATE statements share their WHERE
   clause verbatim with positionally-complete parameters (no skipped
   placeholders).
3. Proof: `tests/test_tenant_stamping.py` — pure classification tests,
   SQL-predicate guards (NULL-only, appointment-stamp requirement,
   disjoint source maps), and AST-verified kwargs on every stamped
   `find_or_create_contact` site (refactors that drop the stamp fail).

### Review Contract

- Acceptance criteria:
  1. Every listed writer passes `business_context_id` (AST-asserted for the
     four kwargs sites; text-asserted for the two dict sites).
  2. Booking uses the runtime `context.id`, matching its appointment stamp.
  3. B2B writers stamp `churnsignals`, ending NULL-context B2B lead writes.
  4. Backfill only ever updates rows `WHERE business_context_id IS NULL`
     (asserted on both UPDATE statements).
  5. Appointment-linkage backfill requires the appointment's own
     `business_context_id = 'effingham_maids'`.
  6. EOM and B2B source maps are disjoint (asserted).
  7. Unknown sources (`manual`, `sms`, `call`, ...) are never guessed —
     reported and left NULL.
  8. Dry run performs zero writes; `--apply` is required to mutate.
- Reachability proof: writers execute on their existing production flows
  (booking tool, gmail digest task, backfill task, calendar import script,
  public briefing gate, campaign-reply intake); the backfill is
  operator-run: `python scripts/backfill_business_context.py [--apply]`
  against the live pool (same bootstrap as `import_calendar_contacts.py`).
  Observable effect: NULL-context contact counts drop to the unclassifiable
  remainder, reported by source.
- Affected surfaces: the six writer call sites (one kwarg each), new
  script, new test file. No API/schema/read-path changes.
- Risk areas: misclassification in backfill (mitigated: writer-unique
  source values, disjoint maps, appointment stamps NOT NULL by schema,
  dry-run default, unknowns left NULL); stamped writers now dedupe
  tenant-scoped via Phase 1's provider logic — intended (they claim
  legacy NULL rows and can no longer merge into foreign tenants).
- Reviewer rules triggered: R1 (matches #2151 Phase 2), R2 (tests named
  above), R3 (tenant isolation — leak stopped at source), R4 (data safety:
  NULL-only updates, dry-run default, no guessing), R5 (backward compat: additive kwargs; unstamped callers
  unchanged), R6 (jobs: digest/backfill/intake keep their fail-open error
  handling; the stamp adds no new failure path), R8 (idempotent backfill), R10
  (maintainability: the extracted-package mirror gate is stamped
  identically to its atlas_brain twin, keeping the two in lockstep), R14.

### Files touched

- `atlas_brain/api/b2b_vendor_briefing.py`
- `atlas_brain/autonomous/tasks/email_backfill.py`
- `atlas_brain/autonomous/tasks/email_intake.py`
- `atlas_brain/autonomous/tasks/gmail_digest.py`
- `atlas_brain/tools/scheduling.py`
- `extracted_competitive_intelligence/api/b2b_vendor_briefing.py`
- `plans/PR-EOM-Tenant-Separation.md`
- `scripts/backfill_business_context.py`
- `scripts/import_calendar_contacts.py`
- `tests/test_tenant_stamping.py`

## Mechanism

Each writer gains one `business_context_id` kwarg/key — the provider
(`create_contact`) already persists it (`035:27` column) and, since #2153,
uses it for tenant-safe dedupe (same-tenant page first, then claimable
NULL-context page, never foreign). The backfill reuses the UPDATE
predicates as SELECT COUNTs for the dry run, so what it reports is exactly
what `--apply` would touch.

## Intentional

- **Read-path scoping deferred** — enforcing default `business_context_id`
  filters on EOM read surfaces (MCP crm server, contacts API) is its own
  surface-mapping exercise; bundling it here would blow the slice.
- **`churnsignals` context id is not pre-registered** — the column is a free
  VARCHAR by design (`035:27`); `atlas_comms` context registration is a
  voice/SMS concern, not a CRM constraint.
- **No test harnesses for the six async flows** — the stamps are one-line
  kwargs inside large flows (calls, digests, IMAP backfills); AST-level
  assertion pins the contract without building six heavy fixtures.
- **Backfill leaves unknowns NULL** — `manual`/`sms`/`call` sources exist in
  both worlds historically; guessing would corrupt tenancy. They surface in
  the report for operator decision (Phase 3 material).

## Deferred

- Read-path tenant filtering defaults for EOM surfaces (next slice of
  #2151 Phase 2).
- Phase 3: customer-master import (calendar ICS / appointments-driven).
- Operator backfill runbook (ordering matters — a live stamped writer can
  claim a NULL row for either tenant before backfill classifies it): merge
  -> pull runtime worktree -> run backfill `--apply` (tier 1, plus tier 2
  with attestation) BEFORE restarting the service -> restart -> re-run the
  backfill (idempotent) to sweep rows written during the window. The
  remaining-NULL report feeds Phase 3.

Parked hardening: none new (Phase 1's parked items unchanged).

## Verification

- Suite `tests/test_tenant_stamping.py` — 8 passed.
- Adjacent suites `tests/test_leads_intake.py` + `tests/test_b2b_vendor_briefing_quote_gate.py` — 65 passed combined.
- `python -m py_compile` on all seven touched Python files — clean.
- NOT run: the backfill against the live DB (operator-run post-merge;
  dry-run first by design).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/b2b_vendor_briefing.py` | 1 |
| `atlas_brain/autonomous/tasks/email_backfill.py` | 1 |
| `atlas_brain/autonomous/tasks/email_intake.py` | 1 |
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 1 |
| `atlas_brain/tools/scheduling.py` | 1 |
| `extracted_competitive_intelligence/api/b2b_vendor_briefing.py` | 1 |
| `plans/PR-EOM-Tenant-Separation.md` | 172 |
| `scripts/backfill_business_context.py` | 136 |
| `scripts/import_calendar_contacts.py` | 1 |
| `tests/test_tenant_stamping.py` | 134 |
| **Total** | **449** |
