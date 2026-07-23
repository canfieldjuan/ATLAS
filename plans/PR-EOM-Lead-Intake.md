# PR-EOM-Lead-Intake

## Why this slice exists

Issue #2151 Phase 1 (operator-requested 2026-07-22). The EOM website estimate
form's only backend is a third-party email relay (Web3Forms); Atlas never sees
the lead at submit time. Verified current state (citations re-confirmed
against code 2026-07-22):

- Website `script.js:106` posts to a dead `/api/atlas-notify` (endpoint exists
  nowhere) carrying only `{type, form_type, timestamp}` — no lead fields.
- The only lead→CRM path is `gmail_digest._process_lead_emails`: default-off
  (`autonomous.enabled=False`, `config.py:2045`), once daily (cron
  `5 7 * * *`, `scheduler.py:356`), drops service/frequency/square_feet
  (`_parse_form_fields`, `gmail_digest.py:290-301`), truncates message to 200
  chars (`:360`), stamps no `business_context_id` (`:343-350`).
- No customer acknowledgement is possible: every EOM email template requires
  price + service date (`send_estimate`, `mcp/email_server.py:125-133`); no
  price-free request-acknowledgement template exists.

### Contract revision (2026-07-23, post-Codex review)

Review evidence proved two provider behaviors the original contract wrongly
declared out of scope, plus three endpoint gaps:

- `DatabaseCRMProvider.log_interaction` pops `_inserted` before returning, so
  the endpoint's duplicate-email gate read a field production never returns.
  Revised surface: expose a public `inserted` key (additive; callers
  unaffected).
- `create_contact` dedupes by phone/email globally and merges incoming fields
  (including `business_context_id`, `contact_type`, `tags`) into whatever
  contact matches — an EOM web lead sharing an email with a non-EOM contact
  would mutate that foreign-tenant record. Revised surface: when the caller
  stamps `business_context_id`, scope both dedupe searches to that tenant
  (unstamped callers keep legacy global dedupe).
- Endpoint additions: pre-side-effect daily submission throttle (429),
  dialable-digit phone validation, and a mounted-route smoke test.

`crm_provider` is therefore no longer blanket non-scope; only these two
surgical additions are in scope, nothing else in the provider moved.

### Problem-derived contract

- Root cause: Atlas has **no ingress for website lead submissions** — the
  funnel's write path terminates at a third-party email relay, so the CRM
  write and the instant customer acknowledgement are structurally unreachable
  at submit time regardless of scheduler/config tuning.
- Correct fix must touch/change: a new public intake endpoint under
  `atlas_brain/api/` (mounted via `atlas_brain/api/__init__.py` under `/api/v1`); a
  price-free/date-free acknowledgement template under
  `atlas_brain/templates/email/` (+ `__init__` exports); CORS origins in
  `main.py` so the browser can call the endpoint from
  effinghamofficemaids.com; the stale contact data in adjacent email content
  (3 skills docs' phone, 1 template address).
- Must not change: `gmail_digest`/`email_classifier` (remain as redundant
  backfill), `email_provider` transport, any B2B
  endpoint, invoicing/receivables (#2133), schema/migrations, other writers'
  tenant stamping (Phase 2), customer backfill (Phase 3), the website repo
  (named follow-up PR).

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. New public endpoint `POST /api/v1/leads/intake` (`atlas_brain/api/leads.py`)
   accepting the full estimate-form payload (name/email/phone/service/
   frequency/square_feet/message + honeypot `website` + `source_page`):
   upserts the lead via `find_or_create_contact` with `contact_type="lead"`,
   `source="web"`, `source_ref="website_estimate_form"`,
   `business_context_id="effingham_maids"`,
   `tags=["website","estimate_request"]`; logs an **untruncated** `web_form`
   interaction; best-effort sends the acknowledgement email (reply-to
   `info@effinghamofficemaids.com`) when an email is present and the
   interaction isn't a same-day duplicate (`log_interaction` `_inserted`
   flag); email failure never fails the request.
2. New template `atlas_brain/templates/email/request_acknowledgement.py` —
   price-free, date-free, guardrail-compliant — exported via
   `atlas_brain/templates/email/__init__.py`.
3. `main.py`: add `https://effinghamofficemaids.com` +
   `https://www.effinghamofficemaids.com` to `_cors_origins`.
4. Data hygiene shipped with the email content: phone `(217) 821-2370` →
   `(217) 207-3097` in `atlas_brain/skills/email/cleaning_confirmation.md`,
   `atlas_brain/skills/email/estimate_confirmation.md`, and
   `atlas_brain/skills/email/proposal.md`; address `503 S. 5th Street` →
   `1901 S. 4th St. Ste #1, Effingham IL 62401` in
   `atlas_brain/templates/email/estimate_confirmation.py` (matches
   `atlas_brain/templates/email/invoice.py` + the live website).
5. Post-review hardening (Codex reconciliation): tenant-scoped dedupe in
   `create_contact` when `business_context_id` is stamped; public `inserted`
   flag on `log_interaction` returns; pre-side-effect daily submission
   throttle (HTTP 429, cap 5/identity/day); phone must carry >=7 digits to
   count as a contact channel; route-level smoke test for the mounted path.
6. Proof: `tests/test_leads_intake.py` (repo unit style, mocked CRM/email) —
   18 tests including provider-level dedupe-scoping regressions and the
   mounted-route smoke (200/422/429).

### Review Contract

- Acceptance criteria:
  1. Valid payload → contact upserted with kwargs
     `business_context_id="effingham_maids"`, `contact_type="lead"`,
     `source="web"` (test-asserted).
  2. Logged interaction summary contains service, frequency, square_feet, and
     the FULL message (test uses >200-char message; asserts no truncation).
  3. Honeypot-filled submission returns success, touches neither CRM nor
     email.
  4. Same-day duplicate (`_inserted=False`) does not re-send the
     acknowledgement.
  5. Email-provider exception still returns success with `email_sent=false`.
  6. Submission with neither email nor phone → 422.
  7. Template contains `(217) 207-3097`, no `$`, no price/quote language;
     send call wires `reply_to=info@effinghamofficemaids.com`.
  8. Only additive `include_router` in `atlas_brain/api/__init__.py`; no
     existing router moved.
  9. Daily cap blocks BEFORE any side effect (test-asserted: no CRM call, no
     email on 429 path).
  10. `create_contact` dedupe searches carry `business_context_id` when the
      caller stamps one, and stay unscoped when not (both test-asserted).
  11. Mounted `POST /api/v1/leads/intake` returns 200/422/429 via TestClient.
  12. Public response carries no CRM identifiers (contacts.py exposes
      unauthenticated per-id reads; returning the UUID would map
      email/phone -> id).
  13. An existing EOM contact matched by email/phone is used as-is — never
      merged, re-typed to lead, or identity-rewritten from public input.
  14. Throttle identity is digit-normalized (formatting variants of one
      phone share a cap bucket); payload caps fit the contacts schema
      (email <=254, phone <=32).
  15. Global hourly acknowledgement ceiling: past the cap the lead is still
      captured but no email is sent.
  16. Throttle bucket uses last-10-digit phone semantics matching
      search_contacts' lookup; resolution is phone-first (more unique
      channel).
  17. Scoped dedupe never matches a foreign-tenant contact but DOES match
      and claim NULL-context historical contacts (SMS/call linkers keep
      resolving existing customers pre-backfill).
  18. A failing ack-volume guard skips the email (fail-closed for sends)
      and never fails the captured lead.
- Reachability proof: entrypoint `POST /api/v1/leads/intake` on the
  atlas_brain app (port 8012), already publicly proxied by Tailscale Funnel
  (`https://atlas-brain.tailc7bd29.ts.net/api` → `127.0.0.1:8012/api`;
  `tailscale serve status` verified 2026-07-22). Observable effects:
  `contacts` + `contact_interactions` rows, acknowledgement email. Website JS
  pointing at it is the named follow-up in the site repo.
- Affected surfaces: `atlas_brain/api/__init__.py` router list;
  `atlas_brain/main.py` CORS origin list; `atlas_brain/templates/email/__init__.py`
  exports; skills email docs (contact line only).
- Risk areas: public unauthenticated POST (spam) — honeypot + length caps +
  same-day dedupe, same trust model as the existing public
  `b2b/briefings/gate`; CORS additions are origin-scoped; email misconfig
  degrades to CRM-only capture (capture > acknowledgement).
- Reviewer rules triggered: R1 (requirements match #2151 Phase 1), R2 (test
  evidence: 11 unit tests), R3 (security review of the new public endpoint: honeypot,
  input validation, origin-scoped CORS, daily throttle), R5 (backward compatibility: additive
  router only; no existing API surface changed), R6 (error handling: email best-effort,
  503 path), R8 (idempotency: same-day dedupe via interaction dedupe key),
  R11 (config: CORS origin list), R12 (deployment: endpoint live on next
  restart, no migration), R14 (verify against codebase).

### Files touched

- `atlas_brain/api/__init__.py`
- `atlas_brain/api/leads.py`
- `atlas_brain/main.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/skills/email/cleaning_confirmation.md`
- `atlas_brain/skills/email/estimate_confirmation.md`
- `atlas_brain/skills/email/proposal.md`
- `atlas_brain/templates/email/__init__.py`
- `atlas_brain/templates/email/estimate_confirmation.py`
- `atlas_brain/templates/email/request_acknowledgement.py`
- `plans/PR-EOM-Lead-Intake.md`
- `tests/test_leads_intake.py`

## Mechanism

`leads.py` keeps logic in an injectable async core
`_process_lead_intake(payload, crm, email_provider)` with a thin FastAPI route
supplying real providers — matching the repo's unit-test style (no HTTP
client). The CRM write uses `find_or_create_contact`'s `**extra` passthrough,
proven by `comms/webhooks.py:1545` (business_context_id),
`gmail_digest.py:343` (tags), `b2b_vendor_briefing.py:432`
(source/source_ref). Duplicate suppression rides the existing
`contact_interactions` dedupe key: `log_interaction` returns
`_inserted=False` for a same-day identical submission → skip the repeat
email. Sending goes through `get_email_provider().send(...,
reply_to=BUSINESS_EMAIL)` exactly like `mcp/email_server.py:169-174`.

## Intentional

- **No auth on the endpoint** — public form target, same trust model as the
  existing public `b2b/briefings/gate`; a browser form cannot hold a secret.
  Guards are honeypot + length caps + dedupe, not tokens.
- **Email awaited inline, not BackgroundTasks** — caller is fire-and-forget
  JS; 1–2s latency invisible; inline lets the response report `email_sent`
  truthfully.
- **`from_email` not forced** — Gmail sends from the authenticated account;
  forcing `info@...` risks send-as-alias rejection. Reply-To is
  `info@effinghamofficemaids.com` (send_estimate's proven pattern);
  From-address alignment is deploy env `ATLAS_EMAIL_DEFAULT_FROM` (Deferred).
- **Web3Forms untouched** — remains the operator-notification channel in
  parallel; cutover is a later operator decision.
- **No new config flags** — endpoint always mounted; with
  `EmailConfig.enabled=False` (default) the send degrades gracefully and the
  CRM write still lands.

## Deferred

- Website-repo follow-up PR pointing the form JS at this endpoint (same arc).
- Per-IP/email rate-limit table beyond honeypot+dedupe.
- Cap atomicity (waived on review): the daily cap is a read-then-act check,
  so a concurrent burst can briefly exceed it. Same pattern as the existing
  public briefing gate; the cap still bounds sustained abuse, and the new
  global hourly ceiling bounds the email blast radius. A DB-side atomic
  guard is parked hardening.
- Operator notification from Atlas (Web3Forms already emails the operator;
  duplicate channel deferred until Web3Forms cutover).
- Phases 2–3 of issue #2151 (tenant stamping/read scoping; customer backfill).
- Deploy env note: `ATLAS_EMAIL_DEFAULT_FROM=info@effinghamofficemaids.com`.

Parked hardening: per-IP/email rate-limit table; DB-side atomic cap guard (both above).

## Verification

- Pending before push: `pytest tests/test_leads_intake.py -v`;
  `pytest tests/test_b2b_vendor_briefing_quote_gate.py -v` (adjacent suite);
  `python -m py_compile` on touched Python files. Results recorded here after
  runs.
- Manual against the live 8012 process: NOT run (production; deploy follows
  merge). Post-deploy smoke: POST a test payload via the Funnel URL, verify
  `contacts` row + acknowledgement email.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/__init__.py` | 2 |
| `atlas_brain/api/leads.py` | 291 |
| `atlas_brain/main.py` | 8 |
| `atlas_brain/services/crm_provider.py` | 28 |
| `atlas_brain/skills/email/cleaning_confirmation.md` | 2 |
| `atlas_brain/skills/email/estimate_confirmation.md` | 2 |
| `atlas_brain/skills/email/proposal.md` | 2 |
| `atlas_brain/templates/email/__init__.py` | 5 |
| `atlas_brain/templates/email/estimate_confirmation.py` | 2 |
| `atlas_brain/templates/email/request_acknowledgement.py` | 66 |
| `plans/PR-EOM-Lead-Intake.md` | 249 |
| `tests/test_leads_intake.py` | 481 |
| **Total** | **1138** |
