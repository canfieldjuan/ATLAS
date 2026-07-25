# PR-Lead-Ack-Resend-Sender

## Why this slice exists

Operator-requested (Juan, 2026-07-24). The EOM estimate-request acknowledgement
email (`atlas_brain/api/leads.py` -> `format_request_acknowledgement`, shipped in
PR-EOM-Lead-Intake) is reaching customers **from the Gmail account address, not
from `info@effinghamofficemaids.com`**. PR-EOM-Lead-Intake explicitly deferred
this ("`from_email` not forced ... From-address alignment is deploy env
`ATLAS_EMAIL_DEFAULT_FROM`", plan lines 223-226/245). That deferral rested on a
wrong assumption, now corrected against code:

- `leads.py:323-328` passes **no `from_email`**; the Gmail transport only sets a
  `From` header when one is supplied (`atlas_brain/tools/gmail.py:135-136`), so
  From is unset and Gmail uses the authenticated account. `ATLAS_EMAIL_DEFAULT_FROM`
  is consulted **only on the Resend/email_tool path** (`atlas_brain/tools/email.py:268`),
  never on the direct Gmail path (`atlas_brain/services/email_provider.py:515-528`).
- `CompositeEmailProvider.send` selects Gmail by **credential availability**
  (`email_provider.py:673`, `is_available()` `:486-490`), **ignoring**
  `gmail_send_enabled`; that flag is honored only in `email_tool.execute`
  (`tools/email.py:306`). So setting the env From or the flag alone does not
  move this send off Gmail.

Juan chose Resend for this transactional email (API-key auth, no Gmail OAuth
upkeep, sends from the verified brand domain). Resend the service supports HTML;
Atlas's Resend REST payload omits the `html` key (`tools/email.py:314-319`) --
noted for the deferred HTML issue, not changed here.

### Problem-derived contract

- Root cause: the acknowledgement never specifies a sender and is routed to
  Gmail-if-credentialed, so it goes out as the Gmail account. Neither the From
  env nor `gmail_send_enabled` reaches this path.
- Correct fix must touch/change: a way to route **this one send** through Resend
  from `info@` without moving any other Atlas email off Gmail -- i.e. a per-call
  provider override on `CompositeEmailProvider`, a `force_resend` signal that
  reaches the Gmail-first gate in `email_tool`, and an explicit `from_email` on
  the acknowledgement send in `leads.py`.
- Must not change: the Gmail-first default for every other caller
  (invoicing reminders, monthly invoices, `invoicing_server`, `billing.py`,
  `b2b_vendor_briefing`, `email_server`); the acknowledgement's dedupe/volume
  guards; the Resend REST/attachment logic; `gmail_send_enabled` semantics for
  the email_tool path.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. `atlas_brain/services/email_provider.py`: `CompositeEmailProvider.send` gains
   an optional named param `provider`; `provider="resend"` skips the Gmail
   attempt and calls `ResendEmailProvider.send` directly. `ResendEmailProvider.send`
   stamps `force_resend=True` into the email_tool params (it is the Resend
   provider by definition and must not fall back to Gmail-first).
2. `atlas_brain/tools/email.py`: the Gmail-first gate (`:306`) additionally
   requires `not params.get("force_resend")`, so a forced send goes straight to
   the Resend REST payload (which already applies `from_email or default_from`
   and `reply_to`).
3. `atlas_brain/api/leads.py`: the acknowledgement send passes
   `provider="resend"` and `from_email="Effingham Office Maids <info@effinghamofficemaids.com>"`
   (display-name form via the exported `BUSINESS_NAME`/`BUSINESS_EMAIL`);
   `reply_to` stays `info@`.
4. Bundled, operator-directed copy freshening (approved same session,
   2026-07-24): `atlas_brain/templates/email/request_acknowledgement.py`
   "what happens next" is Juan's 5-step version (call to book a <20-min
   walkthrough; the estimate team Mayra Canfield and Tina Gomez; show them
   around; cost given on-site with same-visit scheduling; FREE, no obligation).
   Same file family, plain-text reflow so it renders human-formatted; all
   existing copy guardrails preserved.
5. Proof: `tests/test_leads_intake.py` extended -- the acknowledgement send now
   asserts `provider="resend"` + `from_email` (info@); plus two composite-routing
   unit tests (Composite override skips Gmail even when available; default still
   prefers Gmail). The `email_tool` force_resend gate is verified by inspection,
   not a direct unit test, to avoid mocking the unit under test (real-adapter
   rule / maturity-sweep INTERNAL_MOCK ratchet).

### Review Contract

- Acceptance criteria:
  1. The acknowledgement send call carries `provider="resend"` and a `from_email`
     containing `info@effinghamofficemaids.com` (test-asserted).
  2. `CompositeEmailProvider.send(provider="resend")` calls Resend and does NOT
     call Gmail even when `self._gmail.is_available()` is True (test-asserted);
     `provider` never leaks into the underlying provider kwargs.
  3. `CompositeEmailProvider.send` with no override still prefers Gmail when
     available (default path for all other callers unchanged; test-asserted).
  4. `ResendEmailProvider.send` stamps `force_resend=True` and `email_tool.execute`
     yields the Gmail-first gate to it -- verified by **inspection** and the
     end-to-end composite route, NOT by a direct `email_tool` unit test: mocking
     `email_tool`'s own HTTP client / transport (the only way to unit-test its
     send) is mocking the unit under test, which trips the maturity-sweep
     INTERNAL_MOCK ratchet and violates the repo's real-adapter rule. The two
     force_resend one-liners are covered by the composite skip-Gmail test above
     plus code review.
  5. Existing acknowledgement guardrails hold: no `$`, no "quote"/"same-day",
     `(217) 207-3097` present, `within 24 hours` present, request-line echo,
     empty-name fallback (unchanged tests stay green).
  6. Copy: the 5-step "what happens next" renders as clean per-item paragraphs
     (plain-text reflow), ASCII-only.
- Reachability proof: entrypoint is the existing live `POST /api/v1/leads/intake`
  (atlas_brain app, Tailscale Funnel). Observable effect: the acknowledgement
  arrives From `info@effinghamofficemaids.com` via Resend (Resend dashboard
  message log). No new route, no migration.
- Affected surfaces: `email_provider.py` send routing; `email_tool` Gmail gate;
  `leads.py` acknowledgement send; the acknowledgement copy template.
- Risk areas: the shared send path is touched, but the default branch is
  unchanged (guarded by test 3); only `leads.py` opts into Resend. Requires
  `ATLAS_EMAIL_API_KEY` set in the deploy env (operator: Resend account + domain
  verified + key ready, confirmed 2026-07-24) -- without it the send fails
  best-effort and the CRM capture still lands (existing behavior).
- Reviewer rules triggered: R1 (matches operator request), R2 (test evidence:
  extended wiring test + 2 new composite-routing tests; email_tool gate by
  inspection per real-adapter rule), R5 (backward compatibility: default provider
  selection unchanged for all non-leads callers), R6 (error handling: send stays
  best-effort; email failure never fails the lead), R11 (config:
  `ATLAS_EMAIL_API_KEY` deploy env), R14 (verify against codebase).

### Files touched

- `atlas_brain/api/leads.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/templates/email/request_acknowledgement.py`
- `atlas_brain/tools/email.py`
- `plans/PR-Lead-Ack-Resend-Sender.md`
- `tests/test_leads_intake.py`

## Mechanism

The email stack prefers Gmail at two layers -- `CompositeEmailProvider.send`
(by credential availability) and `email_tool.execute` (by `gmail_send_enabled`).
A surgical Resend route threads a signal through both: `provider="resend"` on the
composite skips the first, and `force_resend` (stamped by `ResendEmailProvider`,
read at the email_tool gate) skips the second, landing on the existing Resend
REST payload which already honors `from_email`/`reply_to`. `leads.py` opts in and
passes the display-name `info@` sender. Because `provider` is a named param it is
never forwarded to `GmailEmailProvider.send` (which has no `**kwargs`), and every
other `get_email_provider().send(...)` caller keeps the Gmail-first default.

## Intentional

- **Per-call override, not a global flag flip.** Scope is the lead
  acknowledgement only; invoicing/billing/briefings/MCP email stay on Gmail
  (blast radius rejected by operator). `gmail_send_enabled` semantics are
  untouched.
- **`ResendEmailProvider` always forces Resend.** It is the Resend provider;
  routing it back through email_tool's Gmail-first preference was a latent
  oddity -- `force_resend` fixes it without changing the default composite path.
- **Display-name From** (`Effingham Office Maids <info@...>`) rather than a bare
  address, for a branded sender. Reply-To stays `info@`.
- **Copy freshening bundled** because it is the same email touched in the same
  operator session; both changes ship together as "the acknowledgement email".

## Deferred

- HTML (multipart) acknowledgement email -- logged as its own GitHub issue
  (mockup built this session). Enabler: add the `html` key to the Resend REST
  payload (`tools/email.py:314-319`) and multipart/alternative to the Gmail
  transport (`tools/gmail.py:128-131`, currently html-only when html is set),
  plus de-hardcoding the estimate-team names into config.
- A real `from_name`/display-name config field (inline formatting used for now).
- Deploy env: ensure `ATLAS_EMAIL_API_KEY` is set (operator step; Resend ready).

Parked hardening: none.

## Verification

- `pytest tests/test_leads_intake.py -q` -> 40 passed (36 prior + 4 new; the
  extended wiring test asserts the Resend routing). Run this session.
- `pytest tests/test_leads_intake.py tests/test_email_actions_api.py
  tests/test_email_auto.py tests/test_email_graph_sync_routing.py
  tests/test_extracted_competitive_email_provider_port.py -q` -> 89 passed (no
  regression on the shared email path). Run this session.
- Post-deploy smoke: submit a test estimate form (or POST `/api/v1/leads/intake`)
  and confirm the acknowledgement arrives From `info@effinghamofficemaids.com`
  via Resend (Resend message log). The GA4/Resend live check is an operator step
  after the deploy env key is confirmed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/leads.py` | 12 |
| `atlas_brain/services/email_provider.py` | 12 |
| `atlas_brain/templates/email/request_acknowledgement.py` | 20 |
| `atlas_brain/tools/email.py` | 5 |
| `plans/PR-Lead-Ack-Resend-Sender.md` | 185 |
| `tests/test_leads_intake.py` | 55 |
| **Total** | **289** |
