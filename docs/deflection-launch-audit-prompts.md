# Deflection paid funnel — launch-readiness audit prompts

Two complementary blocker-hunting prompts for the FAQ-deflection PAID funnel, one per repo.
They share a report format and dedupe against the tracker (`canfieldjuan/atlas` issue **#1386**),
so findings from both land in one place.

- **Prompt A (this repo, `canfieldjuan/atlas`)** — the ATLAS backend + the serverless funnel
  reference copy at `portfolio-ui/api/content-ops/deflection/*.js`.
- **Prompt B (`canfieldjuan/atlas-portfolio`)** — the DEPLOYED public funnel (juancanfield.com).
  Resolves the cross-repo unknowns Prompt A is blind to, and is told to diff against this repo's
  reference copy so divergence (the root cause of the dominant bug class) is caught directly.

**The dominant bug class:** the portfolio and ATLAS disagree about state at checkout/fulfillment
time (price, currency, result-page URL, report existence, payment-method/event handling). ATLAS
fails closed on mismatch, so **the buyer pays and gets nothing.** The highest-yield technique is
to take each confirmed bug and *generalize it to every other value that crosses the boundary.*

---

## Prompt A — ATLAS backend + serverless reference copy (`canfieldjuan/atlas`)

```
You are auditing the FAQ-deflection PAID funnel for production-launch blockers. Several have
already been found; your job is to find the REST in one pass — especially siblings of the
known ones — and report them in a precise, verifiable format.

## Product (the money path)
A buyer uploads a support-ticket CSV -> deterministic clustering -> a free locked "snapshot" ->
Stripe Checkout ($1,500 one-time) -> Stripe webhook unlocks the paid report -> an email delivers
the result link. Anything that takes money but fails to deliver value is a blocker.

## Two repos that must AGREE (read both)
- `canfieldjuan/atlas` - the ATLAS backend + the serverless funnel in `portfolio-ui/api/content-ops/deflection/*.js` (submit, checkout, report, atlas-report, result-page, events).
- `canfieldjuan/atlas-portfolio` - the DEPLOYED public site (juancanfield.com). Treat divergence between this and `atlas` as a primary risk.
Key backend files: `atlas_brain/api/billing.py` (Stripe webhook), `extracted_content_pipeline/deflection_report_access.py` (paid state, `mark_paid`), `atlas_brain/content_ops_deflection_delivery.py` (delivery worker), `campaign_customer_data.py` (`_load_csv_rows`), `support_ticket_input_package.py` (clustering).

## PRIMARY LENS: portfolio <-> ATLAS state disagreement
The dominant bug class: the portfolio decides/sends something, ATLAS independently re-validates
it with different or stricter rules, rejects it, and the buyer pays but gets nothing.
Confirmed instances - find EVERY OTHER value that both sides touch and could disagree on:
- price (`>=` in portfolio vs exact allowlist in ATLAS)
- currency (portfolio accepts non-USD vs ATLAS exact-currency gate)
- result-page URL (portfolio default `/systems/...` vs live route `/services/...`)
- report existence (portfolio creates Checkout without ATLAS confirming the report exists)
- payment method / event type (async Checkout completes pending; ATLAS only fulfills `checkout.session.completed` with payment_status==paid and ignores `checkout.session.async_payment_succeeded`)
For EACH field crossing the boundary - `account_id`, `currency`, `request_id` shape, the
`source` metadata tag, `success_url`/`cancel_url`, `client_reference_id`, `price_id` vs inline
`price_data`, timeouts, event types - check: does ATLAS validate it differently than the
portfolio produces it? If yes and the mismatch strikes AFTER payment, it's a blocker.

## Also hunt these classes
1. Pay-but-no-value: any path where money is taken but the artifact isn't unlocked/delivered.
2. Silent failures: `UPDATE ... WHERE` that can affect 0 rows with no check; swallowed
   exceptions; handlers returning 2xx after a failed side effect (so Stripe won't retry).
3. Config/env drift & fail-open defaults: defaults pointing at stale values; "only works if
   ENV_X is set"; test-vs-live key/price/endpoint mismatch.
4. Idempotency: double-charge (no idempotency key / no "already paid?" check), double-send,
   replay safety.
5. Trust/security: capability-URL (`request_id`) entropy + who can generate it, access
   control, PII retention/deletion, XSS / CSV-formula-injection on rendered/exported ticket
   text, missing rate-limits on public/unauthenticated endpoints.
6. Real-world input: code/tests that only exercise clean demo data - encoding/BOM/delimiter,
   missing columns, untagged tickets that fragment clustering, exports lacking reply text.
7. Operational gates deferred to "later": scheduler/cron, monitoring, alerting - anything
   that blocks an UNATTENDED launch.
8. Stale proof: tests that hardcode a URL/amount/path (may be locking in a bug); outdated
   smoke artifacts.

## Search heuristics (tells)
`>=` / `<=` on amounts; format-only validation; client-supplied values trusted server-side;
default strings that differ from the live route; `except: pass`; missing rowcount checks;
`os.environ`/`process.env` with fallback defaults; hardcoded literals in tests; `TODO`,
`deferred`, `later slice`, `manual`, `--send`; public routes with no auth/rate-limit.

## Rules
- VERIFY every finding against actual code with file:line. Do not speculate.
- If something can only be confirmed live (atlas-portfolio deploy, real Stripe dashboard),
  mark it "NEEDS LIVE CHECK" and state the exact check.
- Cross-check the existing tracker (`canfieldjuan/atlas` issue #1386) and SKIP duplicates;
  only report new findings or sharper detail on existing ones.
- Stay on the paid funnel + its messy-reality paths. Don't wander into unrelated ATLAS.

## Output - one block per finding
- SEVERITY: BLOCKER / operational-gate / stale-proof / hardening
- Claim: one sentence
- Mechanism: what the code does, with file:line on both sides if it's a boundary bug
- Why it matters: the buyer/business impact (lead with "buyer pays and ...")
- Required before launch: concrete fix(es)
- How to verify: a test to add or a live check to run
```

---

## Prompt B — deployed public funnel (`canfieldjuan/atlas-portfolio`)

```
You are auditing `canfieldjuan/atlas-portfolio` (the DEPLOYED public funnel at
juancanfield.com) for production-launch blockers in the FAQ-deflection PAID flow. A parallel
audit of the ATLAS backend repo (`canfieldjuan/atlas`) is underway and tracked in its issue
#1386; your job is the PORTFOLIO half - resolve the cross-repo unknowns that audit couldn't
see, and find portfolio-side blockers - reported in the same precise, verifiable format.

## Product (the money path)
Buyer uploads a support-ticket CSV -> deterministic clustering -> free locked "snapshot" ->
Stripe Checkout ($1,500) -> Stripe webhook unlocks the paid report -> email delivers the link.
Anything that takes money but fails to deliver value is a blocker. This repo is the
PUBLIC, internet-facing half - treat it as the primary attack + UX surface.

## The two repos must AGREE
`atlas-portfolio` (this repo) sends/decides things; the ATLAS backend independently
re-validates them and FAILS CLOSED on mismatch - so when they disagree, the buyer pays and
gets nothing. The `atlas` repo contains a reference copy of the funnel serverless functions at
`portfolio-ui/api/content-ops/deflection/*.js`. Compare this repo's deployed versions
against that copy and flag every divergence (logic, defaults, validation, routes).

## RESOLVE THESE FIRST (the backend audit was blind to them - each is launch-gating)
1. `request_id` generator - find where it's created (`content-ops-<...>`). Confirm it's
   `crypto.randomUUID()`-grade (>=122-bit). It is the ONLY secret protecting a paid report,
   it's client-supplied to the backend, and the backend only checks a permissive regex. If
   it's weak/sequential/guessable -> paid reports are enumerable. Report exact source + entropy.
2. CSV parse location - does THIS repo parse the CSV in client/serverless JS, or hand raw
   bytes to the ATLAS backend? If it parses here, audit encoding/BOM/delimiter/quoting/HTML
   handling (real Zendesk/Intercom/Help Scout/Freshdesk exports, not clean demo data).
3. Result-page route reconciliation - what is the ACTUAL canonical route? The backend's
   email builder defaults to `/systems/support-ticket-deflection/results/{id}` while a route
   table uses `/services/faq-deflection/results/{id}`. Confirm the real deployed path and make
   the email link, the route, and any rewrite all agree - a wrong link = paid buyer hits 404.
4. Server-error rendering - when the backend returns a typed error
   (`private_blob_unavailable`, `atlas_submit_invalid_json`, timeout, 409 missing-report),
   does the upload/result page show a clear recoverable message, or a blank/dead page?
5. Paid-report rendering XSS - the report renders verbatim customer ticket text. Confirm
   every field is HTML-escaped on output (and that any CSV export prefix-escapes `= + - @`).
6. Checkout pre-verification - does this repo's Checkout creation confirm with ATLAS that
   the report EXISTS, belongs to the configured account, and is locked/unpaid BEFORE calling
   Stripe? If it charges on syntactic `request_id` alone -> pay-but-locked.
7. Price + currency exactness - does this repo charge an amount AND currency that EXACTLY
   match ATLAS's config, or accept any `unit_amount >= 150000` / any 3-letter currency?
   Either drift -> pay-but-locked. Also: is Price validation skipped on the restricted-key path?

## PRIMARY LENS (find more): portfolio -> ATLAS state disagreement
For every value this repo sends across the boundary - `account_id`, `currency`, `request_id`
shape, the `source` metadata tag, `success_url`/`cancel_url`, `client_reference_id`,
`price_id` vs inline `price_data`, payment_method_types, timeouts, event types - ask: does the
backend validate it differently than this repo produces it, and would the mismatch strike
AFTER payment? Those are blockers.

## Also hunt (public-surface emphasis)
- Trust/security: capability-URL entropy (#1), unauthenticated upload/submit endpoints with
  no rate-limit (cost/DoS abuse), PII exposure (customer emails/ticket text in client state,
  logs, or public URLs), XSS/CSV-injection.
- Pay-but-no-value paths; fail-open defaults ("works only if ENV_X set"); test-vs-live keys.
- Idempotency: double-click -> double Checkout; duplicate submit.
- Stale proof: tests/smoke docs hardcoding a stale URL/amount/path; the 2026-05-30 hosted
  smoke showed a production 404 - rerun and attach fresh proof.

## Rules
- VERIFY every finding against actual code with file:line. Don't speculate.
- For anything provable only against the LIVE deploy or the Stripe dashboard, mark
  "NEEDS LIVE CHECK" with the exact check.
- Cross-check `canfieldjuan/atlas` issue #1386 and SKIP duplicates; report new findings or
  sharper detail only.
- Stay on the paid funnel + its messy-reality/public-attack paths.

## Output - one block per finding
- SEVERITY: BLOCKER / operational-gate / stale-proof / hardening
- Claim: one sentence
- Mechanism: what the code does, file:line (and the `atlas` copy it diverges from, if any)
- Why it matters: lead with the buyer/business/security impact
- Required before launch: concrete fix(es)
- How to verify: a test to add or a live check to run
```
