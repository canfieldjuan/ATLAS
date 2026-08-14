# PR-EOM-Commercial-Billing-Candidates

## Why this slice exists

The current monthly EOM invoice task combines calendar selection with invoice creation, PDF persistence, service-invoiced markers, CRM interaction logging, notifications, and optional mail. Its `review_mode` still creates invoice rows and PDFs, so it is not a safe candidate-review boundary for the Billing & Payments workspace. Coordinating issue #2362 requires a provider-first, side-effect-free candidate preview before the tracker or Website can expose commercial billing review.

### Diff-budget justification

This is above the repository's 400-LOC soft target because source normalization, fixed-cent calculation, collision/blocker projection, stable fingerprinting, and its failure/retry/staleness proof form one indivisible read contract. Splitting the route from that proof would either publish a dead internal generator or recreate the unsafe scheduler boundary. Financial persistence, approval, delivery preference, tracker proxy, and Website UI remain separate follow-up slices.

### Problem-derived contract

- Root cause: the only existing commercial billing computation is embedded in a writeful scheduler, which prevents an operator from inspecting source evidence and blockers before an invoice, PDF, Gmail draft, email, or invoiced marker exists.
- Correct fix must touch/change: add one read-only candidate service that derives deterministic commercial candidates from active auto-invoice services, calendar events, and canonical CRM identity; expose it only through the deployed full receivables router; enroll a focused contract suite in the invoicing workflow.
- Must not change: monthly scheduler behavior, invoice/payment/deposit/receipt/MCP/Gmail state, existing recipient semantics, customer-type definitions, database schema, the dormant slim EOM profile, or tracker/Website consumers.

## Scope (this PR)

Ownership lane: eom/billing-candidate-preview
Slice phase: Vertical slice
Max files: 5

1. Add an authenticated GET preview that returns all active auto-invoice service bundles for one explicit month, including source events, line items, exact integer cents, explicit blockers, and a deterministic source fingerprint.
2. Make preview computation structurally read-only: it may call only service, calendar, and canonical-CRM reads; it cannot create invoices/PDFs/drafts/mail, mark services invoiced, log CRM interactions, or notify.
3. Treat unresolved source evidence as a blocker rather than a silent candidate selection: missing canonical customer/email/delivery preference, invalid rate/rate label, missing events/hours, zero/invalid total, non-commercial classification, and ambiguous keyword matches are visible in output.
4. Keep the delivery method null and block it when no explicit preference exists; later durable billing-run and Square work owns the migration and approval behavior recorded as H-13 in #2363.

### Review Contract

- Acceptance criteria:
  - An authenticated GET `/api/v1/receivables/commercial-billing-candidates?billing_period=YYYY-MM` reaches the active full receivables router and returns a bounded deterministic candidate response; a missing, malformed, or out-of-range period reaches no provider collaborator.
  - A normal commercial Per Visit bundle groups confirmed calendar events by source date, exposes location/UID evidence, calculates cents with `Decimal(str(value))`, and carries a stable candidate key plus SHA-256 source fingerprint.
  - Per Month has one monthly line without fabricated events; Per Hour is visibly blocked until hours are supplied; no-event, invalid-rate, invalid-rate-label, zero-total, missing canonical identity/email, missing delivery preference, non-commercial type, and ambiguous keyword inputs each emit their own blocking code.
  - Equal source inputs produce byte-for-byte equivalent candidate content and fingerprint across retry; changing a rate, canonical recipient, service identity, or calendar event changes the fingerprint without performing a write.
  - Route authorization rejects missing/wrong bearer credentials before the preview service is called, and calendar or canonical-CRM source unavailability maps to a stable 503 response without leaking the underlying exception; a recovered source permits a clean retry.
  - Read-only fakes fail if any create/update/delete/PDF/Gmail/mail/CRM-interaction/notification method is reached; normal, failure, retry, stale-evidence, and recovery cases prove the public computation does not invoke them.
  - The workflow-selected existing invoice validation tests pass unchanged and the scheduler has no diff, proving this route does not rewire the writeful scheduler or change its legacy behavior.
  - Both pull-request and main-push workflow path filters and the local workflow-equivalent command enroll the new focused candidate suite.
- Reachability proof: a FastAPI app including the actual `atlas_brain.api.invoicing.receivables.router` uses its production auth dependency and an explicit candidate-service dependency override; service tests inject read-only collaborators rather than replacing first-party globals.
- Affected surfaces: `atlas_brain/services/commercial_billing_candidates.py`, the active full receivables router, invoice-check workflow enrollment, and the focused contract suite.
- Risk areas: financial rounding, accidental effects, calendar ambiguity, canonical-customer disclosure, stale evidence, authorization, and deployment topology.
- Reviewer rules triggered: R1, R2, R3, R5, R7, R8, R12, R14. R3 is satisfied by pure deterministic retry behavior; R8 has no idempotency write because this GET commits no financial or delivery state.

### Boundary-change enumeration

- Boundary path/seam: authenticated full-provider `GET /receivables/commercial-billing-candidates` to `CommercialBillingCandidateService.preview`.
- Replaced-path behaviors: none; the old scheduler remains uncalled and unchanged. This is an additive route with no fallback to `monthly_invoice_generation.run`.
- Guard-relevant fields: `billing_period` is strict `YYYY-MM`; calendar event status/date, service rate/rate label/keyword, canonical recipient/customer type, and output blocker/fingerprint fields are normalized at the new service boundary.
- Caller x input shape: tracker will later call the GET with one query value; direct callers receive 401 before service access without the dedicated receivables bearer token, 422 for malformed query input, 503 for source availability failure, or a JSON preview result.

### Deployed-config probing

- Deployed/default config values: existing `settings.invoicing.auto_invoice_calendar_id` is read only as the current commercial-calendar selector; no feature flag, email setting, save path, or delivery credential is read.
- Explicit value probe: a nonempty injected calendar ID is passed only to calendar `list_events` and becomes fingerprint evidence.
- Absent value probe: an empty configured calendar ID becomes `None`, preserving existing provider default-calendar read semantics without selecting a delivery method.
- Default-session/default-context probe: service and route tests construct explicit read-only collaborator/config values; no live provider, user session, financial row, Gmail account, or environment credential is touched.
- Side-effect ordering: all service calls are reads. Candidate serialization/fingerprinting completes after those reads; no write or delivery operation exists before, during, or after it.

### Closure declaration

- Rate-label set: CLOSED and ENUMERATED from the current scheduler's documented branches: `Per Visit`, `Per Month`, and `Per Hour`. Unknown labels yield `invalid_rate_label` and no implicit Per Visit behavior.
- Public blocker-code set: CLOSED and AUTHORED HERE by the candidate contract. A source condition outside its named set becomes `source_evidence_invalid`, remains blocking, and is preserved in the candidate's source evidence rather than being silently accepted.
- Output ordering: CLOSED and deterministic: candidates by canonical contact UUID string, services by service UUID string, and source events by UTC start then UID.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_candidates.py`
- `plans/PR-EOM-Commercial-Billing-Candidates.md`
- `tests/test_commercial_billing_candidates.py`

## Mechanism

`CommercialBillingCandidateService` accepts explicit read collaborators for active service agreements, calendar evidence, canonical customer/recipient lookups, and the configured commercial calendar ID. It parses the requested calendar month before any read, reads active auto-invoice agreements and confirmed in-period events, assigns matching events using the existing longest-keyword/alphabetical tie rule only for a deterministic preview, and records every multi-match as an approval blocker rather than concealing it.

The service groups line items by canonical contact ID and preserves source dates, event IDs, summaries, calendar IDs, and locations. It converts every persisted numeric input through `Decimal(str(value))`, rejects nonfinite/non-cent data, performs all arithmetic in integer cents, and applies `ROUND_HALF_UP` exactly once when calculating tax cents. It returns no invoice-shaped database object: only a preview candidate with a candidate key, explicit line/evidence values, zero-or-more blocker objects, null delivery method, and a SHA-256 fingerprint of canonical JSON source evidence. The response is deterministic and can therefore be regenerated safely; it does not persist a run or pretend to approve a candidate.

The active full router imports the service through an explicit dependency seam and maps only `CommercialBillingCandidatesUnavailableError` to a stable 503. It does not import the monthly task, invoice repository, PDF renderer, Gmail tooling, mail sender, notification client, or CRM interaction writer. The dormant `atlas_brain.eom_api.receivables` copy is deliberately unchanged: deployed evidence identifies the full `atlas_brain.api.invoicing.receivables` router as the active provider, and the new reusable service is API-independent. Any future slim-profile activation must mount the full contract or add its own compatibility proof; H-06 remains open in #2363.

## Intentional

- The preview is an additive read contract and is safe to deploy before tracker or Website consumers. It does not create a billing run, invoice, PDF, draft, sent state, payment, service marker, audit record, or email.
- `deliveryMethod` is null and `missing_billing_delivery_preference` is blocking for every otherwise eligible commercial candidate because current code has no canonical delivery preference. Customer type never implies Gmail or Square.
- Event locations are source evidence, not canonical service-site identity. A service-specific `calendar_id` remains evidence only; this slice preserves the current global-calendar selection rather than changing invoicing behavior.
- Per-hour source events are retained but not assigned fabricated hours. Approval is blocked rather than estimating a total.
- The existing invoice repository still has legacy float conversion. This new read-only candidate path uses exact cents but does not alter old invoice output; H-01 remains a prerequisite before an approval writer is enabled.

## Deferred

- H-13 in #2363: add an audited persisted delivery preference, manual Square reference capture, and canonical service/site/calendar identity before any candidate approval writer.
- H-14 in #2363: repair the base-equal maturity-ratchet baseline omission for the existing receivables module and separately enroll receipt-template coverage; this slice does not accept or broaden that baseline.
- #2362 next slices: durable billing-run persistence and staleness reconciliation; tracker proxy; Website review UI; explicit approval/invoice/PDF/Gmail draft recovery; sent-mail reconciliation; Square queue; operating documentation.
- H-06 in #2363: evaluate the duplicated full/slim receivables route models before a provider-wide billing-run API expansion.

Parked hardening: H-14 is recorded without changing its baseline in this PR. Financial writer hardening, schema migration, and delivery preference are deliberately deferred because the preview has no write path.

## Verification

- `python -m pytest tests/test_commercial_billing_candidates.py -q` — 11 passed (local; all collaborators are fakes, every address uses example.test, and calendar plus both canonical-CRM reader outage/recovery paths leave zero writes).
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55441/atlas_receivables_test python -m pytest tests/test_monthly_invoice_generation.py -k "update_invoice_clears_needs_hours_when_line_items_are_billable or line_items_are_billable_requires_all_positive_quantities" -q` — 2 passed (isolated local PostgreSQL 16).
- Same isolated database: the workflow's receivables/repository command with `tests/test_commercial_billing_candidates.py` enrolled — 231 passed, 1 unrelated torch/pynvml deprecation warning.
- Same isolated database: the workflow's invoicing MCP/OAuth command — 43 passed.
- `python -m ruff check atlas_brain/api/invoicing/receivables.py atlas_brain/services/commercial_billing_candidates.py tests/test_commercial_billing_candidates.py` — passed.
- `python -m compileall -q atlas_brain/api/invoicing/receivables.py atlas_brain/services/commercial_billing_candidates.py` — passed.
- `python scripts/sync_pr_plan.py plans/PR-EOM-Commercial-Billing-Candidates.md --check`, `git diff --check`, and `git diff --quiet origin/main -- atlas_brain/autonomous/tasks/monthly_invoice_generation.py` — passed.
- Exact local maturity-ratchet checks found two failures that reproduce unchanged on the base revision: the missing API baseline entry for pre-existing receivables mocks and receipt-template coverage enrollment. H-14 records the evidence; neither failure is accepted or altered in this slice.
- The broad legacy monthly-task suite was not run with `ATLAS_INVOICING_ENABLED=true`: doing so could select a configured non-test provider or datastore. The exact workflow-selected existing invoice tests passed against isolated PostgreSQL; this slice leaves the scheduler unchanged.
- Repository local pre-push/review wrapper will run immediately before publication; GitHub-hosted statuses are observed but do not replace local acceptance evidence per the operator instruction.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 5 |
| `atlas_brain/api/invoicing/receivables.py` | 34 |
| `atlas_brain/services/commercial_billing_candidates.py` | 996 |
| `plans/PR-EOM-Commercial-Billing-Candidates.md` | 120 |
| `tests/test_commercial_billing_candidates.py` | 742 |
| **Total** | **1897** |
