# PR-EOM-SMS-Retry-Resume

## Why this slice exists

Issue #2230 records two parked hardening items from the EOM funnel ingress
work. Current code already contradicts the workflow-filter half: both the
`pull_request` and `push` path filters in
`.github/workflows/atlas_eom_lead_pipeline_checks.yml` include
`atlas_brain/storage/migrations/352_eom_inbound_delivery_receipts.sql`.

The remaining live gap is inbound SMS retry ownership: an SMS row can be
persisted while CRM/contact processing is incomplete, and a duplicate provider
delivery must neither skip that work nor duplicate side effects.

This is approximately 2767 LOC by the repo diff-size audit because the review repair has to
state and test the full SMS retry ownership model in one pass: claim
eligibility, completed-unlinked rows, owner-token-fenced processing, persisted
context rehydration, before-ack 503 retry surfacing, active-lease retry
surfacing, stale-worker recovery, CI-enrolled live Postgres evidence, and
post-link continuation. Review repair also covers linked-but-incomplete rows:
they remain resumable until processing reaches a terminal state, but resume with
the existing contact id so the CRM link/interaction side effect is not repeated.
The current review round also folds in route-entry ack budgeting, authoritative
claim row state, narrow inbound auto-reply outbox recovery, durable finalization,
and real-Postgres proof for retry-pending/status fencing.
It also includes one test-only unit-gate stabilization after CI proved an
unrelated content-factory linearity guard was asserting runner speed instead of
scaling behavior.

### Problem-derived contract

- Root cause: `handle_inbound_sms` treated SMS persistence and CRM/contact
  processing as separable background work while duplicate detection treated an
  existing `MessageSid` as enough evidence that the work was complete. That can
  lose CRM work. The previous lease repair then stored ownership in
  `error_message` but let normal status writes clear it and acknowledged
  provider retries while an incomplete row was still leased.
- Correct fix must touch/change: SMS processing must atomically claim
  incomplete rows before CRM/intelligence work and receive authoritative row
  state from that claim, fence terminal updates and outbound sends with an owner
  token, persist a recoverable inbound auto-reply send decision before provider
  contact, require durable terminal finalization, process first deliveries and
  resumable duplicates before provider
  acknowledgement, and return 503 for claim/CRM handoff failures or active
  incomplete leases so provider retry is the durable retry mechanism for this
  slice. Provider-facing work must have a bounded ack budget that begins at
  route entry. Retries must use persisted row values and persisted business
  context; stale stored context IDs must fail closed. CRM infrastructure
  failures must be distinguishable from legitimate insufficient identity, and
  post-link interaction logging failures must not undo the contact link or
  suppress later SMS handling.
- Must not change: EOM lead lifecycle rules, CRM provider ownership rules,
  estimate booking, customer/site onboarding, payments, customer-visible
  auto-reply copy, or the already-enrolled migration-352 workflow filters. The
  unrelated content-factory test-only stabilization must not change product
  verification semantics.

## Scope (this PR)

Ownership lane: eom-crm/sms-ingress-retry
Slice phase: production hardening
Max files: 7
Actual files: 7

1. Resume linked-but-incomplete duplicate SMS rows while preserving their
   existing contact link.
2. Add owner-token fencing to SMS contact-processing claims and terminal
   updates using existing SMS row fields.
3. Resume only explicitly incomplete/recoverable rows: `received`,
   `processing`, or `retry_pending`.
4. Run first-delivery and duplicate-resume processing under a bounded
   provider-ack deadline so claim/CRM failures, active incomplete leases, and
   slow processing can return 503.
5. Preserve persisted body/media/from/to/context values during retry
   rehydration and fail closed for stored-but-unresolvable context IDs.
6. Continue SMS action planning/notification/auto-reply after a contact is
   linked even if non-EOM interaction logging fails.
7. Persist inbound auto-reply decisions before provider send, distinguish
   pending from provider-accepted sends, retry pending sends, and skip only
   already-sent replies.
8. Enroll the live Postgres claim proof in the EOM workflow that already owns a
   Postgres service.
9. Add focused tests for owner-token claim fencing, before-ack 503 behavior,
   active-lease 503 behavior, persisted context handling, completed-unlinked
   skip, CRM infra retry propagation, post-link continuation, route-entry ack
   timeout, slow-but-admitted processing, auto-reply idempotency/recovery,
   finalization failure retry, notification dedupe, and real-Postgres retry
   fencing.
10. Stabilize the unrelated content-factory linearity guard by checking scaling
   behavior rather than absolute wall-clock runner speed.

### Review Contract

Acceptance criteria:

- A duplicate inbound SMS with `contact_id` set and incomplete/recoverable
  status keeps running processing before acknowledgement, but passes the
  persisted contact id forward instead of creating/linking CRM again.
- A duplicate inbound SMS whose unlinked row is already complete returns empty
  TwiML and does not run processing again.
- A duplicate inbound SMS with no `contact_id` and incomplete/recoverable status
  runs `_process_inbound_sms` before acknowledgement with persisted row values
  and `claim_processing=True`.
- A processor whose claim is already owned exits before intelligence or
  auto-reply side effects.
- A provider retry that observes an incomplete row still owned by another
  worker receives 503 rather than consuming the retry with 200.
- Stale processing rows can be reclaimed only by replacing the owner token; old
  owners cannot complete or send after losing ownership, and normal status or
  notification writes do not clear the active owner token before completion.
- New inbound SMS rows persist before contact processing; claim/CRM handoff
  failures and provider-ack timeouts return 503 before acknowledgement.
- The provider-ack timeout covers media parsing, provider callback, duplicate
  lookup, persistence, recovery lookup, duplicate resume, and processing.
- Persisted empty body/media values are preserved during retry rehydration.
- Persisted `business_context_id` is restored before phone-number routing is
  used as a legacy fallback.
- A stored but unresolvable `business_context_id` returns 503 and never falls
  through to phone-number routing.
- CRM handoff failures return 503 before acknowledgement instead of becoming
  terminal completed-unlinked rows or unconsumed retry-pending rows.
- CRM database/provider infrastructure failures propagate as retryable while
  legitimately identityless SMS content can still finish without a contact.
- Post-link interaction logging failures do not undo the contact link or stop
  action planning, notification, and auto-reply processing.
- Claimed processing receives authoritative row/contact state from the claim
  operation, not a second racing reload.
- Auto-reply sends are recoverable for a given inbound SMS row: a retry that
  sees a pending reservation retries the provider send, a retry that sees a sent
  reservation skips duplicate provider contact, and provider/mark-sent failures
  do not complete the inbound lease.
- A retry whose inbound row already has `notified = TRUE` does not re-post the
  manager notification.
- The before-ack budget is not shorter than the configured SMS LLM/auto-reply
  latency this path admits.
- Final processing completion failures or owner-mismatch no-ops propagate a
  retry/lost-ownership outcome instead of acknowledging success.
- The EOM workflow sets `ATLAS_EOM_SMS_RETRY_POSTGRES_URL` and includes
  `tests/test_eom_sms_webhook_retry.py`, so the live claim proof runs against
  the workflow Postgres service instead of skipping.
- The unrelated content-factory linearity guard still fails on quadratic
  behavior but no longer depends on a fixed 1-second CI runner budget.

Affected surfaces: inbound SMS webhook duplicate handling, SMS intelligence CRM
link handling, SMS message repository claim helpers, and focused tests.

Risk areas: duplicate auto-replies, duplicate CRM interactions, stale-worker
completion overwrites, and retry payload drift replacing persisted SMS data.

Triggered reviewer rules: R1/R2/R5/R6/R7/R8/R12 code-grounded issue-claim
verification, webhook retry behavior, crash/race-safe processing, and focused
test coverage.

Execution model and invariant mapping: one SMS row is the lease record. A worker
may start CRM/contact processing only by atomically changing an unlinked
`received`, `retry_pending`, or stale `processing` row to `processing` with a
fresh owner token. Non-terminal status/notification writes keep that token.
Every terminal transition or outbound-send boundary checks the token; a stale
owner whose token has been replaced cannot complete or send. If another
delivery arrives while a non-stale owner holds the incomplete row, the handler
returns 503 so the provider retry remains alive rather than being consumed.
Slow provider-facing work is capped from route entry by
`SMS_INBOUND_BEFORE_ACK_TIMEOUT_SECONDS`; timeout is retryable and the row stays
reclaimable by lease expiry. CRM contact and interaction writes are keyed by the
provider message identity (`source_ref` / `crm_event_id`) so retries use the
same idempotency evidence rather than mutable retry payloads. Auto-replies use a
deterministic outbound reservation keyed to the inbound SMS row before provider
contact, so a later retry cannot send a duplicate reply after crash or
cancellation.

Reachability proof: focused unit tests call the duplicate-retry resume helper,
the `/sms/inbound` handler, `SMSMessageRepository.claim_contact_processing`, and
`_process_inbound_sms` with injected external-edge fakes, asserting claim,
before-ack processing/503, active-lease 503, timeout 503, skip, persisted
value/context, owner-token preservation/fencing, completion, post-link
continuation, linked-row resume with an existing contact id, CRM infra retry
propagation, route-entry timeout behavior, auto-reply reservation idempotency,
and retry-pending behavior. The EOM workflow enrolls the live Postgres claim
and retry-fencing proof.

### Files touched

- `atlas_brain/api/comms/webhooks.py`
- `atlas_brain/comms/sms_intelligence.py`
- `atlas_brain/storage/repositories/sms_message.py`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `plans/PR-EOM-SMS-Retry-Resume.md`
- `tests/test_content_factory_copy_verification.py`
- `tests/test_eom_sms_webhook_retry.py`

## Mechanism

- `SMSMessageRepository.claim_contact_processing` atomically marks a resumable
  SMS row `processing`, stores an opaque owner token in `error_message`, returns
  authoritative row state plus `_claim_acquired`, and allows stale `processing`
  reclaim only by replacing that owner token.
- `SMSMessageRepository.owns_contact_processing`,
  `touch_contact_processing_owner`, `update_contact_processing_status`,
  `mark_contact_processing_retry_pending`, and
  `mark_contact_processing_complete` fence heartbeat, non-terminal progress,
  and terminal updates by owner token.
- `SMSMessageRepository.reserve_auto_reply_for_inbound` persists a deterministic
  outbound auto-reply reservation before provider contact,
  `get_auto_reply_for_inbound` returns its pending/sent state, and
  `mark_auto_reply_sent` records provider acceptance afterwards.
- `_process_inbound_sms` claims before CRM/intelligence work, uses the claimed
  row to preserve an existing contact link, heartbeats/checks the owner token
  before side-effect boundaries, returns `retry_pending` to before-ack callers
  on claim/CRM failures, distinguishes skipped intelligence from terminal
  opt-out/spam, keeps the owner token alive across processing status/notification
  writes, skips already-sent notifications on retry, retries pending auto-reply
  sends, skips sent auto-replies, and only returns success after the terminal
  row transition is durably written by the active owner.
- `handle_inbound_sms` persists the row before awaiting `_process_inbound_sms`;
  it starts the ack deadline at route entry and returns 503 for before-ack retry
  outcomes, active incomplete leases, and ack-budget timeouts so
  SignalWire/provider retry remains the durable retry mechanism for this slice.
- The duplicate retry branch awaits `_process_inbound_sms` only when the row is
  still incomplete or recoverable, including linked rows that have not reached a
  terminal status; completed rows return empty TwiML without more work.
- Retry rehydration falls back only for missing/`None` persisted values, so an
  intentionally empty body or media list is not replaced by retry payload drift.
- Retry context rehydration uses persisted `business_context_id` first, fails
  closed when that stored ID no longer resolves, and uses phone-number routing
  only for legacy rows without a stored context.
- `sms_intelligence.process_inbound_sms` accepts an existing contact id for
  resumed linked rows, returns an explicit retry outcome for pre-link CRM
  infrastructure failures, returns a separate skipped-intelligence outcome for
  disabled/empty-body cases, and logs post-link interaction logging failures
  while the SMS flow continues.
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml` runs
  `tests/test_eom_sms_webhook_retry.py` with the workflow Postgres database URL
  exported as `ATLAS_EOM_SMS_RETRY_POSTGRES_URL`.
- The content-factory unit-gate guard uses a generous large-input CPU-time
  ceiling instead of asserting a fixed wall-clock ceiling.

## Intentional

- The fix uses existing `status`, `processed_at`, and `error_message` fields
  instead of adding schema. `received`/`retry_pending` are immediately
  resumable, stale `processing` is recoverable with owner-token fencing, and
  `ready`/`notified` are complete even when no contact is linked.
- Normal successful duplicate retry responses remain empty TwiML; claim/CRM
  handoff failures, active incomplete leases, and provider-ack timeouts return
  503 before acknowledgement so the provider has a reason to retry.
- The inbound auto-reply outbox is intentionally narrow: it recovers pending
  direct auto-reply sends and skips already-sent replies for the same inbound SMS
  row without changing customer-visible auto-reply copy or introducing a broader
  worker queue.
- The workflow edit is limited to this SMS retry test enrollment. The migration
  352 path filters remain unchanged because they were already enrolled.
- The content-factory test stabilization is test-only and does not change
  content-factory product verification semantics.

## Deferred

- A broader exactly-once SMS processing state machine with a dedicated owner
  column and durable worker queue is deferred until inbound SMS volume or
  duplicate side effects justify it.
- Reminder dedupe and a broader outbound-message worker are deferred;
  linked-but-incomplete retries resume through the existing contact id, and this
  slice only reserves the direct inbound auto-reply tied to the retried SMS row.

Parking predicate: this slice parks only hardening that would require a new
durable queue/schema-backed worker system or reminder/outbound orchestration
outside the direct inbound SMS retry ownership path.

Parked hardening under that predicate: broader durable-queue orchestration and
reminder/outbound orchestration only.

## Verification

- `python scripts/audit_plan_doc.py plans/PR-EOM-SMS-Retry-Resume.md` — PASS; required plan sections present.
- `python -m py_compile atlas_brain/api/comms/webhooks.py atlas_brain/comms/sms_intelligence.py atlas_brain/storage/repositories/sms_message.py tests/test_eom_sms_webhook_retry.py` — PASS.
- `pytest -q tests/test_eom_sms_webhook_retry.py` — 35 passed, 1 skipped, 1 warning locally; the skipped live Postgres proof is CI-enrolled through `ATLAS_EOM_SMS_RETRY_POSTGRES_URL`.
- `python -m pytest -q tests/test_eom_lead_ingress.py::test_real_sms_link_uses_eom_lead_resolver tests/test_eom_lead_ingress.py::test_sms_link_uses_provider_identity_without_a_local_sms_row tests/test_eom_lead_ingress.py::test_sms_fallback_uses_eom_lead_resolver tests/test_content_factory_copy_verification.py::test_scope_lookup_scales_with_negation_scopes_present` — 4 passed.
- `python -m pytest -q tests/test_content_factory_copy_verification.py` — 324 passed.
- `python scripts/audit_plan_doc_files_touched.py plans/PR-EOM-SMS-Retry-Resume.md origin/main` — PASS; plan files match git diff.
- `python scripts/audit_plan_doc_diff_size.py plans/PR-EOM-SMS-Retry-Resume.md origin/main` — PASS; diff-size drift remains inside the plan threshold.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-EOM-SMS-Retry-Resume.md` — PASS; all path claims resolve.
- `python scripts/audit_review_rules_triggered.py origin/main --plan plans/PR-EOM-SMS-Retry-Resume.md` — PASS; plan declares every rule the diff triggers.
- `python scripts/audit_pr_body.py --repo-root . --base-ref origin/main /tmp/pr2246-body.md` — PASS.
- `git diff --check origin/main -- . ':!node_modules'` — PASS.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*'` — ratchet gate passed; no new brittleness above baseline.
- `python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8` — ratchet gate passed; no new brittleness above baseline.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 6 |
| `atlas_brain/api/comms/webhooks.py` | 539 |
| `atlas_brain/comms/sms_intelligence.py` | 131 |
| `atlas_brain/storage/repositories/sms_message.py` | 369 |
| `plans/PR-EOM-SMS-Retry-Resume.md` | 283 |
| `tests/test_content_factory_copy_verification.py` | 18 |
| `tests/test_eom_sms_webhook_retry.py` | 1421 |
| **Total** | **2767** |
