# PR-Security-Paid-Funnel-Alert-Channel

## Why this slice exists

#1656 still lists "Wire alert delivery to a real channel" for the paid
deflection funnel. The earlier paid-funnel alert-sink slice routed
`DEFLECTION_PAID_FUNNEL_INCIDENT` records into the centralized alert manager,
but the application startup contract still admits a live-shaped Stripe
deflection setup whose only guaranteed callback is `log_alert_callback`.

Root cause: the paid-funnel incident emitter and alert rule became routable,
but startup never checked the checkout-terms relationship: "deflection price
plus accepted amount/currency can expose checkout" and "a non-log alert channel
is configured." Review caught that reconstructing the predicate from adjacent
Stripe secret/webhook/default-amount settings was a symptom fix, so this PR now
mirrors the checkout terms predicate directly.

## Scope (this PR)

Ownership lane: security/paid-funnel-alert-channel
Slice phase: Production hardening
Max files: 5

1. Add a startup preflight that treats deflection checkout price plus accepted
   amount/currency as requiring centralized alerts plus ntfy delivery settings.
2. Prove fail-closed/pass cases for log-only, alerts-disabled, malformed ntfy,
   configured ntfy, unrelated Stripe with no price, and price without webhook.
3. Archive the merged #1820 plan doc as required teardown housekeeping; direct
   root-main housekeeping was not safe because the root checkout is another
   session's deflection branch.

### Review Contract

- Acceptance criteria:
  - [ ] Deflection price plus accepted amount/currency cannot pass with log-only
        callbacks.
  - [ ] Unrelated Stripe deployments without a deflection checkout price are
        not blocked by the default non-zero amount.
  - [ ] Deflection price still triggers when Atlas's Stripe webhook secret is
        omitted.
  - [ ] Disabling centralized alerts is not treated as a safe fallback for the
        paid funnel.
  - [ ] Ntfy with absolute HTTP(S) URL/topic passes; non-absolute URLs fail.
  - [ ] Repos/scripts that import config still can do so; enforcement is tied
        to application startup, not module import.
  - [ ] The existing incident emit/routing behavior remains untouched.
- Affected surfaces: Atlas startup alert initialization, focused startup tests,
  and plan archive housekeeping.
- Risk areas: accidentally blocking unrelated non-paid local runs, weakening
  the paid-funnel paging claim into another warning-only guard, and confusing
  log persistence with human notification.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R13, R14.

### Files touched

- `atlas_brain/main.py`
- `plans/INDEX.md`
- `plans/PR-Security-Paid-Funnel-Alert-Channel.md`
- `plans/archive/PR-Security-Labels-As-Code.md`
- `tests/test_atlas_main_voice_startup.py`

## Mechanism

`main.py` derives paid-funnel liveness from the same checkout terms the
deflection route enforces: non-empty price, positive amount, accepted amount,
and three-letter alphabetic currency. When true, early startup requires enabled
alerts, enabled ntfy, an absolute HTTP(S) ntfy URL, and a non-empty topic.

The preflight runs at application startup rather than Pydantic settings import.
That preserves script/config importability but makes a real service lifecycle
fail closed before it can process paid-funnel Stripe events with only the log
sink available.

## Intentional

- No new alert transport is introduced; this slice codifies ntfy as the
  existing real human-notification channel for the current paid-funnel surface.
- The incident emit sites in `api/billing.py` and
  `content_ops_deflection_delivery.py` stay unchanged because the previous
  alert-sink slice already proved they reach `AlertManager`.
- The preflight proves configuration shape, not network reachability to an ntfy
  server. A live transport probe would be a broader operational-health slice.
- The preflight does not require Atlas's Stripe webhook secret because the
  portfolio checkout can create the paid session after Atlas returns checkout
  terms; webhook delivery is a separate payment-confirmation failure surface.
- Local review's cross-layer caller hint for `lifespan` points at same-named
  FastAPI lifespan functions in other packages, not call sites of
  `atlas_brain.main.lifespan`.

## Deferred

- #1656 follow-up: add webhook/Sentry/Otel-style alert delivery as a second
  real channel if ntfy is not the long-term pager surface.
- #1656 follow-up: add an optional runtime health probe that verifies the
  configured alert transport can actually publish.

Parked hardening: none.

## Verification

- Focused startup preflight tests: `19 passed, 1 warning in 2.50s`.
- Python compile check for touched runtime/test modules: passed.
- Whitespace diff check: passed.
- Pending before push: local review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/main.py` | 121 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Security-Paid-Funnel-Alert-Channel.md` | 115 |
| `plans/archive/PR-Security-Labels-As-Code.md` | 0 |
| `tests/test_atlas_main_voice_startup.py` | 153 |
| **Total** | **392** |
