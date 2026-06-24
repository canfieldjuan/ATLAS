# PR-Security-Paid-Funnel-Alert-Channel

## Why this slice exists

#1656 still lists "Wire alert delivery to a real channel" for the paid
deflection funnel. The earlier paid-funnel alert-sink slice routed
`DEFLECTION_PAID_FUNNEL_INCIDENT` records into the centralized alert manager,
but the application startup contract still admits a live-shaped Stripe
deflection setup whose only guaranteed callback is `log_alert_callback`.

Root cause: the paid-funnel incident emitter and alert rule became routable,
but the orchestrating startup layer never checked the cross-config relationship
between "Stripe can take money for deflection reports" and "a non-log alert
delivery channel is configured." This PR fixes the root for the current ntfy
delivery surface by adding that startup preflight and tests; it does not change
the already-routed incident emit sites.

## Scope (this PR)

Ownership lane: security/paid-funnel-alert-channel
Slice phase: Production hardening

1. Add a startup preflight that treats a configured Stripe deflection-report
   funnel as requiring centralized alerts plus ntfy delivery settings.
2. Prove the preflight fails closed for log-only, alerts-disabled, and malformed
   ntfy channel configs, and admits a configured ntfy channel.
3. Archive the merged #1820 plan doc as required teardown housekeeping; direct
   root-main housekeeping was not safe because the root checkout is another
   session's deflection branch.

### Review Contract

- Acceptance criteria:
  - [ ] A paid deflection funnel with Stripe secret + webhook secret + positive
        checkout price config cannot pass startup preflight with log-only
        callbacks.
  - [ ] Disabling centralized alerts is not treated as a safe fallback for the
        paid funnel.
  - [ ] Enabling ntfy with a non-empty URL and topic satisfies the preflight.
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

`main.py` gets small pure helpers that derive whether the paid deflection
funnel is live-shaped from the existing `settings.saas_auth` fields:
Stripe secret, webhook secret, and explicit or default positive checkout amount
configuration. When that predicate is true, early startup requires
`settings.alerts.enabled`, `settings.alerts.ntfy_enabled`, a non-empty ntfy URL,
and a non-empty ntfy topic before the alert manager initializes.

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

- Focused startup preflight tests: `13 passed, 1 warning in 2.58s`.
- Python compile check for touched runtime/test modules: passed.
- Whitespace diff check: passed.
- Pending before push: local review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/main.py` | 76 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Security-Paid-Funnel-Alert-Channel.md` | 110 |
| `plans/archive/PR-Security-Labels-As-Code.md` | 0 |
| `tests/test_atlas_main_voice_startup.py` | 116 |
| **Total** | **305** |
