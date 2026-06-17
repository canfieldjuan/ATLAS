# PR-Churn-UI-Async-Race-Sweep

## Why this slice exists

The dependency queue has repeatedly been blocked by churn-ui test failures on
PRs that do not touch churn-ui. The latest review escalation named the
`Onboarding.test.tsx` stale-response case and the broader class of synchronous
`getBy*`/`getByText` assertions after async UI transitions.

Root cause: churn-ui tests mix route/data updates, mocked network promises, and
immediate synchronous DOM reads. That leaves assertions racing React's async
render cycle and can make an older mocked response look like product breakage
when the real issue is an unstable test harness. This slice fixes the root for
the queue-level flake class by awaiting the user-visible async states and by
locking the stale-response proof around explicit request ordering.

## Scope (this PR)

Ownership lane: security/dependencies
Slice phase: Production hardening

1. Stabilize churn-ui async tests that have blocked unrelated dependency PRs,
   starting with the reported Onboarding stale-response failure and same-route
   query hydration path.
2. Sweep the churn-ui test suite for the same failure mode where a test reads
   async UI state synchronously after navigation, user events, or mocked
   request resolution.
3. Keep production behavior changes limited to concurrency hardening required
   by the stale-response contract, if the focused tests expose a real product
   race.

### Review Contract
- Acceptance criteria:
  - [ ] The reported Onboarding same-route stale-response scenario waits for
        the newer request and proves the older response cannot render.
  - [ ] Same-class churn-ui test assertions changed by this PR use `findBy*`
        or `waitFor` around async route/data/user-event transitions rather
        than immediate synchronous reads.
  - [ ] The churn-ui focused test command and package test suite pass locally.
  - [ ] The existing broader churn-ui lint debt remains deferred and is not
        widened into this slice.
- Affected surfaces: frontend, tests.
- Risk areas: concurrency, CI stability, frontend behavior.
- Reviewer rules triggered: R1, R2, R9, R12, R13, R14.

### Files touched

- `atlas-churn-ui/src/pages/Affiliates.test.tsx`
- `atlas-churn-ui/src/pages/Challengers.test.tsx`
- `atlas-churn-ui/src/pages/Onboarding.test.tsx`
- `atlas-churn-ui/src/pages/Opportunities.test.tsx`
- `atlas-churn-ui/src/pages/Prospects.test.tsx`
- `atlas-churn-ui/src/pages/Reports.test.tsx`
- `atlas-churn-ui/src/pages/Reviews.test.tsx`
- `atlas-churn-ui/src/pages/VendorTargets.test.tsx`
- `atlas-churn-ui/src/pages/Vendors.test.tsx`
- `atlas-churn-ui/src/pages/Watchlists.test.tsx`
- `plans/PR-Churn-UI-Async-Race-Sweep.md`

## Mechanism

The Onboarding test drives the same route-change sequence with deferred
promises, asserts the first request starts, navigates to the second query,
asserts the second request starts, then resolves the stale request before the
current request. The proof only accepts the current vendor after the current
request resolves and keeps the stale vendor absent.

The sweep applies the same principle to nearby churn-ui tests that wait on
async data, route, or user-event state: external router transitions run inside
React `act`, async DOM state is awaited with `findBy*`/`waitFor`, and
Opportunities now performs explicit `afterEach` cleanup so grouped suite runs
do not leak React work into environment teardown.

## Intentional

- This slice is a test-stability and concurrency-proof slice, not the full
  churn-ui ESLint 10 lint-debt cleanup.
- The scan is targeted at async race patterns that have blocked the dependency
  queue; broad style rewrites are intentionally out of scope.

## Deferred

- Churn and Atlas UI lint debt blocks ESLint 10: parked in `HARDENING.md` from
  the previous ESLint 10 batch. That lint cleanup remains a separate medium
  effort slice.

Parked hardening: none added by this slice.

## Verification

- PASS: `npm --prefix atlas-churn-ui test -- --run src/pages/Onboarding.test.tsx`.
- PASS: `for i in $(seq 1 20); do npm --prefix atlas-churn-ui test -- --run src/pages/Onboarding.test.tsx >/tmp/churn-onboarding-after-$i.log 2>&1 || { echo "failed run $i"; cat /tmp/churn-onboarding-after-$i.log; exit 1; }; done; echo "20 focused Onboarding runs passed after sweep"`.
- PASS: `npm --prefix atlas-churn-ui test -- --run src/pages/Onboarding.test.tsx src/pages/Reviews.test.tsx src/pages/Prospects.test.tsx src/pages/Reports.test.tsx src/pages/Watchlists.test.tsx src/pages/Challengers.test.tsx src/pages/Vendors.test.tsx src/pages/Affiliates.test.tsx src/pages/Opportunities.test.tsx src/pages/VendorTargets.test.tsx`.
- PASS: `npm --prefix atlas-churn-ui test`.
- PASS: `npm --prefix atlas-churn-ui run build`.
- Pending before push: `python scripts/sync_pr_plan.py plans/PR-Churn-UI-Async-Race-Sweep.md --check`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-churn-ui/src/pages/Affiliates.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/Challengers.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/Onboarding.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/Opportunities.test.tsx` | 16 |
| `atlas-churn-ui/src/pages/Prospects.test.tsx` | 10 |
| `atlas-churn-ui/src/pages/Reports.test.tsx` | 14 |
| `atlas-churn-ui/src/pages/Reviews.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/VendorTargets.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/Vendors.test.tsx` | 6 |
| `atlas-churn-ui/src/pages/Watchlists.test.tsx` | 6 |
| `plans/PR-Churn-UI-Async-Race-Sweep.md` | 113 |
| **Total** | **195** |
