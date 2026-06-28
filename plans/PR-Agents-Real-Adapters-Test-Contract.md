# PR-Agents-Real-Adapters-Test-Contract

## Why this slice exists

Two consecutive deflection billing-entitlement slices exposed the same process
failure: tests used fake pools/query-string assertions that re-spelled the SQL
boundary instead of exercising the real adapter behavior, and fail-closed
billing defects reached review behind green local tests. The operator explicitly
called out the pattern with "USE THE REAL DB ADAPTERS."

Root cause: AGENTS.md still described integration tests as "services + ports
with fakes" and did not state the boundary rule clearly enough. Builders could
mock the component under test instead of only mocking true external transports.

This PR fixes the process root by making the real-adapter default explicit in
the repo's builder contract, including examples of allowed mocks, forbidden
component fakes, generated-fixture drift, and the required deferral language
when a real adapter is genuinely too expensive for one slice.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Update the test guidance in AGENTS.md so integration tests exercise the real
   service/adapter and fake only true external boundaries.
2. Add a new 3e.1 subsection with the concrete real-adapter rule and examples
   tied to recent failure modes.

### Review Contract

- Acceptance criteria:
  - [ ] The integration-test bullet no longer implies fake ports are the
        default.
  - [ ] The new rule distinguishes mock-allowed external seams from forbidden
        component-under-test fakes.
  - [ ] The rule covers SQL filters, validators/projections, generated fixtures,
        and shared validator definitions.
  - [ ] The rule states what to do when real-adapter coverage is deferred.
- Affected surfaces: builder/reviewer process documentation only.
- Risk areas: docs clarity and avoiding accidental overreach into product code.
- Reviewer rules triggered: R1, R10, R14.
- boundary-probe: read the diff against the two known defect classes: fake DB
  pools hiding SQL fail-closed behavior and re-spelled validators diverging from
  real consumers.

### Files touched

- `AGENTS.md`
- `plans/PR-Agents-Real-Adapters-Test-Contract.md`

## Mechanism

AGENTS.md's test section changes the integration-test definition from
"services + ports with fakes" to "real service + adapter; fake only true
external boundaries." A new 3e.1 subsection then spells out the rule in builder
terms:

- mock third-party/network/transport/time/randomness seams only;
- never fake the component whose behavior the test is meant to prove;
- derive generated/producer-shaped fixtures instead of hand-authoring copies;
- share one validator/projection definition across same-shaped checks;
- if real adapter coverage is deferred, name why and track the replacing slice.

## Intentional

- This is docs/process only. It does not retrofit existing test suites in this
  PR; the purpose is to put the rule on main before the next builder session.
- The wording allows fakes for true external services such as Stripe/Resend
  transports. The target is over-mocking ATLAS-owned adapters and validators,
  not banning all test doubles.

## Deferred

- Existing fake-pool tests are not swept here. Future product slices should
  replace them when they touch the guarded boundary or when review exposes a
  concrete defect class.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-Agents-Real-Adapters-Test-Contract.md --check` - passed.
- `bash scripts/push_pr.sh /tmp/agents-real-adapters-pr-body.md -u origin HEAD` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 39 |
| `plans/PR-Agents-Real-Adapters-Test-Contract.md` | 92 |
| **Total** | **131** |
