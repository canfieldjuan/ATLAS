# PR-EOM-Portal-Reconciliation

## Why this slice exists

Issue canfieldjuan/eom-timetracker#20 identifies Atlas as the source for
customer service economics, while the EOM portal owns operational Sites. PR
canfieldjuan/eom-timetracker#52 added guarded Customer and Site updates to the
portal. This slice proves the first safe Atlas-to-portal write without
inventing a second customer-matching system.

Diff-budget override: the safety-critical operator command, its real-entrypoint
external-seam tests, and the required contract plan are one indivisible proof;
splitting them would publish either an unproved write path or tests with no
reachable implementation.

### Problem-derived contract

- Root cause: Atlas currently imports portal customers into its CRM but has no
  reverse path for service economics. A Contact can represent several portal
  Sites while Atlas stores one Contact address and no durable Site identity,
  so name, address, phone, email, or calendar matching cannot safely choose a
  Site.
- Correct fix must touch/change: add a private operator entrypoint that accepts
  one exact Atlas service ID and one exact existing portal Site ID; tenant-scope
  the Atlas read to `effingham_maids`; verify the linked Contact's stamped
  portal Customer ID owns that Site; map only supported service rate labels;
  emit a deterministic preview and plan hash; and require that exact hash plus
  the current portal update token before applying only `rate` and `rateType`.
  Apply must bind approval to the portal origin and hold shared locks on the
  selected Atlas service and Contact until the one portal write finishes, so
  neither authoritative source nor target can drift outside the approved plan.
  Focused tests must prove the real CLI path, zero-write preview/no-op behavior,
  fail-closed identity and mapping boundaries, hash drift protection, guarded
  apply payload, and surfaced stale/HTTP failures without retry.
- Must not change: existing portal-to-Atlas synchronization, CRM/storage
  schemas, repositories, config, invoices, contacts, billing recipients,
  addresses, calendar import, schedules, public APIs, MCP tools, or UI. The
  command must not infer identity, create, merge, archive, reassign, clear,
  retry, or update a portal Customer.

## Scope (this PR)

Ownership lane: eom/portal-reconciliation
Slice phase: Vertical slice

1. Reconcile one explicitly selected Atlas service's economics to one
   explicitly selected existing EOM portal Site.
2. Prove real command-entrypoint reachability, preview/hash/apply concurrency,
   and all fail-closed identity boundaries.

### Review Contract

- Acceptance criteria:
  - Preview prints a canonical plan and deterministic SHA-256 hash and never
    PATCHes the portal.
  - Apply requires the matching hash recomputed from fresh Atlas and portal
    state, binds it to the portal origin, locks the Atlas source rows through
    the write, and sends only `expectedUpdateToken`, `rate`, and `rateType`.
  - Exact tenant, active service/contact, stamped Customer identity, Site
    ownership, and supported rate-label checks all fail closed.
  - An unchanged target is a no-op; stale or other HTTP failures are reported
    nonzero and are not retried.
- Reachability proof: invoke `scripts/reconcile_eom_portal_site.py` through its
  CLI `main()` and observe preview JSON/hash or one guarded Site PATCH.
- Affected surfaces: one private reconciliation script, its focused tests, and
  required script maturity enrollment if the repository gate identifies it.
- Risk areas: cross-tenant reads, targeting a Site owned by another Customer,
  Atlas or portal drift after approval, cross-origin hash replay, accidental
  broad payloads, secret leakage, retries after a concurrency conflict, and
  unsupported rate-label coercion.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

### Files touched

- `plans/PR-EOM-Portal-Reconciliation.md`
- `scripts/reconcile_eom_portal_site.py`
- `tests/test_reconcile_eom_portal_site.py`

## Mechanism

Reuse the existing typed portal configuration, login, and active-customer fetch
helpers. Read the selected service and linked Contact in one explicitly
tenant-scoped query. Locate the selected Site only inside the fetched active
portal roster, verify its `customerId`, translate the three exact supported
rate labels, and compare the desired values with the Site.

The canonical preview contains the stable identities, portal origin and current
token, current economics, and desired economics. Preview prints it with its
SHA-256 hash. Apply performs the same fresh derivation while holding shared
locks on the selected Atlas service and Contact, compares the supplied hash
before mutation, then sends one guarded PATCH before releasing those locks.
There is only one possible remote mutation, so a failed apply cannot leave a
multi-entity partial write.

## Intentional

- Exact IDs are an intentional operator boundary until Atlas has a durable
  service-to-Site binding.
- Contact address, names, phones, emails, and calendar labels are deliberately
  rejected as identity signals.
- A separate reverse-direction command keeps the existing portal-to-Atlas sync
  behavior unchanged while reusing its transport primitives.

## Deferred

- Durable service-to-Site binding/bootstrap for multi-site customers.
- Review-only customer primary/billing candidate previews.
- Address and calendar-resolution workflows after stable per-Site identity
  exists.

Parked hardening: none.

## Verification

- python -m pytest tests/test_sync_eom_portal_customers.py
  tests/test_reconcile_eom_portal_site.py -q
  - passed, 70 tests.
- python -m ruff check scripts/reconcile_eom_portal_site.py
  tests/test_reconcile_eom_portal_site.py
  - passed.
- python -m py_compile scripts/reconcile_eom_portal_site.py
  tests/test_reconcile_eom_portal_site.py and
  python scripts/reconcile_eom_portal_site.py --help - passed.
- python scripts/maturity_sweep.py scripts --tests-root tests --baseline
  tests/maturity_sweep/baseline_scripts.json --top 25
  - passed with no new brittleness above baseline.
- Independent correctness review - LGTM after source-lock, origin-binding, and
  strict external-confirmation boundary probes were fixed.
- Cold diff audit - no gaps: the scoped/locked source read is at
  `scripts/reconcile_eom_portal_site.py:39`, identity and plan construction at
  `scripts/reconcile_eom_portal_site.py:107`, the sole guarded PATCH at
  `scripts/reconcile_eom_portal_site.py:186`, and real-entrypoint/boundary
  proofs at `tests/test_reconcile_eom_portal_site.py:146` and
  `tests/test_reconcile_eom_portal_site.py:189`. No existing runtime module or
  excluded behavior changed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Portal-Reconciliation.md` | 144 |
| `scripts/reconcile_eom_portal_site.py` | 310 |
| `tests/test_reconcile_eom_portal_site.py` | 344 |
| **Total** | **798** |
