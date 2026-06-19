# PR: Backstop green follow-up

## Why this slice exists

#1712 adds the advisory repo-wide unit backstop/auditor change, but the first
backstop runs surfaced two harness-boundary problems that should be fixed in a
follow-up instead of widening #1712: legacy import-time `asyncpg` fakes can
poison later DB fixture tests, and live/service-backed DB tests need to sit
outside the unit-only backstop lane without dropping mixed-file unit coverage.

## Scope (this PR)

Ownership lane: testing/backstop-hygiene
Slice phase: Robust testing

1. `tests/conftest.py`: preload the real `asyncpg` module and
   `asyncpg.exceptions` before test module collection when the dependency is
   installed, so legacy `sys.modules.setdefault("asyncpg", MagicMock())` helpers
   cannot replace the real driver.
2. `tests/conftest.py`: mark tests that request the shared `db_pool` fixture as
   `integration`, keeping pure unit tests in mixed files inside the unit lane.
3. `tests/conftest.py`: keep a smaller explicit allowlist only for live files
   that create their own pools and do not request `db_pool`.
4. `plans/PR-Backstop-Green-Follow-Up.md`: document this cleanup slice for the
   local PR review contract.

Out of scope: expanding #1712, changing the backstop command, or fixing
remaining true unit failures before this harness boundary is validated.

- Reviewer rules triggered: R1.

### Files touched

- `plans/PR-Backstop-Green-Follow-Up.md`
- `tests/conftest.py`

### Review Contract

Acceptance criteria:

- [ ] Pytest collection still succeeds with the collection hook signature pytest
      expects.
- [ ] Unit backstop collection sees the real `asyncpg` module when requirements
      install it, so legacy per-file setdefault fakes cannot replace it.
- [ ] Tests using the shared `db_pool` fixture are marked `integration` before
      marker filtering.
- [ ] Self-pooling live files without `db_pool` remain excluded through the
      explicit filename allowlist.
- [ ] Pure unit tests in mixed files such as `tests/test_evidence_gate.py` stay
      in the unit-only backstop lane.
- [ ] No production code or backstop command changes.

Affected surfaces: pytest collection, test marker classification, and the
backstop follow-up plan doc.

Risk areas: over-marking a unit test as integration, or changing pytest
collection behavior for unrelated tests. Mitigated by marking at item level for
`db_pool` users and keeping the file allowlist only for self-pooling live files.

Reviewer rules triggered: R1.

## Mechanism

Pytest loads `tests/conftest.py` before collecting test modules. Importing the
real `asyncpg` driver there means older per-file helpers that use
`sys.modules.setdefault("asyncpg", ...)` become no-ops in the CI/backstop
environment where requirements install the driver. The collection hook then
adds the existing `integration` marker to tests that request `db_pool`, plus a
short list of self-pooling live files that create their own pools and do not use
that fixture. This keeps the repo-wide unit backstop focused on unit tests
without excluding pure unit tests that share a file with DB-backed tests.

## Intentional

- Centralize the source fix in the test harness instead of touching every
  historical test module that contains the same `asyncpg` setdefault pattern.
- Keep live/service-backed tests classified out of the unit lane using the
  existing `integration` marker rather than faking service state.
- Mark DB-backed tests at item level where possible so mixed files retain unit
  coverage.
- Leave #1712 focused on the advisory backstop/auditor workflow.

## Deferred

- Physically replacing every legacy per-file `asyncpg` setdefault fake can be a
  later cleanup if reviewers want that churn after this smaller harness fix.
- Slice 3 residual unit failures should be triaged after the backstop is rerun
  with this boundary fix in place.

## Parked hardening

Security Guardrails `startup_failure` remains parked as workflow-level hardening
because that workflow is identical to `main` and separate from this backstop
cleanup slice.

## Verification

- Local py_compile for tests/conftest.py and tests/test_mcp_servers.py passed.
- Local inspection confirmed `tests/test_evidence_gate.py` DB-backed tests use
  `db_pool` while its pure unit tests do not.
- Full backstop validation should run in CI or an environment after
  requirements install, because the local Codex runtime did not have `asyncpg`
  installed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Backstop-Green-Follow-Up.md` | ~110 |
| `tests/conftest.py` | ~35 |
| **Total** | **~145** |
