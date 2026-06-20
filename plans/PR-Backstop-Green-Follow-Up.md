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
2. `tests/conftest.py`: mark tests that request DB-backed fixtures such as
   `db_pool` or `live_pool` as `integration`, keeping pure unit tests in mixed
   files inside the unit lane.
3. `tests/conftest.py`: skip self-pooling live files before module import when
   the marker expression explicitly excludes `integration`; otherwise keep the
   explicit marker allowlist for those files when integration tests are selected.
4. `plans/PR-Backstop-Green-Follow-Up.md`: document this cleanup slice for the
   local PR review contract.

Out of scope: expanding #1712, changing the backstop command, editing every
legacy per-file `asyncpg` fake, or fixing remaining true unit failures before
this harness boundary is validated.

- Reviewer rules triggered: R1.

### Files touched

- `plans/PR-Backstop-Green-Follow-Up.md`
- `tests/conftest.py`

### Review Contract

Acceptance criteria:

- [ ] Pytest collection still succeeds with the collection hook signatures
      pytest expects.
- [ ] Unit backstop collection sees the real `asyncpg` module when requirements
      install it, so legacy per-file setdefault fakes cannot replace it.
- [ ] Tests using shared DB-backed fixtures such as `db_pool` or `live_pool` are
      marked `integration` before marker filtering.
- [ ] Self-pooling live files without DB fixture names are ignored before module
      import when the active marker expression excludes `integration`.
- [ ] The same self-pooling live files are still marked `integration` when they
      are collected for integration-aware runs.
- [ ] Pure unit tests in mixed files such as `tests/test_evidence_gate.py` and
      `tests/test_b2b_phase5b_migration_roundtrip.py` stay in the unit-only
      backstop lane.
- [ ] No production code or backstop command changes.

Affected surfaces: pytest collection, test marker classification, and the
backstop follow-up plan doc.

Risk areas: over-marking a unit test as integration, or changing pytest
collection behavior for unrelated tests. Mitigated by marking at item level for
known DB fixture users and pre-collection skipping only the self-pooling
live-file allowlist when `integration` is explicitly excluded.

Reviewer rules triggered: R1.

## Mechanism

Pytest loads `tests/conftest.py` before collecting test modules. Importing the
real `asyncpg` driver there means older per-file helpers that use
`sys.modules.setdefault("asyncpg", ...)` become no-ops in the CI/backstop
environment where requirements install the driver. For the unit backstop marker
expression, the ignore hook skips the short list of self-pooling live files
before module import, so live-only dependencies cannot fail collection before
marker deselection. For collected tests, the item hook adds the existing
`integration` marker to tests that request known DB-backed fixtures (`db_pool`,
`live_pool`), plus that same short list of self-pooling live files. This keeps
the repo-wide unit backstop focused on unit tests without excluding pure unit
tests that share a file with DB-backed tests.

## Intentional

- Centralize the source fix in the test harness instead of touching every
  historical test module that contains the same `asyncpg` setdefault pattern.
- Keep live/service-backed tests classified out of the unit lane using the
  existing `integration` marker rather than faking service state.
- Mark DB-backed tests at item level where possible so mixed files retain unit
  coverage.
- Skip only self-pooling live files before import when `integration` is
  explicitly excluded from the active marker expression.
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

- Local py_compile for `tests/conftest.py` passed with the bundled Codex Python
  runtime.
- Local inspection of the bundled pytest hookspec confirmed
  `pytest_ignore_collect(collection_path, config)` is the expected signature.
- Local marker simulation confirmed pure Phase 5b parse items remain unmarked
  while `live_pool`, `db_pool`, and self-pooling live items receive
  `integration`.
- Local inspection confirmed `tests/test_evidence_gate.py` DB-backed tests use
  `db_pool` while its pure unit tests do not.
- Local inspection confirmed `tests/test_b2b_phase5b_migration_roundtrip.py`
  uses `live_pool` only for real Postgres tests while the SQL parse test has no
  DB fixture.
- Full backstop validation should run in CI or an environment after
  requirements install, because the local shell Python was unavailable and the
  local Codex runtime did not have project requirements installed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Backstop-Green-Follow-Up.md` | ~130 |
| `tests/conftest.py` | ~50 |
| **Total** | **~180** |
