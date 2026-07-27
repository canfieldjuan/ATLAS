# PR-EOM-Read-Scoping-Test-Evidence

## Why this slice exists

The operator's review of #2165 confirmed the fixes but graded two of the
regressions as structural evidence: the `get_customer_context`
post-validation and the atomic interaction query were pinned by source-text
search (`_row_visible(ctx.contact)` substring, SQL fragments), not by
executing the behavior. This slice upgrades both to behavioral tests.

### Problem-derived contract

- Root cause: the two aggregate paths were hard to execute in unit tests
  (service constructed inside the tool; SQL built inside the provider), so
  #2165 pinned their source instead of their behavior.
- Correct fix must: execute the TOCTOU refusal end-to-end through the MCP
  tool with a stub service whose fetch returns a foreign row, and execute
  the provider's scoped interaction query against a capturing pool,
  asserting the join + bound parameters land in one statement.
- Must not change: any production code (test-only slice), the existing
  structural pins (kept as belt).

## Scope (this PR)

Ownership lane: eom-crm/read-scoping
Slice phase: robust testing

1. `tests/test_crm_read_scoping.py`: behavioral TOCTOU pair (guard sees a
   claimable NULL row, service fetch returns the claimed-foreign row →
   refusal; still-visible row → serialized with scope flags) and a
   behavioral provider test executing `get_interactions` against a
   capturing fake pool (scoped: join + `(contact, tenant, limit)` params
   in one statement; unscoped: legacy statement).
2. `tests/maturity_sweep/baseline_atlas_brain_storage.json`: recorded
   intentional +1 INTERNAL_MOCK on `atlas_brain/storage/database.py` (the fake pool
   monkeypatches `get_db_pool`, the provider's only seam), plus ratchet
   DECREASES the sweep re-recorded from earlier test additions.

### Review Contract

- Acceptance criteria:
  1. The TOCTOU test constructs the actual race (different rows on the
     two fetches) through the real MCP tool code path and asserts
     refusal; its positive twin proves the same wiring serializes a
     visible row.
  2. The provider test asserts tenant + limit are bound parameters of
     one statement containing the contact join, and that unscoped calls
     keep the legacy SQL.
  3. No production file changes.
- Reachability proof: pytest suite (CI + local); no runtime deploy needed.
- Affected surfaces: tests + one sweep baseline.
- Risk areas: baseline acceptance grows database.py's recorded
  INTERNAL_MOCK 31→32 (documented here; the alternative was leaving the
  provider path structurally pinned).
- Reviewer rules triggered: R1 (#2165 review follow-up), R2 (this IS the
  test slice), R5 (no behavior change), R14.

### Files touched

- `plans/PR-EOM-Read-Scoping-Test-Evidence.md`
- `tests/test_crm_read_scoping.py`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`

## Mechanism

The stub service returns a minimal `CustomerContext` stand-in whose
`contact` is the foreign row, so the tool's post-validation runs against
exactly the artifact the race produces. The capturing pool records the SQL
and parameters the provider actually executes, which is the strongest
DB-free evidence that scoping and pagination share one statement.

## Intentional

- **Structural pins kept** — cheap belt under the new behavioral tests.
- **Baseline acceptance over a new production seam** — injecting a pool
  seam into the provider just for tests would be production churn on a
  hot path; one recorded INTERNAL_MOCK on a file already carrying 31 is
  the smaller cost, and the sweep re-record also captured several ratchet
  decreases.

## Deferred

- True database-integration tests (live Postgres) for the scoped queries —
  no DB harness exists in this suite today.

Parked hardening: none new.

## Verification

- `tests/test_crm_read_scoping.py` — 57 passed (3 new behavioral).
- Adjacent suites — 179 passed combined; 6 pre-existing env failures.
- Maturity ratchets (mcp + storage, CI flags) — clean after the recorded
  acceptance.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Read-Scoping-Test-Evidence.md` | 100 |
| `tests/test_crm_read_scoping.py` | 85 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 18 |
| **Total** | **~205** |
