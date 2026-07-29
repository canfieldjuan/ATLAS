# PR-EOM-Read-Scoping-Fixes

## Why this slice exists

PR #2157 merged with two defects Codex reported two minutes after the
merge (post-merge review round 6), both independently reproduced by the
operator's boundary probes against the deployed runtime:

1. **Bare scoped search omitted legacy rows.** `_scoped_search` was
   first-page-wins: it returned the default tenant's page and queried the
   NULL-context legacy page only when the tenant page was empty. Any
   tenant hit therefore hid claimable legacy customers, contradicting the
   promised `{tenant} ∪ {NULL legacy}` population. Probe result:
   one EOM row + one NULL row supplied, only the EOM row returned, the
   NULL page never queried.
2. **UUID reads had a guard-then-refetch race.** `_guard_contact_id`
   fetched and validated a row, discarded it, and the tool re-fetched by
   id without a tenant predicate. A NULL row claimed by another tenant
   between the awaits passed the guard and serialized as the now-foreign
   row. Probe result: guard fetch NULL, second fetch churnsignals,
   response `found=true` with `returned_context=churnsignals`.

### Problem-derived contract

- Root cause: (1) the scoped-search helper modeled the population as a
  fallback order instead of a union; (2) read tools validated a different
  fetch than the one they returned (time-of-check/time-of-use).
- Correct fix must: query and merge BOTH pages on every default-scoped
  search (truncated to the caller's limit, tenant rows first); make each
  read validate the exact row object it returns (single fetch for
  `get_contact`; post-validate `ctx.contact` for `get_customer_context`),
  and make child reads whose rows carry no tenant column
  (`contact_interactions`) atomic in SQL via a join on the owning
  contact's context.
- Must not change: explicit-argument exact-page semantics, no-default
  legacy behavior, mutation CAS-claim paths (already race-safe via
  `claim_contact`), appointment/call child queries (already atomic in
  SQL from #2157 rounds 3-4).

## Scope (this PR)

Ownership lane: eom-crm/read-scoping
Slice phase: production hardening

1. `atlas_brain/mcp/crm_server.py`: `_scoped_search` merges the tenant
   and NULL-context pages (both queried, tenant first, truncated to
   limit); new `_row_visible()` predicate applied to the same row object
   a tool returns — `get_contact` is single-fetch-validate,
   `get_customer_context` validates `ctx.contact` after the service
   fetch, `_guarded_contact` routes through the same predicate;
   `get_interactions` passes the effective scope to the provider.
2. `atlas_brain/services/crm_provider.py`: `get_interactions` gains an
   optional atomic tenant predicate (join on the owning contact,
   tenant-plus-NULL) so the interaction page cannot outlive the guard.
3. `atlas_brain/services/customer_context.py`: `_gather` threads the
   scope into the interactions query as well (appointments and calls
   already threaded).
4. Proof: `tests/test_crm_read_scoping.py` — the wrong-behavior pin
   (`test_search_scoped_hit_skips_null_fallback`) is REPLACED by
   merge-semantics tests; TOCTOU regression test asserts the validated
   row is the returned row with exactly one fetch.

### Review Contract

- Acceptance criteria:
  1. A default-scoped search with hits on BOTH pages returns tenant rows
     followed by NULL-context rows, truncated to the caller's limit
     (test-asserted call pair + merge order + truncation).
  2. `get_contact` on a UUID performs exactly one provider fetch and
     validates that row object; a sequence NULL-then-foreign can no
     longer serialize the foreign row (test-asserted await count 1).
  3. `get_customer_context` refuses when the service-fetched
     `ctx.contact` is no longer visible under the effective addressing.
  4. `get_interactions` is atomic: the SQL page joins the owning contact
     and applies tenant-plus-NULL in the same statement.
  5. Explicit-argument and no-default behaviors are byte-identical to
     #2157 (existing tests unchanged and green).
- Reachability proof: deployed CRM MCP surface on this box (default
  tenant already live via `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT`);
  post-merge deploy = pull runtime worktree + restart `atlas-api`.
- Affected surfaces: CRM MCP read paths only; mutations untouched.
- Risk areas: double query on every default-scoped search (two indexed
  lookups, accepted in #2157 for the miss case, now paid on every scoped
  search — correctness over one indexed read); pagination semantics of
  the merged search page (limit applies per page then merged-truncated,
  same shape as `list_contacts`).
- Reviewer rules triggered: R1 (#2151/#2157 follow-up), R2 (regression
  tests for both probes), R3 (tenant isolation, fail-closed, TOCTOU
  closed), R4 (no mutation-path changes), R5 (explicit/no-default
  behavior preserved, test-asserted), R8 (race closed atomically in SQL,
  not by re-checking), R10 (one `_row_visible` predicate shared across
  tools), R14.

### Files touched

- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/customer_context.py`
- `plans/PR-EOM-Read-Scoping-Fixes.md`
- `tests/test_crm_read_scoping.py`

## Mechanism

`_scoped_search` becomes a union (the same population definition
`list_contacts` already uses), so every resolution path that rides it —
search, get-by-name, customer-context name/phone/email — inherits the
fix. `_row_visible` is the single visibility predicate for fetched rows;
tools apply it to the object they serialize, which is the only shape that
survives a concurrent claim. Interactions join their owning contact's
context in one statement, so there is no second read to race against.

## Intentional

- **Interactions predicate is tenant-plus-NULL even under an explicit
  argument** — the pre-guard already enforced exact-page for explicit
  callers; rows cannot move tenant→NULL (claims only go NULL→tenant), so
  the NULL branch is unreachable for explicit reads and one SQL shape
  serves both. Documented here rather than split into two statements.
- **`get_contact_appointments` keeps guard + atomic child query** — its
  child rows carry their own NOT-NULL tenant stamp and the query is
  already scoped in SQL (#2157 round 4); the contact row itself is not
  returned, so there is no row-identity to validate.
- **Mutations unchanged** — `update/delete/log` already resolve races via
  the compare-and-set `claim_contact` (#2157 rounds 4-5): a row stolen
  between guard and write fails the CAS and the mutation aborts.

## Deferred

- Tenant-addressable email store (context omits emails under scope until
  then; #2157 round 4).
- `atlas_brain/api/contacts.py` auth/scoping (own slice; UI-coupled).
- `claude-review` commit status for this lane (advisory per
  `docs/REVIEWER_MERGE_GATE.md` until made a required branch-protection
  check; flagged to the operator).

Parked hardening: none new.

## Verification

- Suite `tests/test_crm_read_scoping.py` — 54 passed, including the two
  probe regressions.
- Suites `tests/test_mcp_servers.py` + `tests/test_leads_intake.py` +
  `tests/test_tenant_stamping.py` + `tests/test_customer_context.py` —
  176 passed combined; 6 failures pre-existing on `origin/main`
  (email/Twilio/calendar env-dependent, verified on a pristine worktree
  during #2157).
- Maturity ratchets (mcp + storage lanes, CI flags) — clean.
- Post-merge: pull runtime worktree, restart `atlas-api`, re-run the
  operator's two boundary probes (EOM+NULL search must return both;
  guard/refetch race no longer constructible — single fetch).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/crm_server.py` | 90 |
| `atlas_brain/services/crm_provider.py` | 35 |
| `atlas_brain/services/customer_context.py` | 5 |
| `plans/PR-EOM-Read-Scoping-Fixes.md` | 150 |
| `tests/test_crm_read_scoping.py` | 100 |
| **Total** | **~380** |
