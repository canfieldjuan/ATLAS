# PR-EOM-Portal-Customer-Sync

## Why this slice exists

Issue #2156 slice A2 (owner correction after slice A ran): the calendar
import's 24-month window necessarily contains ended customer relationships
(service stops without CANCELLED markers), so calendar history cannot define
who is a CURRENT customer, and the slice-B watcher would treat lapsed
customers as known-active. The owner's canonical roster is the portal's
server-owned Customer aggregate. Verified on wiw backend main (5062a70):
`GET /api/admin/customers` returns active customers with nested sites
(`locationType` = Residential/Commercial), `atlasContactId` is a first-class
designed cross-system link, and auth is `POST /api/auth/login` -> Bearer
token behind `get_current_admin`.

### Problem-derived contract

- Root cause: slice A let calendar history define `status='active'`; the
  CRM cannot distinguish a current customer from a past one.
- Correct fix must touch/change: a new operator script that authenticates
  to the backend at RUNTIME (getpass or a pre-obtained env token; no
  credential ever stored/printed), fetches the active Customer aggregate,
  resolves each customer to a CRM contact (`atlasContactId` first, then the
  slice-A identity ladder phone -> email -> site addresses), writes through
  the slice-A guarded machinery (diffed, archive/tenant-guarded, source and
  tags only via controlled paths), stamps `metadata.portal_customer_id`
  (the slice-B watcher predicate) via a guarded jsonb merge, and demotes
  previously-imported active "customers" that matched no portal customer to
  `status='inactive'` + `past_customer` tag — provenance-scoped to
  `calendar_import`/`portal_sync` rows only. Dry-run by default with
  `--apply` (demotion-bearing; #2155 convention); non-zero exit on errors.
- Must NOT change: the wiw backend (strictly read-only consumer);
  `scripts/import_eom_customers_live.py` (helpers imported, not modified);
  `atlas_brain/services/crm_provider.py`; `atlas_brain/api/leads.py`;
  schema (no migrations — metadata is existing jsonb); money paths; leads,
  manual, web, and every non-pipeline source (never demoted); no credential
  written to disk, argv, env files, or logs.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. New `scripts/sync_eom_portal_customers.py`: portal -> CRM customer sync
   with demotion, dry-run default, `--apply`, `--base-url`.
2. New `tests/test_sync_eom_portal_customers.py`: 16 behavioural tests on
   the slice-A stub harness.
3. This plan doc.

### Review Contract

- Acceptance criteria:
  1. Every synced contact carries `business_context_id='effingham_maids'`,
     `contact_type='customer'`, `status='active'`, segment tags from active
     sites' `locationType` + `portal` (asserted).
  2. `atlasContactId` resolution wins over the channel ladder and
     short-circuits it (asserted).
  3. Creates strip `source`/`tags` (slice-A race rules) and stamp
     `source='portal_sync'` + tags post-create; matched rows keep their
     recorded provenance (both asserted).
  4. `metadata.portal_customer_id` is stamped through a guarded jsonb
     merge (archive + tenant predicates inside the UPDATE); a rejected
     stamp is a run error (both asserted).
  5. Demotion candidates are provenance-scoped in SQL
     (`contact_type='customer' AND status='active' AND source = ANY
     (calendar_import, portal_sync)`); matched ids are excluded; demoted
     rows get `status='inactive'` + `past_customer` tag through the
     archive/tenant-guarded update (all asserted).
  6. Dry-run performs zero writes while reporting accurate create/update/
     demote previews (matched ids feed the demotion preview; asserted).
  7. Runtime auth only: an env token short-circuits any prompt (asserted);
     the login body contains no file writes and the password is never
     printed (source-asserted).
  8. Non-zero exit when any record errored (reuses slice-A
     `exit_code_for` semantics via the errors counter).
  9. Demotion is SKIPPED entirely when any sync error occurred -- a
     partial match set never drives demotions (source-asserted; Codex A2
     round 1, BLOCKER).
  10. An `atlasContactId` link to a NULL-context legacy row is claimed via
      the CAS (the id link is the identity); a link to a FOREIGN tenant is
      ignored (reported) and resolution falls to the ladder (both
      asserted; Codex A2 round 1, BLOCKER).
  14. Resolution checks the STAMPED portal id first (the stable key after
      a create -- channel drift can never duplicate a synced customer);
      managed tags (portal/segments/past_customer) are REPLACED on portal
      matches while foreign tags survive (an active match sheds
      past_customer); dry-run computes the real diff + stamp need so clean
      re-runs preview zero updates; the config surface is the single typed
      ATLAS_TOOLS_* name; nameless records cannot crash the run loop (all
      asserted; Codex A2 round 4).
  13. Portal-id and name validation precede ANY write; nameless active
      records are run errors (gating demotion), and the already-stamped
      fallback carries the same archive/tenant guards as the stamp itself
      (all asserted; Codex A2 round 3).
  12. An empty (or all-inactive) portal roster fails closed -- it would
      demote the entire base (asserted); inactive customers are belt-
      filtered client-side; portal emails normalize to provider casing
      before diffing; a missing portal id is a run error, not a silent
      predicate gap; clean re-runs neither rewrite the stamp nor error
      (IS DISTINCT FROM guard + already-stamped check); and the demotion
      UPDATE itself re-checks tenant/type/active/provenance (all
      asserted; Codex A2 round 2).
  11. The portal token/base-url resolve through typed
      `ATLAS_TOOLS_EOM_PORTAL_*` settings with process-env override
      (live-parse verified; Codex A2 round 1, R11).
- Reachability proof: operator script, invoked directly; the fetch/auth
  seam is exercised at the client boundary (stub client), and the
  write paths run the same slice-A machinery already live-proven by the
  #2158 production run.
- Reviewer rules triggered: R1, R2, R4, R6, R8, R11, R12, R14.
  - R1: behavior implements the #2156 A2 owner correction (portal defines
    active; calendar is enrichment).
  - R2: every criterion has a named test; the slice-A machinery reused
    here was live-verified in production by #2158.
  - R4: data safety — guarded writes, provenance-scoped demotion,
    fail-closed claim identity, no credential persistence.
  - R6: errors surface and fail the run; auth/fetch failures exit
    non-zero via SystemExit.
  - R8: idempotency — diffed no-op-free updates, stable portal-id stamp,
    re-runs converge (matched set stable).
  - R11: dependencies & config -- two optional typed
    `ATLAS_TOOLS_EOM_PORTAL_*` fields on `ToolsConfig`, default None;
    absent config changes no behavior.
  - R12: env/config — only `EOM_PORTAL_TOKEN` (optional, never written)
    and `--base-url`; no secrets in the repo.
  - R14: this contract is the reviewer checklist.

### Files touched

- `atlas_brain/config.py`
- `plans/PR-EOM-Portal-Customer-Sync.md`
- `scripts/sync_eom_portal_customers.py`
- `tests/maturity_sweep/baseline_scripts.json`
- `tests/test_sync_eom_portal_customers.py`

## Mechanism

`portal_login` (env token else getpass -> `POST /api/auth/login`) ->
`fetch_portal_customers` (`GET /api/admin/customers`, active-only default)
-> per customer: `customer_to_contact_data` (stamped payload; segment tags
from active sites) -> `resolve_contact` (`atlasContactId` by id with
archived guard, else slice-A ladder reusing `_phone_digits`,
`_search_channel`, `resolve_by_address`, with `claim_legacy_row` CAS for
NULL-context rows) -> writes via `_update_matched` / create-then-stamp
(`_guarded_update`) -> `stamp_portal_id` guarded jsonb merge ->
`demote_unmatched` over the provenance-scoped candidate set minus this
run's matched ids.

## Intentional

- The backend is strictly read-only here; the `atlasContactId` write-back
  (PATCH to the portal so the link becomes bidirectional) is DEFERRED —
  it mutates the ops system and deserves its own slice.
- Demotion never touches leads or non-pipeline sources: only rows this
  pipeline itself claimed as active customers are eligible.
- `metadata.portal_customer_id` (not a new column) keeps the slice
  migration-free; the slice-B watcher predicate reads it.
- Credentials: getpass at runtime or a pre-obtained env token; the
  password variable is function-local and never persisted or printed.

## Deferred

- Portal write-back of `atlasContactId` (own slice; mutates the backend).
- The #2156 slice B watcher (this provides its predicate).
- Operator run post-merge: dry-run first, review the demotion preview,
  then `--apply`.

## Verification

- `tests/test_sync_eom_portal_customers.py` — 26 passed.
- Maturity note: the scripts-lane ratchet baseline gains ONLY this PR's
  new script (deliberate per-record operator patterns recorded); the
  pre-existing unbaselined `import_eom_customers_live.py` (main's code,
  untouched here) stays advisory-red and is tracked with #2159.
- `tests/test_eom_live_calendar_import.py` — 55 passed (adjacent; the
  imported slice-A machinery is unmodified).
- `python -m py_compile` on the script — clean.
- NOT run pre-merge: the live sync (operator-run; needs the portal admin
  password at the prompt; dry-run first by design).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 10 |
| `plans/PR-EOM-Portal-Customer-Sync.md` | 200 |
| `scripts/sync_eom_portal_customers.py` | 359 |
| `tests/maturity_sweep/baseline_scripts.json` | 8 |
| `tests/test_sync_eom_portal_customers.py` | 252 |
| **Total** | **829** |
