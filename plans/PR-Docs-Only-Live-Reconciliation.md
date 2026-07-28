# PR-Docs-Only-Live-Reconciliation

## Why this slice exists

#2247 was opened as an explicit `Docs-only: true` plan-archive cleanup after
#2240 and #2245 merged. It immediately failed the required `live-reconciliation`
check even though the diff is Markdown-only and there are no unresolved Codex
connector threads. That recreates the same "red immediately" workflow friction
this lane is supposed to remove.

### Problem-derived contract

- Root cause: `scripts/check_ai_reconciliation_live.py` requires current-head
  Codex review attestation for every PR before evaluating whether there is any
  review work to reconcile. Explicit docs-only PRs already pass the PR body
  contract without a plan, and a pure Markdown archive cleanup may have no Codex
  review surface before the required check runs.
- Correct fix must touch/change: teach the live reconciliation evaluator to
  recognize the existing `Docs-only: true` body marker and bypass only the
  current-head review-attestation requirement when there are zero unresolved
  Codex threads; add focused tests proving docs-only/no-thread passes while
  docs-only/open-thread and non-doc/no-review still fail.
- Must not change: unresolved Codex threads must remain blocking, current-head
  `CHANGES_REQUESTED` must remain blocking, non-doc PRs must still require
  current-head Codex review or clean-review-comment attestation, and product
  behavior must remain untouched.

## Scope (this PR)

Ownership lane: workflow/docs-only-live-reconciliation
Slice phase: Workflow/process

1. Add a docs-only/no-open-Codex-thread bypass for live reconciliation review
   attestation.
2. Keep the merged #2240/#2245 plan archive cleanup in this branch because it
   is the concrete docs-only case that exposed the missing bypass.
3. Add focused live-reconciliation tests for the new pass and fail branches.

### Review Contract

- Acceptance criteria:
  - `scripts/check_ai_reconciliation_live.py` returns success for a body whose
    first nonblank line is `Docs-only: true` when there are no unresolved Codex
    connector threads and no current-head `CHANGES_REQUESTED` review.
  - The same docs-only body still fails when an unresolved Codex connector
    thread exists.
  - A non-doc PR body with no current-head Codex review/clean-comment
    attestation still fails.
  - The archive cleanup moves only
    `PR-Codex-Review-Scope-Reset.md` and
    `PR-Codex-Review-Thread-Event-Canary.md` into `plans/archive/` and refreshes
    `plans/INDEX.md`.
- Reachability proof: `python -m pytest tests/test_check_ai_reconciliation_live.py -q`
  exercises the real live-check evaluator entrypoint in dry-run mode and asserts
  the observable exit-code/messages for docs-only and non-doc cases.
- Affected surfaces: `scripts/check_ai_reconciliation_live.py`,
  `tests/test_check_ai_reconciliation_live.py`, and merged plan archive
  bookkeeping under `plans/`.
- Risk areas: over-broad docs-only detection, accidentally clearing open Codex
  threads, accidentally clearing `CHANGES_REQUESTED`, and weakening non-doc PR
  review freshness.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_ai_reconciliation_live.py` review
  attestation gate.
- Replaced-path behaviors: current behavior requires review attestation for all
  PRs; this slice replaces that with "review attestation required unless the PR
  body is explicitly docs-only and there are no open Codex threads."
- Guard-relevant fields: PR body first nonblank line, unresolved Codex thread
  set, current-head Codex review states, and clean-review comments.
- Caller x input shape: GitHub Actions calls the script with live `--pr`;
  tests call `evaluate()` and `main()` with JSON fixture files.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: default bot identities remain
  `chatgpt-codex-connector` and `chatgpt-codex-connector[bot]`; no env/config
  default changes.
- Explicit value probe: tests use the default exact Codex identity set and an
  explicit `Docs-only: true` marker.
- Absent value probe: tests keep a non-doc body without review attestation
  failing.
- Default-session/default-context probe: local dry-run tests exercise the same
  default evaluator path as the workflow script after fetched inputs are loaded.
- Side-effect ordering: unresolved threads and `CHANGES_REQUESTED` reviews are
  evaluated before the docs-only attestation bypass can pass.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Docs-Only-Live-Reconciliation.md`
- `plans/archive/PR-Codex-Review-Scope-Reset.md`
- `plans/archive/PR-Codex-Review-Thread-Event-Canary.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

The evaluator parses the PR body for the existing docs-only marker using the
same strict placement rule as the PR body contract: the first nonblank line must
be exactly `Docs-only: true`. `evaluate()` still computes current-head change
requests and unresolved Codex threads first. If neither blocking condition
exists and the body is docs-only, it returns success without requiring a Codex
review attestation. Otherwise the existing current-head review/clean-comment
attestation requirement remains unchanged.

## Intentional

- Do not exempt arbitrary Markdown diffs; only the explicit body marker gets the
  bypass.
- Do not skip live reconciliation entirely for docs-only PRs; open Codex threads
  and current-head `CHANGES_REQUESTED` still fail.
- Keep the two plan archive moves in this PR because #2247 is the live docs-only
  failure that proves the need for this narrow gate change.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` - 46 passed.
- `python -m py_compile scripts/check_ai_reconciliation_live.py` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main --check` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Docs-Only-Live-Reconciliation.md` - passed.
- `python scripts/audit_plan_doc_files_touched.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed.
- `python scripts/audit_plan_doc_diff_size.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed, estimate 237 actual 237.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Docs-Only-Live-Reconciliation.md` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/archive-review-workflow-plans-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 4 |
| `plans/PR-Docs-Only-Live-Reconciliation.md` | 153 |
| `plans/archive/PR-Codex-Review-Scope-Reset.md` | 0 |
| `plans/archive/PR-Codex-Review-Thread-Event-Canary.md` | 0 |
| `scripts/check_ai_reconciliation_live.py` | 20 |
| `tests/test_check_ai_reconciliation_live.py` | 60 |
| **Total** | **237** |
