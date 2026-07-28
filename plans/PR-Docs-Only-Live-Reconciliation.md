# PR-Docs-Only-Live-Reconciliation

## Why this slice exists

#2247 was opened as an explicit `Docs-only: true` plan-archive cleanup after
#2240 and #2245 merged. It immediately failed the required `live-reconciliation`
check before Codex had reviewed the current head. The first implementation
fixed that symptom by trusting the PR body marker alone, and Codex correctly
flagged that as a bypass: a non-Markdown PR could lead with `Docs-only: true`
and skip current-head review attestation when no threads were open.
The follow-up fix then proved filenames but not Git blob type, leaving the same
class open for Markdown-named symlinks or gitlinks. The diff may exceed the 400
LOC target because both bypasses live in the same required merge gate and must be
fixed together before #2247 can safely merge.
The next Codex rounds found additional proof gaps in that same gate: incomplete
GitHub pull-file listings, an untested docs-only `CHANGES_REQUESTED` veto path,
review/thread/body races around file proof, deleted/renamed source proof using
base-tip instead of merge-base trees, malformed changed-file status admission,
non-doc PRs paying the docs-only file-proof cost, and installed watcher copies
missing the canonical body-classifier dependency. The current Codex round found
one more race in the same R8 class: the file proof recorded the head revision
but not the base revision and changed-file count used to derive the merge-base
tree proof, so a retargeted PR could mix an older docs-only file proof with a
newer final PR snapshot.

### Problem-derived contract

- Root cause: `scripts/check_ai_reconciliation_live.py` had no trusted
  changed-file proof for the docs-only attestation bypass. Filename suffix alone
  is also insufficient because the canonical admission classifier rejects
  Markdown-named symlinks/gitlinks by inspecting Git object mode/type.
- Correct fix must touch/change: require both the `Docs-only: true` body marker
  and a non-empty Markdown-only GitHub changed-file list whose base/head entries
  are non-executable regular blobs before bypassing current-head review
  attestation; prove the fetched file list is complete, revalidate review/thread
  state after file proof, read deleted/renamed sources from the merge-base tree,
  reuse the canonical docs-only body marker parser, validate changed-file
  status/source fields, fetch file proof only when the docs-only exemption is the
  live decision path, package canonical parser dependencies into installed
  watchers, bind the docs-only file proof to the same base/head/changed-file
  generation that is compared after proof collection, and test
  docs-only/no-thread/Markdown-only pass plus non-Markdown, symlink/gitlink-shaped,
  malformed status, docs-only/open-thread, docs-only/`CHANGES_REQUESTED`,
  trailing body/base/count races, and non-doc/no-review fail.
- Must not change: unresolved Codex threads must remain blocking, current-head
  `CHANGES_REQUESTED` must remain blocking, non-doc PRs must still require
  current-head Codex review or clean-review-comment attestation, and product
  behavior must remain untouched.

## Scope (this PR)

Ownership lane: workflow/docs-only-live-reconciliation
Slice phase: Workflow/process

1. Add a docs-only/no-open-Codex-thread/Markdown-only bypass for live
   reconciliation review attestation.
2. Keep the merged #2240/#2245 plan archive cleanup in this branch because it
   is the concrete docs-only case that exposed the missing bypass.
3. Add focused live-reconciliation tests for the new pass and fail branches.

### Review Contract

- Acceptance criteria:
  - `scripts/check_ai_reconciliation_live.py` returns success for a body whose
    first nonblank line is `Docs-only: true` only when there are no unresolved
    Codex connector threads, no current-head `CHANGES_REQUESTED` review, and the
    live changed-file list is non-empty Markdown-only regular blobs.
  - The same docs-only body still fails for a non-Markdown or non-regular-blob
    changed-file list unless current-head Codex review/clean-comment attestation
    is present.
  - The same docs-only body still fails when an unresolved Codex connector
    thread exists.
  - The same docs-only body still fails when current-head Codex has requested
    changes, even if the changed-file proof is otherwise valid.
  - Incomplete GitHub pull-file listings and review/thread/body movement during
    live proof collection fail closed instead of granting the bypass.
  - Deleted or renamed Markdown paths are proved against the PR merge-base tree,
    not the mutable base branch tip.
  - The docs-only file proof fails closed if the PR base SHA, head SHA, or
    changed-file count changes before final evaluation.
  - Missing/unknown changed-file statuses and renamed rows without a source path
    fail closed.
  - Non-doc PRs and PRs that already have current-head Codex attestation do not
    fetch docs-only tree/file proof.
  - Installed PR watcher copies include the canonical docs-only body classifier
    dependencies required by the live checker.
  - A non-doc PR body with no current-head Codex review/clean-comment
    attestation still fails.
  - The archive cleanup moves only
    `PR-Codex-Review-Scope-Reset.md` and
    `PR-Codex-Review-Thread-Event-Canary.md` into `plans/archive/` and refreshes
    `plans/INDEX.md`.
- Reachability proof: `python -m pytest tests/test_check_ai_reconciliation_live.py -q`
  exercises the real live-check evaluator entrypoint in dry-run mode and asserts
  observable exit-code/messages for docs-only, non-doc, incomplete-file-listing,
  and race cases.
- Affected surfaces: `scripts/check_ai_reconciliation_live.py`,
  `scripts/install_codex_wake_bridge.py`,
  `tests/test_check_ai_reconciliation_live.py`, `tests/test_pr_watcher.py`, and
  merged plan archive bookkeeping under `plans/`.
- Risk areas: over-broad docs-only detection, trusting PR body text or filename
  suffixes without blob proof, accidentally clearing open Codex threads,
  accidentally clearing `CHANGES_REQUESTED`, and weakening non-doc PR review
  freshness.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/check_ai_reconciliation_live.py` review
  attestation gate.
- Replaced-path behaviors: current behavior requires review attestation for all
  PRs; this slice replaces that with "review attestation required unless the PR
  body is explicitly docs-only, the PR changed-file list is Markdown-only
  regular blobs, and there are no open Codex threads."
- Guard-relevant fields: PR body first nonblank line, GitHub PR changed-file
  count/status/filenames/previous filenames, merge-base/head tree `mode` and `type`,
  proof base SHA/head SHA/merge-base SHA/changed-file count, unresolved Codex
  thread set, current-head Codex review states, and clean-review comments.
- Caller x input shape: GitHub Actions calls the script with live `--pr`;
  tests call `evaluate()` and `main()` with JSON fixture files, including
  changed-file fixtures.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: default bot identities remain
  `chatgpt-codex-connector` and `chatgpt-codex-connector[bot]`; no env/config
  default changes.
- Explicit value probe: tests use the default exact Codex identity set, an
  explicit `Docs-only: true` marker, and Markdown-only regular-blob fixtures.
- Absent value probe: tests keep a non-doc body without review attestation
  failing.
- Default-session/default-context probe: local dry-run tests exercise the same
  default evaluator path as the workflow script after fetched inputs are loaded.
- Side-effect ordering: unresolved threads and `CHANGES_REQUESTED` reviews are
  evaluated before the docs-only bypass can pass, and the review/thread snapshot
  is compared again after body/file proof collection.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Docs-Only-Live-Reconciliation.md`
- `plans/archive/PR-Codex-Review-Scope-Reset.md`
- `plans/archive/PR-Codex-Review-Thread-Event-Canary.md`
- `scripts/check_ai_reconciliation_live.py`
- `scripts/install_codex_wake_bridge.py`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_pr_watcher.py`

## Mechanism

The evaluator parses the PR body for the existing docs-only marker by importing
the same classifier used by the PR-body contract. In live mode, the script first
collects the normal review/thread attestation state. It fetches the PR's
changed-file list and merge-base/head Git trees from GitHub only if docs-only is
the remaining exemption candidate. The exemption is valid only when every
current filename and rename source has Markdown as its sole suffix, every row has
an admitted status with required source/head fields, the relevant tree entry is
`100644 blob`, and the number of fetched file rows equals GitHub's PR
changed-file count. The file-proof object records the PR base SHA, head SHA,
merge-base SHA, expected changed-file count, and enriched file rows. Live mode
sandwiches file proof between body reads and
review/thread snapshots, including a final body read after the final snapshot,
and fails closed if the body, base/head refs, changed-file count, review
generation, or thread generation moves. Installed watchers copy the checker plus
the canonical body parser and its changed-path helper so the packaged checker
matches CI behavior.

## Intentional

- Do not exempt arbitrary marker-only PRs; the explicit body marker and the live
  Markdown-only regular-blob proof must both be present.
- Do not skip live reconciliation entirely for docs-only PRs; open Codex threads
  and current-head `CHANGES_REQUESTED` still fail.
- Do not trust a partial GitHub file listing or stale pre-file-proof review
  snapshot.
- Do not let a PR retarget or changed-file-count movement reuse a docs-only
  proof from an older base generation.
- Do not make ordinary reviewed PRs depend on docs-only tree/file proof.
- Do not let installed watcher packaging drift from the checker's imports.
- Keep the two plan archive moves in this PR because #2247 is the live docs-only
  failure that proves the need for this narrow gate change.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_check_ai_reconciliation_live.py -q` - 63 passed.
- `python -m pytest tests/test_pr_watcher.py::test_installed_entrypoint_writes_consumer_accepted_snapshot -q` - 1 passed.
- `python -m py_compile scripts/check_ai_reconciliation_live.py scripts/install_codex_wake_bridge.py` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed.
- `python scripts/sync_pr_plan.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main --check` - passed.
- `python scripts/audit_plan_doc.py plans/PR-Docs-Only-Live-Reconciliation.md` - passed.
- `python scripts/audit_plan_doc_files_touched.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed.
- `python scripts/audit_plan_doc_diff_size.py plans/PR-Docs-Only-Live-Reconciliation.md origin/main` - passed, estimate 1090 actual 971.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Docs-Only-Live-Reconciliation.md` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/archive-review-workflow-plans-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 4 |
| `plans/PR-Docs-Only-Live-Reconciliation.md` | 221 |
| `plans/archive/PR-Codex-Review-Scope-Reset.md` | 0 |
| `plans/archive/PR-Codex-Review-Thread-Event-Canary.md` | 0 |
| `scripts/check_ai_reconciliation_live.py` | 353 |
| `scripts/install_codex_wake_bridge.py` | 22 |
| `tests/test_check_ai_reconciliation_live.py` | 480 |
| `tests/test_pr_watcher.py` | 10 |
| **Total** | **1090** |
