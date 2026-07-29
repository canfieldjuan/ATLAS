# PR-Codex-Review-Thread-Event-Canary

## Why this slice exists

#2240 intentionally deferred live proof of `pull_request_review_thread`
reachability until its workflow changes landed on `origin/main`. After #2240
merged, this slice created and resolved real review-thread canaries on #2245.
No AI Reconciliation run with event `pull_request_review_thread` appeared for
the tested PR heads. GitHub's Actions event reference lists
`pull_request_review` and `pull_request_review_comment`, but not
`pull_request_review_thread`, as workflow triggers.

The same live pass exposed a second process mismatch: on the current #2245
head, Codex returned its clean result as a PR comment that names the reviewed
commit, not as a formal review record. The live reconciliation gate therefore
had current-head Codex evidence available, but still failed because it only
accepted formal review records.

This slice is intentionally over the normal diff budget. The first fix only
updated the live reconciliation checker, but review feedback showed the same
attestation boundary also feeds the local PR watcher, wake bridge, shell-owned
PR watcher, and the tests that prove each consumer. A smaller patch would leave
split-brain readiness semantics where one gate accepts a clean Codex comment
and another still blocks on "missing review."

### Problem-derived contract

- Root cause: the previous workflow slice treated `pull_request_review_thread`
  as a plausible GitHub Actions trigger without live proof, and the live
  reconciliation attestation logic treated formal review records as the only
  valid current-head Codex proof even when the connector supplied a clean
  current-head PR comment. The root issue is mismatched GitHub/Codex event and
  attestation shape in every trusted reconciliation/readiness consumer, not a
  missing test fixture.
- Correct fix must touch/change: remove `pull_request_review_thread` from the
  AI Reconciliation workflow trigger list and the review-retrigger handoff
  condition; update the workflow comments and local tests so the repo only
  claims supported review/review-comment events refresh the required
  `live-reconciliation` context. Keep review-thread status as a read/query gate
  inside `scripts/check_ai_reconciliation_live.py`, and extend only its
  current-head attestation input so a Codex-authored clean-review PR comment
  with a reviewed commit prefix matching the current head can satisfy the same
  freshness requirement when the checker/watchers read PR comments. The local
  PR watcher, wake bridge, and owned-PR shell watcher must consume the same
  attestation model. Formal `CHANGES_REQUESTED` reviews must still block.
- Must not change: do not change EOM/product behavior, customer-visible
  surfaces, Codex review policy semantics, branch-protection required context
  names, trusted-base checkout posture, bot identity matching, open PRs in
  other lanes, or the live reconciliation decision logic.

## Scope (this PR)

Ownership lane: workflow/codex-review-thread-event-canary
Slice phase: workflow/process

1. Remove the unsupported `pull_request_review_thread` Actions trigger from
   AI Reconciliation live workflow YAML.
2. Remove the unsupported `pull_request_review_thread` event branch from the
   default-branch review-retrigger workflow.
3. Update tests so they assert the supported trigger surface:
   `pull_request_target`, `pull_request_review`, and
   `pull_request_review_comment`.
4. Accept Codex clean-review PR comments as current-head attestation only when
   the bot author is exact and the reviewed commit prefix matches the PR head.
5. Do not enable `issue_comment` CI refresh in this PR: GitHub only evaluates
   issue-comment workflow triggers from the default branch, so this PR cannot
   prove that path before merging it.
6. Keep the local watcher, wake bridge, and owned-PR shell watcher aligned with
   the same "current-head Codex review attestation" semantics.
7. Record the live canary evidence that disproved the thread-event path and the
   live clean-comment evidence that satisfied the current-head attestation gate.

### Files touched

- `.github/workflows/ai_reconciliation_live.yml`
- `.github/workflows/ai_reconciliation_review_retrigger.yml`
- `plans/PR-Codex-Review-Thread-Event-Canary.md`
- `scripts/check_ai_reconciliation_live.py`
- `scripts/codex_wake_bridge.py`
- `scripts/pr_watcher.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_check_ai_reconciliation_live.py`
- `tests/test_codex_wake_bridge.py`
- `tests/test_pr_watcher.py`
- `tests/test_watch_owned_pr.py`

### Review Contract

- Acceptance criteria:
  - `.github/workflows/ai_reconciliation_live.yml` still runs the required
    trusted-base job on `pull_request_target` and the advisory job on supported
    review/review-comment events; settled by
    `tests/test_check_ai_reconciliation_live.py`.
  - `.github/workflows/ai_reconciliation_live.yml` no longer claims
    `pull_request_review_thread` as a trigger; settled by
    `tests/test_check_ai_reconciliation_live.py`.
  - `.github/workflows/ai_reconciliation_review_retrigger.yml` reruns the
    trusted required context only after supported `pull_request_review` and
    `pull_request_review_comment` workflow runs; settled by
    `tests/test_check_ai_reconciliation_live.py`.
  - The live reconciliation script still reads review threads and fails on
    unresolved scoped Codex/Copilot threads; this PR does not remove that gate.
  - `scripts/check_ai_reconciliation_live.py` accepts current-head clean-review
    PR comments only from exact configured bot logins and only when the comment
    includes a reviewed commit prefix matching the current PR head; settled by
    `tests/test_check_ai_reconciliation_live.py`.
  - `scripts/pr_watcher.py`, `scripts/codex_wake_bridge.py`, and
    `scripts/watch_owned_pr.sh` describe and enforce "current-head Codex review
    attestation" consistently, so no readiness path still requires only a
    formal review record; settled by `tests/test_pr_watcher.py`,
    `tests/test_codex_wake_bridge.py`, and `tests/test_watch_owned_pr.py`.
  - Formal current-head `CHANGES_REQUESTED` reviews still override a clean
    PR-comment attestation; settled by
    `tests/test_check_ai_reconciliation_live.py`.
- Reachability proof:
  - Created/resolved a review-thread canary on #2245 head
    `7762288c0c0b90d0b90fe3823a104d38c9614347`; no
    `pull_request_review_thread` AI Reconciliation run appeared.
  - Created/resolved a review-thread canary on #2245 head
    `aa9e1ee6314aeb8cb800f07013954f7fdfd42d12`; no
    `pull_request_review_thread` AI Reconciliation run appeared.
  - GitHub Actions documentation lists `pull_request_review` and
    `pull_request_review_comment` workflow triggers, but not
    `pull_request_review_thread`.
  - On #2245 head `92eac478eb4397117f2513baecd8b30e7b2b56c8`, Codex posted
    `Codex Review: Didn't find any major issues` with reviewed commit
    `92eac478eb`; after the attestation patch, the live reconciliation command
    accepted that exact-head clean comment and found no open scoped threads.
- Affected surfaces:
  - `.github/workflows/ai_reconciliation_live.yml`
  - `.github/workflows/ai_reconciliation_review_retrigger.yml`
  - `scripts/check_ai_reconciliation_live.py`
  - `scripts/codex_wake_bridge.py`
  - `scripts/pr_watcher.py`
  - `scripts/watch_owned_pr.sh`
  - `tests/test_check_ai_reconciliation_live.py`
  - `tests/test_codex_wake_bridge.py`
  - `tests/test_pr_watcher.py`
  - `tests/test_watch_owned_pr.py`
  - `plans/PR-Codex-Review-Thread-Event-Canary.md`
- Risk areas: silently leaving an unsupported trigger claim in YAML/comments,
  removing supported review/comment refresh coverage, changing the required
  context name, or weakening the actual unresolved-review-thread gate.
- Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam:
  `.github/workflows/ai_reconciliation_live.yml` and
  `.github/workflows/ai_reconciliation_review_retrigger.yml` define which
  GitHub event classes can refresh the live reconciliation check.
- Replaced-path behaviors: unsupported thread-resolution event refresh is
  removed because the live canary and GitHub Actions docs do not support it.
- Guard-relevant fields: workflow event name, PR base SHA checkout, PR head SHA
  selected by the retrigger run, required-context rerun ID, PR comment author,
  clean-review marker, and reviewed commit prefix.
- Attestation recognizer closure: the accepted clean-comment attestation set is
  closed to exact configured Codex bot logins, the canonical clean phrase
  `didn't find any major issues`, and the reviewed-commit marker
  `**Reviewed commit:**` followed by a 10-40 hex-character commit prefix that
  matches the current PR head. Comments outside that set do not attest the head.
- Caller x input shape: GitHub Actions event payloads for
  `pull_request_target`, `pull_request_review`, and
  `pull_request_review_comment`.

### Deployed-config probing

- Deployed/default config values: workflow filenames remain
  `.github/workflows/ai_reconciliation_live.yml` and
  `.github/workflows/ai_reconciliation_review_retrigger.yml`.
- Explicit value probe: tests assert supported review/review-comment event
  wiring, no issue-comment workflow wiring for PR comments until a default-branch
  canary can prove it, required-context rerun wiring,
  exact bot comment authorship, clean comment text, reviewed commit prefix
  matching, comment pagination, and malformed-comment fail-closed behavior.
- Absent value probe: tests assert `pull_request_review_thread` is absent from
  both workflow trigger/condition surfaces.
- Default-session/default-context probe: trusted-base checkout remains pinned
  to `${{ github.event.pull_request.base.sha }}` for live jobs.
- Side-effect ordering: no product/runtime side effects; this is workflow
  trigger correction only.

## Mechanism

The required `live-reconciliation` context still runs under
`pull_request_target` from trusted base code. The advisory review-events job
still runs the same base-ref script when GitHub emits a supported review or
review-comment workflow event. The review-retrigger workflow still runs from
the default branch after those advisory workflow runs and reruns the latest
trusted `pull_request_target` run for the same head SHA.

The difference is that the repo no longer claims thread resolution alone emits
a GitHub Actions workflow event. Unresolved review threads remain part of the
live reconciliation decision; their refreshed status is picked up on the next
supported PR/review/review-comment event or push/body edit.

The live reconciliation gate and the local readiness watchers now treat Codex's
clean PR comment as a current-head attestation when the comment is authored by
an exact configured bot login and its reviewed commit prefix matches the current
PR head. That mirrors the connector behavior observed on #2245 without allowing
arbitrary bot chatter or stale-head comments to satisfy the gate.

## Intentional

- This PR removes the unsupported trigger claim instead of adding a permanent
  canary script, because the one-off live canary already disproved the event
  path and keeping a verifier for an unsupported event would add confusing
  process debt.
- The `scripts/check_ai_reconciliation_live.py` change is limited to
  current-head attestation. The gate still reads GraphQL review threads and
  fails when scoped automated-review threads remain unresolved.
- The watcher field name `codex_head_review_count` is retained for snapshot
  compatibility, but its text and inputs now mean "current-head Codex review
  attestation" rather than "formal review object only."

## Deferred

- If GitHub later adds `pull_request_review_thread` to Actions workflow
  triggers, add a new proof slice before restoring any thread-resolution
  trigger path.

Parking predicate: this slice only parks future platform support for a
dedicated `pull_request_review_thread` Actions trigger. It does not park
same-surface findings about current supported event reachability,
current-head attestation freshness, nullable GitHub authors, or local readiness
consumers; those must be fixed in this PR.

Parked hardening: future proof slice for a newly supported
`pull_request_review_thread` trigger, plus a separate default-branch canary
before enabling `issue_comment` CI refresh for clean Codex PR comments.

## Verification

- Live #2245 canary on head `7762288c0c0b90d0b90fe3823a104d38c9614347`:
  review thread created/resolved; no AI Reconciliation
  `pull_request_review_thread` run appeared.
- Live #2245 canary on head `aa9e1ee6314aeb8cb800f07013954f7fdfd42d12`:
  review thread created/resolved; no AI Reconciliation
  `pull_request_review_thread` run appeared.
- Live #2245 clean-comment attestation on head
  `92eac478eb4397117f2513baecd8b30e7b2b56c8`: Codex posted a clean review
  comment for commit `92eac478eb`; `python scripts/check_ai_reconciliation_live.py --repo canfieldjuan/ATLAS --pr 2245`
  passed with no open scoped Codex review threads.
- `python -m pytest tests/test_check_ai_reconciliation_live.py -q`
  - passed, 42 tests.
- `python -m pytest tests/test_check_ai_reconciliation_live.py tests/test_pr_watcher.py tests/test_watch_owned_pr.py tests/test_report_pr_watcher_state.py tests/test_codex_wake_bridge.py -q`
  - passed, 189 tests.
- `python -m py_compile scripts/check_ai_reconciliation_live.py scripts/pr_watcher.py scripts/codex_wake_bridge.py`
  - passed.
- `bash -n scripts/watch_owned_pr.sh`
  - passed.
- `python scripts/audit_plan_doc.py plans/PR-Codex-Review-Thread-Event-Canary.md`
  - passed.
- `python scripts/audit_plan_doc_files_touched.py plans/PR-Codex-Review-Thread-Event-Canary.md origin/main`
  - passed.
- `python scripts/audit_plan_doc_diff_size.py plans/PR-Codex-Review-Thread-Event-Canary.md origin/main`
  - passed, 0.0% drift.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Codex-Review-Thread-Event-Canary.md`
  - passed.
- `git diff --check origin/main -- . ':!node_modules'`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/ai_reconciliation_live.yml` | 44 |
| `.github/workflows/ai_reconciliation_review_retrigger.yml` | 34 |
| `plans/PR-Codex-Review-Thread-Event-Canary.md` | 276 |
| `scripts/check_ai_reconciliation_live.py` | 146 |
| `scripts/codex_wake_bridge.py` | 2 |
| `scripts/pr_watcher.py` | 106 |
| `scripts/watch_owned_pr.sh` | 83 |
| `tests/test_check_ai_reconciliation_live.py` | 181 |
| `tests/test_codex_wake_bridge.py` | 2 |
| `tests/test_pr_watcher.py` | 136 |
| `tests/test_watch_owned_pr.py` | 162 |
| **Total** | **1172** |
