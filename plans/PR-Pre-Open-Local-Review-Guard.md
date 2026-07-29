# PR-Pre-Open-Local-Review-Guard

## Why this slice exists

The merged AGENTS mechanical enforcement audit found that the local review
bundle is required before opening or updating a PR, but `open_pr.sh` only runs
the PR body audit before `gh pr create/edit`. A builder can push directly and
then open the PR, which lets GitHub see an immediately red branch even though the
AGENTS workflow says `local_pr_review.sh` is the pre-open fast path. This
workflow/process slice closes that helper gap without changing branch
protection, CI required contexts, or product behavior. The diff is over the
400-LOC soft cap because this is a guard/admission-boundary slice and the
correct proof requires both positive and negative wrapper fixtures across
create/edit/docs-only paths, stale `HEAD`, stale `origin/main`, stale body,
dry-run/refspec rejection, remote-branch publication proof, and failed-push
ordering. The Codex review on the first published head found that a zero-exit
`git push --dry-run` or unrelated refspec could still write proof; this update
closes that root cause instead of treating the example only. The follow-up
Codex review found two same-class bypasses: proof was still not bound to the
branch being opened, and Git's long-option abbreviation for `--no-verify` could
disable the managed pre-push hook. The next Codex review found three more
same-class bypasses: Git also abbreviates `--dry-run`, `open_pr.sh` could pass
target-changing create args that were not covered by the proof, and both
wrappers used a mutable body file after review. This update closes those guard
classes too. The current-head Codex review then found the remaining shared seam:
the wrappers still interpreted shell/Git state piecemeal. This update closes
that seam by parsing the supported push option shapes, capturing proof inputs
before review, verifying those inputs remain unchanged after push, and
revalidating the published branch immediately before `gh`.
The next current-head Codex review found three adjacent proof-boundary gaps
inside that same seam: remote-head revalidation still happened before the
`gh pr view` existence probe, environment-selected `GH_REPO` could retarget the
GitHub CLI without passing through argv validation, and the push-arg parser did
not consume operands for value-taking Git push options. This update closes those
remaining parser/target/timing classes.
The following current-head Codex review found that endpoint sampling still
allowed an ABA checkout interleaving, the boundary enumeration did not
disposition each wrapper input shape, and numeric branch names were ambiguous
with GitHub PR-number selectors. This update runs review in an immutable
captured-head worktree, expands the boundary enumeration by caller/input class,
and resolves existing PRs by matched head ref before editing.
The latest current-head Codex review found three remaining target-binding gaps:
a positional non-`origin` push remote could still receive the push while proof
checked `origin`, same-branch fork PRs could be mistaken for the existing
current-repo PR, and the create path still let `gh pr create` infer the head
branch from mutable checkout state. This update rejects non-`origin` positional
remotes, verifies existing PR matches against current owner/repo identity, and
passes the proven branch explicitly with `gh pr create --head`.

### Problem-derived contract

- Root cause: `push_pr.sh` is the only helper that guarantees the mechanical
  local review bundle ran before publication, but `open_pr.sh` has no durable
  local signal proving that the current `HEAD` and PR body are the same inputs
  that passed that bundle.
- Correct fix must touch/change: `push_pr.sh` must write a local, ignored proof
  only after its guarded push succeeds and the current branch's remote-tracking
  ref equals local `HEAD`; `push_pr.sh` must reject dry-run pushes and refspecs
  that do not publish the current branch; `push_pr.sh` must reject
  `--no-verify` and Git-abbreviated spellings that disable the pre-push hook;
  `push_pr.sh` must also reject Git-abbreviated and bundled dry-run spellings,
  reject only the Git spellings that bypass verification, preserve valid
  non-bypass options such as `--no-verbose`, capture branch/head/base/body proof
  inputs before review, run local review against that immutable captured Git
  state, and verify those exact inputs remain unchanged after push; `open_pr.sh`
  must verify the proof's branch, head, base, and body
  snapshot before `gh pr create/edit`, must reject environment-selected target
  overrides such as `GH_REPO`, must revalidate the remote branch SHA after the
  PR existence probe and immediately before the mutating `gh` call, must pass
  the captured branch explicitly to `gh pr create --head`, must verify existing
  PR lookup results belong to the current repository before edit, must pass
  the same immutable body snapshot to `gh`, and must reject target-changing
  create args outside the proof's
  current-branch/current-repo/`origin/main` contract; tests must cover missing,
  stale-head, stale-base, stale-body, stale-branch-at-same-head, create, edit,
  docs-only, dry-run and abbreviated/bundled dry-run, unrelated-refspec,
  abbreviated-no-verify, `--no-verbose`, value-taking push options, stale-remote
  before and after the PR existence probe, base movement during push, same-branch
  fork PR lookup, explicit create-head binding, body
  mutation during push/open, argv/environment target overrides, and failed-push
  paths, plus ABA checkout interleavings and numeric branch names.
- Must not change: no CI workflow, branch protection, PR body contract, plan
  admission contract, product code, remote GitHub state semantics, or other open
  PR lanes.

## Scope (this PR)

Ownership lane: workflow/pre-open-local-review-guard
Slice phase: Workflow/process

1. Add a local pre-open proof handshake between `push_pr.sh` and `open_pr.sh`.
2. Block `open_pr.sh` before any `gh pr create/edit` when proof is missing or
   stale for the current `HEAD`, `origin/main`, or body file.
3. Keep the proof local to the worktree/Git directory so it is not committed and
   not treated as a branch-protection or CI mechanism.

### Review Contract

- Acceptance criteria:
  1. `scripts/push_pr.sh` records a proof after a successful guarded push that
     includes the current `HEAD` SHA, `origin/main` SHA, and SHA-256 of the
     reviewed PR body snapshot plus current branch name, but only after
     `refs/remotes/origin/<current-branch>` equals local `HEAD`.
  2. `scripts/open_pr.sh` checks the proof before create and edit, and exits
     before invoking `gh` if the proof is missing, points at another branch,
     another `HEAD`, another `origin/main`, or another reviewed body hash.
  3. `tests/test_open_pr_wrapper.py` proves missing proof, stale `HEAD`, stale
     `origin/main`, stale branch at the same `HEAD`, and stale body cases never
     invoke fake `gh`, and proves matching proof still passes the body over
     stdin for create, edit, and docs-only bodies. It also proves the wrapper
     rejects target-changing `--head`/`--repo`/non-main `--base` args before
     fake `gh`, rejects `GH_REPO` environment target overrides before fake
     `gh`, revalidates the published branch after the PR-existence probe, and
     that a body mutation during fake `gh` still sends the reviewed snapshot.
    4. `tests/test_push_pr_wrapper.py` proves the push wrapper writes the proof
       after successful current-branch publication and does not write it when the
       push fails, the push is a Git dry-run, the refspec targets another branch,
       Git receives a `--no-verify` spelling/abbreviation, Git receives a
       dry-run abbreviation/bundled short dry-run, `origin/main` moves during the
       push, or the remote-tracking branch does not equal local `HEAD`. It also
       proves a body mutation during fake `git push` cannot change the reviewed
       body hash recorded in proof, that `--no-verbose` remains allowed, that
       value-taking push options consume their operands before remote/refspec
       detection, that `--repo` target overrides are rejected, that staged,
       unstaged, and untracked source worktree changes are rejected before
       immutable review, that source dirtied during push is rejected before proof
       write, and that an ABA checkout interleaving during review still reviews
     the captured head, and that a non-`origin` positional push remote exits
     before review, push, or proof.
    5. Numeric branch names are resolved through an unambiguous head-branch PR
       query before edit; `open_pr.sh` never passes a numeric branch name as the
       positional PR selector to `gh pr view/edit`.
    6. R8 execution model: selected component is Git's immutable object store plus
       `git worktree add --detach <captured-head>` for review isolation, with the
       source checkout used only as an admission surface. The admitted execution
       model is one wrapper process for the proof file, arbitrary concurrent
       source-checkout edits, arbitrary remote/base movement, and ordinary
       `git push` success/failure. Property invariant: a proof is written only if
       the source checkout is clean before immutable review and again after push,
       the reviewed body snapshot hash is unchanged, the captured branch/`HEAD`
       /`origin/main` values still match, and `origin/<captured-branch>` equals
       captured `HEAD`; otherwise the helper fails closed before proof. Explicit
       assumption: a process that can modify the Git object database or rewrite
       the ignored proof file outside this wrapper is outside the local-helper
       trust boundary.
- Reachability proof: wrapper-level surface; exercised through the real
  `scripts/open_pr.sh` and `scripts/push_pr.sh` entrypoints with fake `gh`/`git`
  adapters and observable `gh` invocation/proof-file effects.
- Affected surfaces: `scripts/open_pr.sh`, `scripts/push_pr.sh`,
  `tests/test_open_pr_wrapper.py`, `tests/test_push_pr_wrapper.py`, and this
  plan.
- Risk areas: stale proof acceptance, skipped source worktree edits,
  cross-worktree interleavings, bypassing body-stdin behavior, breaking docs-only
  PR bodies, and accidentally turning the local helper proof into a CI or
  branch-protection claim.
- Reviewer rules triggered: R1, R2, R8, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/push_pr.sh` local-review execution state.
  - Caller x input: managed pre-push hook installed, no managed hook installed,
    or `ATLAS_SKIP_LOCAL_PR_REVIEW=1`.
    - Disposition: changed. All supported push paths run one wrapper-owned local
      review in a temporary worktree pinned to the captured `HEAD`, but only
      after the source checkout is clean; the final `git push` receives
      `ATLAS_SKIP_LOCAL_PR_REVIEW=1` so the mutable checkout is not reviewed a
      second time by the hook, then the source checkout is checked again before
      proof write.
    - Preserved: the PR body snapshot is still passed through
      `ATLAS_CURRENT_PR_BODY_FILE`; invalid bodies still fail before push.
    - Rejected: staged, unstaged, untracked, and dirty-during-push source edits do
      not receive proof; ABA checkout interleavings no longer affect the reviewed
      Git tree.
- Boundary path/seam: `scripts/push_pr.sh` push-argument admission.
  - Caller x input: default args, explicit `origin HEAD`, upstream `-u`,
    force-with-lease, verbosity options, value-taking push options, dry-run
    spellings, hook-bypass spellings, unrelated refspecs, and target overrides.
  - Disposition: preserved for default/current-branch publication, `-u`,
    `--force-with-lease`, `--no-verbose`, and value-taking options whose operands
    are consumed before remote/refspec detection; rejected for non-`origin`
    positional remotes, dry-run
    abbreviations/bundles, `--no-veri*`, unrelated refspecs, missing explicit
    refspecs, and `--repo` target overrides.
  - Deferred/N-A: arbitrary non-origin target publication remains outside this
    helper contract.
- Boundary path/seam: `scripts/open_pr.sh` proof and target admission.
  - Caller x input: create path, edit path, docs-only body, missing proof, stale
    branch/head/base/body proof, stale published branch before/after PR lookup,
    body mutation during `gh`, argv target selectors, and `GH_REPO`.
  - Disposition: preserved for matching proof and stdin body delivery; rejected
    for missing/stale proof, stale published branch, target-changing argv,
    non-main base, `GH_REPO`, direct body args, and body mutation after review.
  - Deferred/N-A: cross-repository PR creation and non-main base PRs require a
    separately reviewed helper.
- Boundary path/seam: `scripts/open_pr.sh` existing-PR resolution.
  - Caller x input: nonnumeric branch names, numeric branch names, no existing
    PR, same-branch fork PR records, one existing PR for the exact current-repo
    head branch, multiple matching PR records.
  - Disposition: changed. Existing PRs are resolved with
    `gh pr list --head <branch> --json number,headRefName,headRepository,headRepositoryOwner,isCrossRepository`
    and edited by returned PR number only after `headRefName`, owner, repository,
    and non-cross-repository status match the current repo and proof branch.
    Numeric branch names are never passed as ambiguous positional selectors.
    Same-branch fork records are ignored. No match takes the create path; multiple
    current-repo matches fail closed.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - local wrapper guard only.
- Explicit value probe: proof file with matching `HEAD`, `origin/main`, and body
  hash plus a published matching remote branch allows fake `gh` invocation in
  wrapper tests.
- Absent value probe: missing proof exits before fake `gh` invocation in wrapper
  tests.
- Default-session/default-context probe: stale proof for another `HEAD`,
  branch, `origin/main`, or body hash exits before fake `gh` invocation in
  wrapper tests.
- Side-effect ordering: `push_pr.sh` writes proof only after the fake successful
  push and remote-tracking branch update; failed push, Git dry-run, unrelated
  refspec, abbreviated/bundled dry-run, abbreviated `--no-verify`,
  `origin/main` movement during push, and stale remote-tracking state leave no
  proof in wrapper tests. Body mutations during fake push/create do not change
  the reviewed snapshot hash or stdin payload. Target-changing `open_pr.sh`
  create args, `GH_REPO`, stale published branch state before fake `gh`, and
  stale published branch state after the PR-existence probe exit before fake
  `gh`. Value-taking push options consume their operands before remote/refspec
  detection, while `--repo` target overrides exit before fake `git push`.

### Files touched

- `plans/PR-Pre-Open-Local-Review-Guard.md`
- `scripts/open_pr.sh`
- `scripts/push_pr.sh`
- `tests/test_open_pr_wrapper.py`
- `tests/test_push_pr_wrapper.py`

## Mechanism

Both wrappers copy the PR body into a temporary snapshot before auditing,
pushing, proving, or feeding `gh`, then delete that snapshot on exit.
`push_pr.sh` parses the supported push argument shapes before review: it rejects
dry-run options including bundled `-n`, abbreviated `--dr*` spellings, refspecs
that do not publish the current branch, non-`origin` positional remotes, push
target overrides such as `--repo`, and only the `--no-veri*` spellings that would
bypass the managed pre-push hook; it also consumes operands for value-taking push
options before identifying the remote and refspec. After refreshing
`origin/main`, it captures the branch, `HEAD`, base SHA, and body snapshot hash
before audit/local review, rejects staged/unstaged/untracked source checkout
changes, then creates a temporary detached worktree at that captured `HEAD` and
runs `local_pr_review.sh` there so the review cannot observe a later checkout
transition in the mutable worktree. The detached review exports the captured
branch as `GITHUB_HEAD_REF`, preserving the current-PR identity needed by the
cross-session drift audit without reviewing the mutable branch checkout. The
final push skips the managed hook because the immutable wrapper review has
already run once. After push success, it checks the source checkout is still
clean, then verifies those captured values remain unchanged and verifies
`refs/remotes/origin/<current-branch>` equals local `HEAD`; only then does it
write the captured values to the ignored proof file under the Git directory.

`open_pr.sh` calculates the same values against its snapshot before any
create/edit operation, rejects target-changing create args outside the
current-branch/current-repo/main-base contract, rejects `GH_REPO` environment
target overrides, refreshes the current remote branch before the existence probe
and again immediately before the selected mutating `gh` call, resolves existing
PRs with an unambiguous head-branch query that must match current repository
owner/name before edit, passes the proven branch explicitly to `gh pr create
--head`, requires that remote SHA to equal local `HEAD`, and feeds the same
snapshot to `gh` over stdin. A stale proof therefore fails closed whenever the
builder commits again, switches to another branch at the same commit,
`origin/main` moves after proof capture, the reviewed branch is force-pushed or
deleted before open, the builder edits/regenerates the PR body after pushing,
the source checkout is dirty before review or before proof write, a body file
mutates during the helper run, the create target changes away from the proof, or
the push did not actually publish the current branch head.

## Intentional

- The proof is local and advisory-helper scoped. It is not a security boundary:
  a user can still bypass helpers or edit local files. The goal is to keep the
  endorsed Atlas helper path from opening immediate-red PRs by accident.
- This slice does not enforce branch naming, draft consent, PR ownership, or
  commit-message shape; those remain separate audit follow-ups.
- The proof records the reviewed body snapshot hash, not the body path, so
  regenerating the same body content does not create needless friction.

## Deferred

- `PR-PR-Mutation-Ownership-Wrapper`: wire session ownership checks into the
  mutation helpers.
- `PR-Open-Ready-Draft-Consent-Guard`: reject `--draft` without operator consent.
- `PR-Branch-Naming-Gate`: enforce or downgrade the builder branch naming
  convention.

Parked hardening: none.

## Verification

- Completed before PR open:
    - `python -m pytest tests/test_open_pr_wrapper.py tests/test_push_pr_wrapper.py -q`
      - passed, 57 tests after the Codex review fixes.
  - `python -m pytest tests/test_content_factory_copy_verification.py -q`
      - passed, 324 tests while investigating the CI unit-gate regression.
  - `python scripts/audit_plan_doc.py plans/PR-Pre-Open-Local-Review-Guard.md`
  - `python scripts/audit_plan_code_consistency.py --base-ref origin/main plans/PR-Pre-Open-Local-Review-Guard.md`
  - `python scripts/sync_pr_plan.py plans/PR-Pre-Open-Local-Review-Guard.md origin/main --check`
  - `python scripts/audit_pr_body.py --repo-root . --base-ref origin/main /tmp/atlas-pr-body-pre-open-local-review-guard.md`
  - `bash scripts/push_pr.sh /tmp/atlas-pr-body-pre-open-local-review-guard.md --force-with-lease origin HEAD`

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Pre-Open-Local-Review-Guard.md` | 325 |
| `scripts/open_pr.sh` | 216 |
| `scripts/push_pr.sh` | 276 |
| `tests/test_open_pr_wrapper.py` | 460 |
| `tests/test_push_pr_wrapper.py` | 922 |
| **Total** | **2199** |
