# AGENTS.md — Atlas multi-agent workflow

Atlas uses **two coordinated Claude Code sessions** for non-trivial
work:

1. **Builder session** — drafts the plan, writes the code, opens the PR.
2. **Reviewer session** — audits each PR independently and posts a
   verdict (BLOCKER / MAJOR / NIT / LGTM). After the verdict, record it as the
   machine-readable `claude-review` gate for the reviewed head SHA with
   `scripts/set_claude_review_status.py` (no open BLOCKER -> `success`, i.e. an
   LGTM or only non-blocking MAJOR/NIT notes; an open BLOCKER -> `failure`).
   Unresolved matrix entries are themselves filed as BLOCKERs, so a review with
   open questions lands on `failure` without changing this mapping; see
   `docs/REVIEWER_MERGE_GATE.md` for the two-gate merge condition and its trust
   boundary.

This file is the contract both sessions work from. The auditor
(prompt at `AUDITOR_PROMPT.md`) handles cross-cutting integration /
canonical / scope checks; this file defines the **PR-shape contract**
that lets the reviewer give the builder a clean LGTM.

**New or restarted builder sessions: read `docs/SESSION_BOOTSTRAP.md`
first.** It carries the get-up-to-speed checklist, the recurring-lapse
list, and the context-discipline rules (stop after opening a PR; read
narrow / run scoped tests during iteration) that keep a session from
compacting mid-work. A session that has drifted post-compaction gets the
redirect prompt in that file.

---

## Review guidelines

Automated reviewers — including the GitHub Codex connector, which reads this
section — follow Atlas's Reviewer Rules Pack in `docs/REVIEWER_RULES.md` (rules
R1–R14) and the PR's Review Contract. Review the same way the Reviewer session does.

- Verify against the codebase, not the PR story (R14): treat the PR description,
  title, and commit messages as unverified claims; reconstruct what the diff
  actually does from the code.
- Every finding cites a rule ID (R1–R14) and `file:line`. Blockers must cite
  `file:line`. Classify each finding as **BLOCKER / MAJOR / NIT / LGTM** per the pack.
- Check the PR's `### Review Contract`: does the diff meet its acceptance criteria?
  Then disposition EVERY rule R1-R14 -- the path-to-rule table in the pack sets how
  deeply each is probed, not which appear, so a rule the table does not trigger is
  still recorded (Pass, or N-A with a reason).
- Hunt the rule categories: requirements match (R1), test evidence (R2),
  security/authorization (R3), data & migration safety (R4), backward compatibility
  (R5), error handling & observability (R6), performance (R7), concurrency &
  idempotency (R8), frontend (R9), maintainability (R10), dependencies & config
  (R11), deployment safety & CI (R12). Fix the class, not the example (R13).
- **Know when you are done** (Review completion, in the pack): a review is complete
  when its matrix is dispositioned -- each acceptance criterion met / not met /
  could-not-determine, EVERY rule R1-R14 pass / fail / not-verified / n-a-with-reason
  (the path table sets probe depth, not which rules appear), and what you
  did not verify listed with the reason. State that matrix. Completeness is never
  "no further case can be found"; on an open surface that point does not exist.
- **Report the class, not the instance.** R13 obliges the builder to fix the class
  rather than the example; the same duty binds you. Findings that share one
  underlying decision are **one** finding naming that decision, with the instances
  as illustrations. If a finding of yours would open "fresh evidence beyond the
  earlier X finding", it belongs merged into X, not filed separately.
- Lead with blockers. `LGTM` (every rule R1-R14 Pass or reasoned N-A, no open
  BLOCKER/MAJOR) is a valid, complete result; do not manufacture NITs.

---

## 0. Product Discipline And Consent Gates

Read `docs/CURRENT_PRODUCT_DISCIPLINE.md` before choosing or widening a product
slice. `BUILD_SPEC.md` is deprecated historical context and must not be used as
the current roadmap or definition of done.

Default to vertical, end-to-end product slices: the smallest real
buyer-visible or operator-visible path that proves the flow through a real
entrypoint and an observable output/state/artifact/job/gate. Workflow/process,
autonomy harness, audit, maturity-gate, or other hardening slices are allowed
only when they unblock the current vertical proof, fix a real
safety/security/privacy/money risk, or are justified by a recent product slice
that failed because that infrastructure gap existed. The plan must name that
blocker, risk, or failed run.

Do **not** change user-facing product shape without explicit operator consent
in the current session or accepted issue/plan. Product shape includes report,
snapshot, email, PDF, landing-page copy/positioning, pricing/checkout,
subscription/entitlement surfaces, buyer-visible tables/cards/sections/labels,
customer-facing promises, and output semantics. If a slice exposes a
product-shape decision, park it in `Deferred`, `HARDENING.md`, or a GitHub issue
and continue only on the technical path that does not decide that shape.

---

## 1. PR shape

Every non-trivial change ships as a single PR with the following
artifacts:

For a human-authored PR, any diff containing a non-Markdown path must add
exactly one `plans/PR-*.md` file. A non-empty Markdown-only diff may omit a
plan only when every changed path is a regular Git blob with `.md` as its sole
suffix and its PR body begins `Docs-only: true`; it may always use the full
plan/body contract instead. Dependabot keeps its explicit generated-PR
exemption. These admission outcomes must be surfaced by local review and CI,
never silently skipped.

### 1a. Plan doc (`plans/PR-<Slice-Name>.md`)

Required sections, in this order:

| Section | Purpose |
|---|---|
| **Why this slice exists** | What's broken / what's missing / what audit item this closes. Tie to a prior plan, audit finding, or a concrete user request. For any coding slice, include a **Problem-derived contract** written before code: root cause from the problem alone, what the correct fix must touch/change to reach that cause, and what must not change. For a fix/defect/review-finding PR, name the **root cause** (the underlying problem, not the surface symptom or the reviewer's wording) and state whether this change fixes the root or treats a symptom -- see §3k. |
| **Scope (this PR)** | The narrow surface this PR touches. Start with an `Ownership lane: <lane>` line, then a `Slice phase: <phase>` line, then a numbered list of intent and a "Files touched" subsection. |
| **Mechanism** | Short prose (and code stub if helpful) explaining *how* the change works -- enough that the reviewer doesn't have to reverse-engineer it from the diff. |
| **Intentional** | Things that look wrong but aren't -- explicit trade-offs and rejected alternatives ("no `warnings.warn` shim because ..."). Saves reviewer cycles. |
| **Deferred** | Things explicitly punted to a follow-up slice. Each item should name the future PR or describe what would unlock it. Include "Parked hardening: none" or list the `HARDENING.md` entries added by this slice. |
| **Verification** | The specific commands the builder ran locally + their pass counts. Reviewer reproduces. |
| **Estimated diff size** | LOC budget; flag if approaching 400 LOC. |

The **Scope** section also carries a **Review Contract** block (a `### Review
Contract` subsection): acceptance criteria the reviewer checks one-by-one,
affected surfaces, risk areas, and the reviewer rule IDs the changed paths
trigger. The builder codes against it; the reviewer reviews against it. See
`docs/REVIEWER_RULES.md` for the rule pack and the path-to-rule trigger table.
For any new runtime, workflow, UI, report, billing, delivery, or public
contract surface, the Review Contract must also name the reachability proof:
the real entrypoint exercised and the observable output/state/artifact/job/gate
result that proves the surface is wired.

### 1b. PR body

For a non-empty human PR that intentionally carries no plan and changes only
regular Git blobs with `.md` as their sole suffix, use this narrower body
instead:

```
Docs-only: true

<optional prose>
```

The marker is invalid for any non-Markdown or empty diff. A docs-only PR that
does carry a plan must use the full body shape below.

Otherwise, mirror the plan-doc framing in the PR description:

```
Plan: plans/PR-<Slice-Name>.md
Slice phase: <phase>
Ownership lane: <canonical-lowercase-lane>

<one-paragraph why>

## Intentional
- ...

## Deferred
- ...

## Parked hardening
- None. (or: `HARDENING.md` entry title and why it was parked)

## Cold diff reconstruction
- Changed: <file:line + what the diff actually does>
- Contract match: <how each change traces to the Problem-derived contract>
- Gaps: <none, or every untraced change / missing contract item / forbidden touch>

## Verification
- ...

## Diff size
N files, +X / -Y
```

### 1c. Commit message

Same `Plan: ...` and `Slice phase: ...` lead lines + Intentional /
Deferred / Parked hardening / Cold diff reconstruction sections as the PR
body. Squash-merge collapses to one canonical commit at merge time.

### 1d. Diff budget

Target **<400 LOC** per PR. Soft cap; over-budget PRs ship if the
slice is genuinely indivisible, but the plan doc must justify the
overage in **Why this slice exists**.

### 1e. Branch naming

`claude/pr-<slice-name>` for builder branches.
`claude/<topic>` for non-PR scratch.

### 1f. Open ready for review

Open the PR as **ready for review** by default. Do not open draft PRs
unless the operator explicitly asks for a draft. Automated review tools
do not review draft PRs, so draft mode burns review time and hides
feedback until the PR is manually marked ready.

### 1g. Teardown on merge

`origin/main` is the only source of truth; local branches and worktrees
are **disposable**. When a PR merges, tear down its worktree and branch
the same session — **worktree first, then branch** (a branch checked out
in a worktree cannot be deleted: `git branch -D` fails with `'<branch>'
is already used by worktree at ...`):

- `git worktree remove <dir>` for any worktree dedicated to it
  (`--force` if it still holds throwaway state). This frees the branch.
- `git branch -D <branch>` (squash-merge leaves the local branch
  unmerged by content, so `-d` refuses — `-D` is expected here).
- Archive the merged plan doc so `plans/` only ever holds **in-flight**
  slices (the plan's content is already preserved in the squash commit).
  On a local `main` synced to `origin/main` (`git checkout main && git
  pull`), move **your own** plan by name and refresh the index:

  ```bash
  git mv plans/PR-<Slice>.md plans/archive/
  python scripts/archive_plans.py index   # rebuild plans/INDEX.md
  ```

  Land that move on `origin/main` as a trivial housekeeping commit (or
  fold the `git mv` into your next branch off `main` if direct main
  commits are gated). Move **only** your own merged plan by name — do
  **not** run `archive_plans.py archive` (bulk) during teardown: it would
  sweep concurrent sessions' still-in-flight plans out of the root. The
  non-blocking "Plans archive backlog" advisory in `local_pr_review.sh` is
  the backstop that nudges you if this step is ever missed.

Do **not** let merged branches or finished worktrees linger. They drift
behind `origin/main`, accumulate stale staged state, and become the
hundreds-of-commits-behind worktree and the 300-file dirty index that
just mirrors already-landed PRs — the exact mess a cleanup session has
to untangle. Before resurrecting anything from a stale local branch,
check it against `origin/main` first (`git cherry -v origin/main
<branch>`); the equivalent change has usually already landed.

Cleanup safety: never run `git clean -f` without a `git clean -nd`
dry-run first, and read the list. Untracked secret files live in the
tree (`.env.bak-*`, `*.production.env`, gitignored per the env section)
and a blanket clean — especially with `-x` — deletes them.

---

## 2. Reviewer verdict shape

The reviewer comments **once per push** with a verdict at one of these
levels:

| Level | Meaning | Builder action |
|---|---|---|
| **BLOCKER** | Correctness, security, contract break, or CI red. Must fix before merge. | Fix or push back with rationale. |
| **MAJOR** | Architectural / scope / pattern concern, **or a proven defect whose blast radius does not warrant blocking**. Strong recommendation but not auto-block. | Fix in this PR if the fix is small; otherwise discuss before deferring. |
| **NIT** | Style, naming, comment polish. Skip-worthy. | Apply if 1-line; skip otherwise. The reviewer should mark NITs as skip-worthy explicitly. |
| **LGTM** | All gates green, R14 verified, no remaining concerns. | Merge. |

### 2a. Reviewer's verification template

The reviewer should produce something like:

```
**Reviewed head:** `<sha from checked-out PR head>`

**Reconstruction (gaps first, per docs/PR_RECONSTRUCTION_PROTOCOL.md):**
- Gaps: <diff != description / diff != correct fix / diff changes unmentioned
  things>, each cited `file:line`.
- Confirmed: <finding + citation>. Contradicted: <finding + citation>.
  Could-not-determine: <finding + why unresolved>.

**Verification (independent):**
1. <claim from PR description> -- verified via <command>
2. <invariant from Mechanism> -- confirmed at <file:line>
3. ...

**Codebase verification (R14):**
- Changed code inspected: <files/lines>.
- Caller/test/artifact spot-checks: <rg command, test path, generated artifact, or route checked>.
- Not verified: <claim skipped + reason>. (Use "None" only if every verdict claim was checked.)

**Plan-doc compliance:** Why / Scope / Mechanism / Files touched /
Intentional / Deferred / Verification -- matches AGENTS.md framework.
Slice phase is named and matches the PR's scope. Parked hardening is
named in Deferred or explicitly marked none.

**Rule results (EVERY rule R1-R14, see docs/REVIEWER_RULES.md):**
- R1 Requirements match: Pass/Fail/Not-Verified/N-A
- R2 Test evidence: Pass/Fail/Not-Verified/N-A
- R3 Security/auth ... R14 Codebase verification: Pass/Fail/Not-Verified/N-A
(List every rule R1-R14. The path-trigger table sets probe DEPTH, not which
rules appear -- a rule the table does not trigger is still recorded, as Pass or
as N-A with a reason. Cite file:line on any Fail. Not-Verified ends the search
but blocks LGTM.)

**boundary-probe:** <N-A, or what guard-shaped probe applied + result. Required
before LGTM on guards, validators, caps, classifiers, gates, sanitizers,
denylists, parser admission rules, or safety checkers.>

**AI reconciliation:** AI findings reviewed: Y/N. All fixed or waived: Y/N.
Waivers justified in PR body: Y/N.

**Defensible trade-offs (no action needed):**
- <decision> -- <why it's the right call>

**<N> NITs (skip-worthy):**
1. ...

LGTM. (or: BLOCKER -- ...)
```

A finding is written as `Rxx (LEVEL) file:line - issue - required fix`,
mapping the rule to the verdict level above. A bare "LGTM" with no rule
matrix and no independent verification is worse than no comment.

### 2b. CI gate

CI must be green before LGTM. If CI is red on a transient failure
(flaky test, infra), the reviewer can call that out separately and
not block.

---

## 3. Builder workflow

### 3a. Plan first

Open `plans/PR-<Slice-Name>.md` and write the full plan doc **before**
any code change. The plan is the contract; the code is the
implementation of the contract.

If the plan changes mid-implementation (you discovered something the
plan missed), update the plan doc in the same commit. The plan and
code ship together.

Code for independent reconstruction
(`docs/CODING_FOR_RECONSTRUCTION_REVIEW.md`). Before coding, derive the
Problem-derived contract from the problem alone: root cause, what the correct
fix must touch/change, and what must not change. Build only to that contract.
Before opening or updating a PR, reconstruct your own diff cold with
`file:line` citations and compare it to the contract. If the diff changes
unmentioned behavior, misses a contract item, or touches something the contract
said to leave alone, fix the gap before calling the slice done.

### 3a.1. Session ownership map

Every builder session must maintain its own local session-state file at the
repository root, using `docs/SESSION_STATE_TEMPLATE.md` as the shape. The
preferred filename is `SESSION_STATE.<session-id>.local.md` (for example,
`SESSION_STATE.codex-workflow-1982.local.md`). The legacy
`SESSION_STATE.local.md` filename is allowed only when exactly one active
session owns that worktree. These files are ignored by git because they are
volatile session state, but one per session is mandatory working context. Set
`ATLAS_SESSION_STATE_FILE` to the session's file path, or pass `--state-file`
to the ownership guard, before doing PR work.

Update the map:

- at session start or after compaction/restart reorientation;
- before opening a PR;
- after pushing a PR update;
- after merging a PR;
- before handing back to the operator.

The map must name the current lane, current task, Spark/subagent routing used
or considered, owned active PR number/title/branch/plan/head SHA when one
exists, PRs this session may touch, PRs this session must not touch, and the
last safe action.

Before inspecting comments, pushing updates, closing, or merging any
PR, the builder must verify all of the following:

1. `gh pr list --state open` has been checked in this resume window.
2. `git log --oneline -15 origin/main` has been checked for already
   landed work.
3. The target PR is listed in this session's state file under "Owned Active
   PR" or "PRs This Session May Touch".
4. The local branch and expected head SHA match the target PR when a
   merge or force-push is about to happen.

Run the local guard before PR mutation whenever the target PR metadata
is known:

```bash
python scripts/check_session_pr_ownership.py \
  --pr <number> \
  --branch <headRefName> \
  --head-sha <headRefOid>
```

The guard defaults to `ATLAS_SESSION_STATE_FILE` when set, then falls back to
`SESSION_STATE.local.md` for legacy single-session worktrees.

If any check fails, stop and ask the operator. A PR in the same lane is
not automatically owned. A PR that "looks abandoned" is not owned. A PR
opened by another session is not owned unless the operator explicitly
reassigns it and the map is updated first.

Starting a new slice is gated the same way as touching a PR. Before you
scaffold a plan or pass `--lane` to `new_pr_plan.sh`, confirm that lane
matches the `current lane` in this session's state file. Opening a new PR in
another lane is the most common silent drift: parallel sessions are
indistinguishable in git, so another lane's slice looks exactly like your
own. If the slice belongs to a different lane:

1. Stop. Do not scaffold the plan or open the PR.
2. Confirm with the operator that the reassignment is intended, and update
   `current lane` in the map first.

Same product area is not the same lane. `clustering/raw-data` and
`PDF/delivery` are both deflection yet are two different sessions' lanes; a
clustering session that opens a PDF/delivery PR has drifted even though the
work is on-topic.

### 3a.2. PR-prep helpers

Four scripts remove the PR-shape and failed-push friction — use them
rather than hand-formatting:

- `bash scripts/new_pr_plan.sh <Slice> --lane <lane> --phase "<phase>"` —
  scaffolds `plans/PR-<Slice>.md` with the required 7 top-level sections,
  nested Problem-derived and Review Contracts, a `### Files touched`
  placeholder, and a zero diff-size table. Refuses to overwrite an existing
  plan without `--force`.
- `python scripts/sync_pr_plan.py plans/PR-<Slice>.md [base-ref]` —
  rewrites `### Files touched` and `## Estimated diff size` from the actual
  `git diff` (tracked vs merge-base plus untracked). Run after
  implementation; `--check` mode fails if the plan is out of sync
  (CI-gateable).
- `bash scripts/push_pr.sh <pr-body-file> [git-push-args]` — pushes with
  `ATLAS_CURRENT_PR_BODY_FILE` exported so the installed pre-push hook
  validates the same body. With the managed Atlas hook installed, the hook is
  the **single** local-review runner; if the managed hook is missing or
  intentionally skipped, the wrapper runs `local_pr_review.sh` before pushing.
  The wrapper rejects `--no-verify`.
- `bash scripts/open_pr.sh <pr-body-file> [gh-pr-create-args...]` — creates a
  PR for the current branch, or updates the existing PR body, using
  `--body-file - < <pr-body-file>`. Do not hand-roll
  `gh pr create/edit --body-file <path>`; under sandboxing `gh` can fail to
  open direct file paths. The shell redirect reads the file and `gh` receives
  the body on stdin. If the PR already exists, this wrapper updates only the
  body; use `gh pr edit` manually for title/base/label changes.

Flow: `bash scripts/new_pr_plan.sh` -> implement ->
`python scripts/sync_pr_plan.py` -> `bash scripts/push_pr.sh` ->
`bash scripts/open_pr.sh`.
Do **not** run a separate manual `local_pr_review.sh` immediately before
`push_pr.sh`; that duplicates the same mechanical bundle and burns context.
Use manual local review for ad hoc triage or when you are not about to push.

### 3b. Per-package guardrails

Touching a package under `extracted_*/` requires the package's audit
gauntlet locally before push. For `extracted_content_pipeline`:

```bash
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline   # if any synced files changed
```

CI runs `scripts/run_extracted_pipeline_checks.sh`; locally is faster
to triage.

### 3c. Local review before PR

Before opening or updating a PR, the builder runs the mechanical local
review bundle exactly once:

```bash
bash scripts/local_pr_review.sh
```

This is the fast path. It catches plan-shape, diff-size, file-claim,
MCP-doc, ASCII, plan/code, cross-session drift, and whitespace failures
before GitHub has to run anything. It also prints advisory cross-layer
caller hints for changed Python symbols so builder and reviewer can see
non-diff files that may need focused tests or inspection.

To make that automatic for this checkout, install the optional pre-push
hook:

```bash
bash scripts/install_local_pr_hook.sh
```

The installer refuses to overwrite unmanaged hooks unless `--force` is
passed. The installed hook can be bypassed intentionally with
`ATLAS_SKIP_LOCAL_PR_REVIEW=1 git push`.
When using `scripts/push_pr.sh`, that skip does not drop local review
entirely: the wrapper runs the bundle before the push because the hook will
skip.

After the mechanical bundle passes, hand the branch to a separate local
reviewer session for judgment review. The reviewer should read the plan
doc and diff locally, run any focused tests needed to verify the claims,
and return a verdict before the builder opens the GitHub PR.

For logic or shared-function PRs, the builder must read the cross-layer
caller hints from local review and either add focused caller-layer tests
or name why the referenced callers are unaffected. The hints are
advisory rather than blocking because outside references can be valid,
but silently ignoring them recreates the diff-only review gap.

Before calling a slice done, reconstruct the diff cold as if someone else
wrote it. Read the changed files and report, with `file:line` citations:

1. what each change actually does;
2. which Problem-derived contract requirement it traces to;
3. any contract requirement missing from the diff;
4. any touched module/behavior the contract said must not change; and
5. any change that does not trace to the contract.

Lead with gaps. Do not open, update, or call the PR done while any untraced
change, missing contract item, or forbidden touch remains. Put the cold
reconstruction in the PR body under `## Cold diff reconstruction` so the
reviewer can audit the builder's self-check.

GitHub Actions still runs the same wrapper after the PR opens. Treat CI
as the final enforcement layer, not the first reviewer.

For normal interactive PR work, after opening or updating a PR, the builder
does **not** wait for CI, automated review, or human review comments. Report
the PR URL, the local verification already run, and any immediately visible PR
status, then stop. The operator will tell the builder when checks are green or
when review comments are ready to inspect. Only resume PR inspection, comment
handling, or merge decisions after that operator signal.

### 3c.1. Long-running coding task PR watcher

The stop-after-open rule above applies to ordinary interactive slices. A
**long-running coding task** is different: if the operator explicitly assigns a
long-running/autonomous arc (for example a Fable-style builder session, a
multi-slice feature arc, or "continue through the approved slices"), the builder
must keep the PR moving instead of halting until the operator notices CI.

For long-running coding tasks, first record which builder surface owns the PR:

- **Claude Code native mode:** use Claude Code's PR subscription/review
  reactivity plus its 30-minute polling. Record that as the push/review-event
  hook and timer/polling path in this session's state file. Do not require a
  local systemd `atlas-pr-watch` timer for Claude Code unless the operator
  explicitly asks for the local watcher too.
- **Codex/local CLI mode:** a desktop notification or `atlas-pr-watch` run does
  not wake an agent by itself. True autonomous resume requires an external wake
  bridge that starts or resumes a Codex run with the watcher state and a prompt
  to read this session's state file, rerun merge guards, and act only on the
  owned PR. Without that bridge, the local watcher is a read-only state producer
  for the next active agent.

For long-running coding tasks, after each PR open or push:

1. Subscribe the session to its owned PR in this session's state file. Record the
   PR number, branch, head SHA, checks URL, review/reconciliation URL or
   commands, builder surface, wake bridge or native subscription path,
   ready-state handoff command, polling cadence, next wake/poll time, and the
   exact action to take when checks turn green or comments appear. If the
   operator grants standing merge authorization for the active builder, record
   the authorization source and scheduled-ready-only merge condition there too.
   The watcher process itself never receives merge authority.
2. Configure the wake path for that builder surface:
   - in Claude Code native mode, subscribe to the owned PR and poll every
     **30 minutes** using Claude Code's native behavior;
   - in Codex/local CLI mode, use an external wake bridge if one exists; if no
     bridge exists, record `Wake bridge: unavailable` and treat
     `atlas-pr-watch` output as a handoff for the next active agent only.
   A push/review-event hook must not reuse the scheduled green-confirmation
   command in a way that can grant merge permission, and an operator-only
   notification is not a builder wake hook.
3. Any local watcher/timer must exit fast after recording state. Do not keep an
   in-chat `sleep` loop or active polling process alive just to wait for green
   CI.
4. On each wake, refresh the PR head, CI/check status, review-thread status,
   live reconciliation, and merge-conflict state before deciding anything.
5. If checks are red or review comments are actionable, summarize the current
   blocker, fix the upstream/root cause inside the current slice, push, resolve
   fixed threads, update the PR body/reconciliation record, and leave the wake
   hooks armed for the next push/review/timer event.
6. If checks are still pending, record the last observed status and next timer
   wake time in this session's state file; do not ask the operator to babysit
   green and do not burn compute by waiting inside the chat turn.
7. Push/review-event wakes are attention-only. Even if a push/review-event wake
   observes green checks, record readiness and wait for the scheduled 30-minute
   Claude poll or Codex/local wake-bridge confirmation before merging.
8. If the scheduled poll/wake reports all required checks green, all
   review/reconciliation gates clean, and merge-conflict/mergeability state
   clean, the active builder follows the merge rules for the current arc. The
   review/reconciliation gates are **two, and both must be green** -- neither
   one alone is sufficient: `live-reconciliation` for the Codex/bot review
   (unaccounted bot threads red it) and `claude-review` for the Claude Code
   reviewer (a per-SHA commit status; absent or `failure` is not clean). See
   `docs/REVIEWER_MERGE_GATE.md`. A re-push resets `claude-review`, so a merge
   waits for the reviewer to re-review the new head. In Codex/local watcher
   mode, first surface that state with `scripts/report_pr_watcher_state.py`. If
   the operator has not authorized the active builder to merge for this arc, or
   has not made `claude-review` a required check, report readiness and wait.
   **Trust boundary (do not skip):** `claude-review` is a plain commit status,
   so any token with `status:write` on the repo can publish
   `claude-review=success`, including the builder if it shares the reviewer's
   GitHub identity. It is therefore a coordination-and-audit signal that keeps
   an honest builder from merging before review, NOT a defense against a builder
   that forges it. It becomes a real gate only when it is posted from a reviewer
   identity the builder does not have (a distinct GitHub App or bot token with
   `status:write` that the builder's token lacks). The operator MUST NOT grant a
   forge-capable builder merge authority on the strength of `claude-review`;
   provision the distinct reviewer identity first.
9. After merge, tear down only the owned worktree/branch, archive the plan as
   required, sync from `origin/main`, and continue to the next approved slice
   if the arc says to continue. The merge itself is the signal to pick up the
   next slice; do not start the next slice before the owned PR is merged or
   explicitly released by the operator.

The watcher/subscription state is mandatory compaction handoff data. A
restarted or compacted long-running session reads its session-state file
before doing anything else and resumes from the recorded PR state. Do not start
a new slice while the owned PR still has unresolved CI, review, reconciliation,
or merge state.

Watcher safety is enforced by `scripts/audit_pr_watcher_safety.py` in local
review. Local watcher executables and configs are status-only: truthy
auto-merge config, PR merge commands, delete-branch merge cleanup, or equivalent
merge behavior in watcher infrastructure is a blocking workflow defect.

### 3c.2. Overnight arc protocol

When the operator assigns an **unattended overnight arc** (a long-running
coding task expected to run to merged-or-blocked with zero operator contact),
the governing runbook is `docs/OVERNIGHT_ARC_WORKFLOW.md`. It wraps this
section's watcher rules with: a mandatory interactive **pre-flight** (task
readiness contract plus all clarifying questions asked while the operator is
still present), the **night-loop deltas** (never wait on the operator; defer
operator-only choices as issues; watcher-driven waiting via
`scripts/watch_owned_pr.sh`, which is status-only per the watcher-safety rule
above; bot-review round cap with the fail-closed-guard exception), the
**true-blocker escalation channel**, and a mandatory **morning report**. The
overnight baton (active arc task/contract, current slice, owned PR + head
SHA, watcher armed-state, morning-report accumulator) is compaction handoff
data per the `CLAUDE.md` compact instructions; a compacted overnight session
re-reads the runbook and verifies the baton against `git`/`gh` state before
proceeding.

### 3d. Thin-slice and hardening triage

Every plan names a slice phase in `Scope (this PR)`, and the PR body
and commit message repeat it. Use these standard phases:

| Phase | Use when |
|---|---|
| `Vertical slice` | Building the thinnest end-to-end product path that proves the real flow. |
| `Functional validation` | Proving the finished flow works on representative inputs and outputs. |
| `Robust testing` | Pushing scale, concurrency, failure, and integration edges after the flow works. |
| `Production hardening` | Closing survivability, observability, security, durability, and operational gaps found during validation or robust testing. |
| `Product polish` | Improving UX, copy, defaults, and ergonomics after the core behavior is proven. |
| `Workflow/process` | Changing repo workflow, review contracts, audits, or developer tooling rather than product behavior. |

The normal product order is `Vertical slice` -> `Functional validation`
-> `Robust testing` -> `Production hardening` -> `Product polish`.
Small corrections can happen out of order, but the plan must name why
the phase is appropriate now. If implementation changes the phase,
update the plan and PR body before review.

For product work, choose a vertical slice by default. A workflow/process,
autonomy-harness, audit, maturity-gate, or hardening slice must name the
specific vertical proof it unblocks, the safety/security/privacy/money risk it
fixes, or the recent product run that failed because this infrastructure was
missing. "This would make future autonomous coding cleaner" is not enough.

For a `Vertical slice`, build the thinnest end-to-end version that
exercises the real flow. A slice is done only when the builder
demonstrates the behavior with a concrete test, script, artifact, or
command output. When the slice introduces a new reachable surface, that proof
must go through the real entrypoint and assert an observable effect, not only a
unit-tested helper.

Only fix inline what the slice cannot function without. Required
inline fixes include:

- Issues that break the slice's stated real flow.
- Violations of this AGENTS contract, the plan, tests, or CI.
- Security issues introduced or exposed by the slice.
- Behavioral test coverage for a security or authorization guard the
  slice introduces or relies on.
- Output that would be misleading, false, or data-untruthful.
- Reviewer BLOCKER findings.

Everything else discovered while working gets appended to root
`HARDENING.md` and left out of the code diff. This includes
non-blocking error-handling gaps, missing validation, naming cleanup,
refactors, and edge cases. Each entry must include file/location,
one-line description, why it matters, rough effort (`S` / `M` / `L`),
category (`correctness`, `polish`, `tech-debt`, or `security`), and the
slice where it was found.

User-facing product shape changes are never "while here" work. They require
explicit operator consent before implementation, even when technically small.

Report parked work in the existing `Deferred` section of the plan doc
and in the PR body under `Parked hardening`. Final builder reports must
include what shipped, how it was demonstrated, and what was parked in
`HARDENING.md` and why.

At the start of each slice, scan `HARDENING.md` for entries touching
the same ownership lane or files. Fix only entries that are required for
the slice to function. For `Robust testing` and `Production hardening`
phases, promote relevant parked entries into the PR scope when they are
the reason the slice exists. Otherwise leave them parked and mention the
reason in `Deferred` if they were considered. Periodically drain or
promote stale entries into the debt register so `HARDENING.md` remains a
working queue, not an archive.

### 3e. Tests

Each PR ships its own tests. Acceptable test patterns:

- **Unit-level**: pure validators, parsers, helpers. Live in
  `tests/test_<package>_<module>.py`.
- **Integration-level**: exercise the **real service + adapter**. Fake
  only true external boundaries (see 3e.1). Live in
  `tests/test_<package>_<service>.py`.
- **Smoke**: thin wrappers that just check imports / wiring.

Locked-in regression tests for deferred follow-ups should name the
future slice in their docstring (e.g. *"after PR-Foo-V2 lands this
test is removed"*) so the test's lifetime is explicit.

Test quality is part of the contract. For meaningful logic changes, a
single trivial happy-path assertion is not enough. Add realistic
non-happy-path coverage proportional to the risk: negative cases,
malformed/sparse inputs, boundary values, varied producer shapes,
failure branches, and representative real-world fixtures. If the slice
intentionally ships only happy-path coverage, the plan's `Intentional`
or `Deferred` section must say why that is acceptable and what will
cover the missing cases.

Reachability is part of test quality. For a new runtime, workflow, UI, report,
billing, delivery, or public contract surface, include a thin smoke proof that
uses the real entrypoint and observes a result: a route response, rendered UI,
generated artifact, persisted row, queued job, sent/queued delivery, or gate
result. Unit-only proof is fine for pure helpers/refactors that add no new
surface, or when the plan explicitly defers wiring and names the follow-up.

**CI enrollment is part of test authoring — same PR.** A test only
protects the codebase if CI runs it. The Atlas Intel UI workflow
(`.github/workflows/atlas_intel_ui_checks.yml`) runs an **explicit
per-test list**, not a glob, so adding a `test:<name>` script to
`atlas-intel-ui/package.json` does **not** make CI run it. Any PR that
adds or renames a `test:*` script must add the matching
`run: npm run test:<name>` step to that workflow **in the same PR**.
The `extracted-checks` suite has an automated enrollment check that
fails on un-enrolled tests; the intel-ui workflow does not, so this one
is manual and has been dropped repeatedly. Reviewer/self check: grep the
workflow's run list for the new test name — `package.json` presence is
not CI execution.

### 3e.1. Real adapters by default — mock only true external boundaries

Use the **real** implementation in tests. Mock only at the outermost
boundary you genuinely cannot or should not hit in CI:

- **Mock-allowed (outermost seam only):** third-party network APIs
  (Stripe, Resend, the ATLAS service), the email/SMS *transport* (the
  sender), wall-clock and randomness, and other paid/external services.
- **Never fake the component whose behavior is under test.** If the test
  exists to prove a SQL filter, use a real database (provision the gate —
  `@pytest.mark.integration` + an env-gated skip + a Postgres service, the
  pattern in `atlas_content_ops_deflection_delivery_checks.yml`), not a
  fake pool. If it exists to prove a validator/projection, call the real
  validator/projection. Don't reach around the thing you're testing.
- **Don't assert on a mock's call arguments -- assert on the real
  adapter's observable state.** A test that checks the positional tuple or
  SQL string passed to `pool.execute(...)` is testing the mock, not the
  behavior: it breaks the moment the call shape changes and proves nothing
  about the outcome. Use the in-memory port adapter (e.g.
  `InMemoryDeflectionReportArtifactStore`) and assert the result -- "the row
  is marked paid", "a mismatched authorized amount is rejected" -- which is
  invisible to the method's signature.
- **Don't hand-author a fixture that is supposed to mirror generated or
  produced output.** Derive it (regenerate from the contract / capture
  from the real producer). A hand-kept copy is a second source of truth
  that silently goes stale.
- **All validators of the same shape must share one definition.** A smoke
  parser, a client guard, and a contract check that each re-encode "what a
  valid row looks like" will drift; import the one real check.

**Why (this is not style — it cost real rework):** a fake DB pool that
ignored the account filter let a scoping test pass while proving nothing,
forcing a real-Postgres redo a slice later; a smoke validator that re-spelled
the row shape laxer than the real client reported success on payloads the
client rejects; a hand-authored ground-truth whose `snapshot_safe_fields`
drifted from the producer left a `snapshot = projection(report)` guard
passing vacuously. Every one was a mock of something that did not need
mocking — extra code to keep in sync, and a new way to be wrong. A real
adapter can't drift from itself. Most recently (#1871): a `mark_paid`
signature change from 3 to 6 positional args broke four tests across three
files -- two raised `too many values to unpack`, two had stale call-arg
tuples -- because they asserted on a fake `_Pool`'s `execute` args instead of
a real store's state; the "fix" patched the fakes to track the new signature,
adding more mock to keep in sync. The real adapter would not have noticed it.

If a real adapter is genuinely too expensive for the slice, say so in the
plan's `Intentional`/`Deferred` and name what real-adapter coverage will
replace it (and the tracking issue), the way #1869 → #1872 did.

### 3f. Working with the manifest

Files listed in `<package>/manifest.json` under `owned` are
package-canonical -- the sync script does not overwrite them. Files
mapped from `atlas_brain/...` are the inverse: edits go to the
`atlas_brain/` source, then the sync script propagates. If unsure
which side a file lives on, run:

```bash
grep -B2 '"target": "<path>"' <package>/manifest.json
```

A `source` line means it's synced; absence (just a `target`) means
it's owned.

### 3g. Auditors must surface, never silently skip

Mechanical audit scripts must report unfamiliar input as drift unless
the skip is explicitly justified in code. Silent skips make the audit
look green while the thing it was supposed to validate disappears from
coverage.

Recent examples this rule is meant to prevent:

| What | Bad shape |
|---|---|
| Unknown `### <Name> MCP Server` headings disappearing from MCP tool-name coverage. | `if name not in HEADER_TO_FILE: continue` |
| Port claims with names not in the normalizer disappearing from MCP port coverage. | `if norm is None: continue` |
| Ports in `MCPConfig` missing from docs without any missing-in-doc check. | Only compare documented rows. |
| Env-var regexes dropping real names with digits, such as `ATLAS_MCP_B2B_CHURN_PORT`. | `[A-Z_]+` without a digit fixture. |

Preferred shape:

```python
norm = NAME_NORMALIZER.get(env_name)
claims.append((line_no, norm or env_name, port, "env"))
# main() then renders unknown names as DRIFT/UNKNOWN, not as skipped.
```

Safe skips are allowed only when the false-positive risk is named:

```python
# Unrelated markdown tables can share this row shape, so admit only
# rows whose first cell normalizes to a known server.
if norm is None:
    continue
```

If the false-positive risk cannot be stated in one sentence, the skip
is probably wrong.

### 3h. Auditors ship with fixture tests

Every new `scripts/audit_*.py` should ship with
`tests/test_audit_<name>.py` in the same slice. The fixture set should
cover:

1. Happy path: known-good input produces the expected OK state.
2. Parser-specific negative case: real-looking input that used to be
   missed, such as `ATLAS_MCP_B2B_CHURN_PORT=8062`.
3. Pathological rejection: absolute path, `..` traversal, malformed
   header, empty section, or an "Out of scope" heading that must not
   satisfy "Scope".

The audit script's `main()` is the contract; fixture tests lock the
parser behavior so a future small regex tweak cannot silently regress
to false-green output.

### 3i. Checkers prove their failure detection

Validators, contract checkers, evaluators, and gate predicates are only useful
when their failure branches are proven to fire. When a PR adds or changes code
whose job is to detect bad input, broken output, unsafe state, or contract
drift, the tests must prove the detector catches the failure, not only that the
happy path passes.

This rule applies to surfaces such as:

- `scripts/check_*.py`, `scripts/audit_*.py`, and `scripts/evaluate_*.py`
- extracted package validators and quality gates
- route/response contract checkers
- predicates that decide whether a gate should run
- helper branches that turn malformed input into errors or blockers

Required coverage shape:

1. **Each detection branch gets a negative fixture.** Feed input that violates
   exactly that rule and assert the specific error, blocker, non-zero exit, or
   false result.
2. **OR predicates get one-marker fixtures.** If a predicate can fire from
   `source`, `provider`, count fields, cluster fields, or any other marker,
   each marker gets a focused test where it is the only marker present.
3. **False-positive surfaces get rejection fixtures.** Broad parsers and type
   checks need tests for lookalikes: strings that are `Sequence`, empty lists,
   malformed-but-realistic JSON, unknown headings, missing keys, or unrelated
   route envelopes.
4. **Evaluator pattern changes prove precision.** If a PR adds or changes a
   denylist, regex, phrase matcher, or pattern list in an evaluator/checker,
   pair the bad-input fixture with at least one allowed near-miss fixture that
   should still pass. Example: a support-ticket claim detector that blocks
   "traffic suggests customers found the answer" also needs a neutral
   measurement sentence such as "use page views as one signal" that remains
   allowed. If the near-miss is intentionally omitted, the plan must name the
   risk, why it is safe for this slice, and the future PR that will add it.
5. **I/O checkers mock the transport, not the checker.** For network/file/DB
   checkers, test the real fetch/read path by mocking `urlopen`, file handles,
   DB cursors, or equivalent transport boundaries. Replacing the checker’s
   own fetch helper with a fake is not enough.
6. **Result-envelope drift fails closed.** If a checker returns `ok`,
   `errors`, `count`, `results`, or similar contract fields, tests must cover
   malformed or contradictory envelopes so missing error lists, count
   mismatches, or non-object payloads do not silently pass.

If a branch is intentionally not covered in the PR, the plan's `Intentional` or
`Deferred` section must name why it is safe to leave out and what future slice
will cover it. "Covered by the happy path" is not enough for detection logic.

### 3j. Class fixes need unseen probes

When a review finding names a defect class rather than one isolated example,
the builder must prove the class is fixed, not only the cited case. Before
claiming the fix complete:

1. Reproduce the reviewer-cited example.
2. Generate or write **5-10 same-class cases the reviewer did not mention**.
   Prefer property/parametrized tests that generate the cases; if that is not
   practical, use fresh fixtures and explain why they exercise the class.
   The cases must be diverse enough to exercise the class, not trivial
   near-duplicates of the cited example.
3. Add the proof to CI-facing tests or a committed artifact whenever possible.
4. If only the cited example was tested, say so explicitly and do not claim the
   class is fixed.

Hardcoding the reviewer's strings, values, paths, or exact examples is an R13
failure (`docs/REVIEWER_RULES.md`). A fix that can pass only the visible review
example is not done.

---

### 3k. Root-cause gate (no symptom fixes)

A fix-type PR -- one that resolves a bug, a defect, a review finding, or an
operator-reported quirk -- must establish the root cause **before it builds**.
In *Why this slice exists*, state:

1. The **root cause**: the underlying problem, not the surface symptom, the
   error message, or the reviewer's exact wording.
2. Whether this change **fixes the root or treats a symptom**. A symptom-only
   fix must justify why the root is deferred and link the follow-up that fixes
   it.

**Fix as far upstream as is correct.** The root cause of the current symptom may
itself be downstream of a deeper defect -- the "root" near the symptom is often
the *consequence* of something coded wrong further up the data/control flow.
Trace the chain to its origin and fix at the most-upstream point that is correct
and in safe scope; an upstream fix removes the defect for every downstream
consumer, not just this one. A patch one layer up from the symptom that leaves
the true upstream cause in place is still a symptom fix. If the true root is
further upstream than this slice can responsibly reach (shared-component blast
radius, another session's lane per §3a.1), **name the upstream root and link the
follow-up** -- do not silently patch downstream and call it root-cause.

A change that *fights another part of the pipeline* -- split-then-remerge,
widen-then-filter, add-then-strip, or building a harness/tool *around* a
deferred validation or decision step instead of taking the step -- is presumed
a symptom fix and must clear this bar explicitly.

Reviewers enforce this **at the plan stage, before code**: a fix-PR whose plan
treats a symptom without this justification is rejected before implementation,
not after a full review round. Chasing the symptom one layer at a time -- fix
the named case, ship, get bounced one layer deeper, repeat -- is the
tail-chasing this gate exists to stop.

### 3k.1. Guard class-closure (open-input guards)

For a guard / validator / sanitizer / classifier / gate / denylist /
parser-admission rule / safety or privacy checker whose input space is **open**
(free text, nested/recursive structures, producer-supplied keys/values), the
root cause of a reported leak or over-scrub is almost never the reported input
-- it is an **open default**: a per-input branch whose fall-through lands on the
unsafe verdict. Fixing the reported string closes nothing; the next string in
the same class is reported next round (the S6A privacy guard ran this loop 9+
rounds). The root-cause fix -- a fail-closed / evidence-gated choke point,
class-closure (not string-closure), a grammar- or evidence-derived property
test, plus the open-category exception and the asymmetric-safe default -- is
defined canonically in `docs/GUARD_CLASS_CLOSURE.md`. It is mandatory before
merge, it is the acceptance gate reviewers require before LGTM, and it
strengthens section 3j from "5-10 unseen cases" to "the generated class" for
open-input guards. The requirements are **not re-listed here**: that doc is the
single source, so this section cannot drift from it (three review rounds on #2077
were spent reconciling parallel restatements).

The reviewer enforcement of this gate lives in the guard boundary-probe section
of `docs/REVIEWER_RULES.md`; the scope caveat (documented neutral/data-column
families keep their admit policy -- the choke point governs the safety verdict,
not every field's text) and the open-category evidence-gated form both live in
`docs/GUARD_CLASS_CLOSURE.md`. A non-converging loop on one decision is governed
by 3k.2 below.

### 3k.2. Convergence circuit-breaker (stop instance-patching a seam)

Section 3k.1 is the fix for open-input guards; this is the process guardrail for
when a fix loop is NOT converging. On a PR where each push closes the reported
review threads but the next push opens a comparable count of **same-class**
findings on the **same file / decision** -- the thread count is flat or rising
over 3 consecutive pushes, not trending to zero -- the builder is
instance-patching a shared decision, not fixing it. This is distinct from the
bot-round *noise* cap (see 4a and `docs/OVERNIGHT_ARC_WORKFLOW.md`): there the
findings are formally-identical re-litigation of a green contract; here the
findings are real, and each patch shifts the boundary and exposes the adjacent
case.

**Counting the trigger under squash-amend.** Builder branches are amended into a
single commit (§1c), so "3 consecutive pushes" is not observable from the commit
graph -- a branch showing one commit may have absorbed a dozen review cycles.
Read the trigger as **3 consecutive review iterations** (a push and the review
round it draws), not 3 commits. How to count those iterations mechanically is
deliberately not specified here: it belongs in a `scripts/` checker with the
fixture tests 3h/3i require, not as untested procedure in prose (see #2198).

When this trips, the next push may NOT add another example-scoped patch (another
token, regex, vocabulary row, or oracle fixture). It must carry a **Decision-Seam
Analysis** in the plan / PR body:

1. **Name the one decision** all the open threads share (the seam) -- e.g. "the
   single admit/skip verdict for a transcript line."
2. **State why that decision is wrong** -- over-broad, under-broad, or an open
   category it cannot enumerate. If the recognizer itself is open, evidence-gate
   it per `docs/GUARD_CLASS_CLOSURE.md` (do not enumerate the category).
3. **Do exactly one of:** (a) fix the seam structurally with a stated default
   direction, and for asymmetric error costs the cheap-error default; (b) waive
   the bounded residual explicitly (<= status-quo, recorded in *Deferred*) and
   reconcile the threads as accepted-not-fixing; or (c) re-scope or park the
   slice. Adding the next instance patch is none of these and is rejected at
   review.

**Why:** Resolution Audit S6C (#2076) ran ~9 rounds and ~35 findings this way --
each round fixed the cited senders and the next round reported new same-class
senders, and every miss dropped a customer question. The round count alone was
not the signal (the bot-noise cap did not apply -- the findings were real); the
signal was that the findings were *the same decision re-litigated*. Naming the
seam and evidence-gating it converged in one push after nine that did not. The
builder pattern this catches -- close each cited example with the narrowest local
patch, never abstract to the generating decision -- is a recurring failure mode,
not a one-off.

### 3k.3. Open-input work: evidence-gate at plan time, one class per slice

3k.1 defines the closure bar and 3k.2 the non-convergence breaker; this is the
BEFORE-code gate that stops an open-input guard from generating the bar's
findings in the first place. It applies to a PR whose core change is a guard /
sanitizer / classifier / parser-admission / privacy-marker / boundary-detector
over free text or producer-supplied structure (the Resolution Audit S6 class).

1. **Plan-stage method gate (primary).** Before any code, the plan names three
   things: the single choke-point decision the guard makes; how ambiguous,
   unrecognized, or malformed input reaches the safe verdict *by default*; and
   the bounded evidence the recognizer keys on. A plan that instead lists shapes
   to handle -- a denylist, a case table, "handle these examples" -- is rejected
   at the plan stage, before the enumeration is written. Reject the enumerative
   method when it is cheapest to reject: on the plan, not after 50 review
   comments. The evidence-gate mechanics (fail-closed / evidence-gated choke
   point, evidence-keyed oracle, asymmetric-safe default, open-category form)
   live in `docs/GUARD_CLASS_CLOSURE.md`; the plan points to them, it does not
   restate them.
2. **One class per slice (surface bound).** An open-input PR handles ONE class or
   decision. A change that touches several open-input classes at once (HTML
   sanitizing AND quote/signature boundaries AND privacy-marker detection AND
   auto-reply detection) is split, one class per slice, so a single PR's blast
   radius is bounded to that class's findings.

**Why:** the S6 sanitizer arc. #2037 did the whole sanitizer in one +1262 change
and drew ~24 findings across four open-input classes at once, then paused. It was
sliced into S6A/A.1/B/C/E -- yet each slice still ran ~50 review threads (#2053,
#2061, #2046, #2076, #2054), because slicing bounds surface but each slice was
still built by enumeration. #2076 fell to a handful of rounds only when it
switched to evidence-gating; #2061 the same. So slicing alone distributes the
comments (~24 became ~244 across five PRs); evidence-gating is the lever that
collapses a class to one choke point. Gate the method at plan time (primary);
slice for surface (secondary). The general form -- pick the structural solution
over the enumerative one, which is also the smaller diff -- is why the
evidence-gated rewrite shrank both the code and the thread count together.

### 3k.4. Open-execution work: take the closed-surface component

3k.1 and 3k.3 govern an open **input** space -- free text, producer-supplied
structure. This section governs the other open space: **execution**. Here the
unbounded axis is not the input but the schedule: thread and process
interleavings, cancellation points, crash points, partial writes, descriptor and
path races. A durability or concurrency protocol, a lock/lease, a cache with a
coherence requirement, a rotation or retry state machine, and a crash-safe file
replacement all live here. None of them is a guard, sanitizer, or classifier, so
3k.1 and 3k.3 do not admit them by their own terms and an open-execution slice
can reach review ungated. That is the gap this section closes.

The failure signature is identical to the open-input one: each round reports a
real, previously-unenumerated schedule; the only available fix is another
special case; the PR body accretes those special cases as intent.

1. **Prefer the component whose surface is already closed.** At the fork, take
   the option whose failure cases someone else has already enumerated -- a
   database transaction over a hand-rolled file lease, an existing lock service
   over a bespoke `flock` protocol, an established library over a protocol
   written for this slice. Durable, concurrent, per-tenant state is a row, not a
   file. This is 3k.3's "pick the structural solution over the enumerative one"
   applied to execution.
2. **State the execution model, whichever option you took.** Selecting a
   closed-surface component narrows the seam; it does not remove it. A component
   supplies primitives, not your application invariant: a read-modify-write under
   an insufficient isolation level still loses updates, and a lock service does
   not define your crash or cancellation behavior. So every open-execution plan
   states the model **its own surface** admits, and for a selected component
   which of that component's guarantees close which part of the seam.

   This section deliberately does **not** carry the list of failure modes to
   cover. A fixed list is itself an enumeration, and the mode it omits is the one
   that bites: interleaving, cancellation, crash, duplicate and out-of-order
   delivery, and lease expiry with stale-holder fencing were each caught as a
   real omission during this section's own review, one per round, and the next
   one is on no list yet. Derive the modes from the surface instead. R8 in
   `docs/REVIEWER_RULES.md` is the floor for anything that retries or redelivers;
   a surface with leases, clocks, or partitions owes the modes those admit.

   The invariants must hold for **every** interleaving the model admits; anything
   not covered is stated as an explicit assumption, never just omitted. "Correct
   under a bounded set of interleavings" is the enumerative answer in formal
   dress -- the schedule left out of the set is the one that corrupts state. A
   plan that instead lists schedules to handle -- "drain cancellation here",
   "fsync there", "reject FIFOs" -- is the same enumeration without even the
   model, and is rejected at the plan stage before it is written.
3. **Name what you rejected -- only if you hand-rolled.** A slice that took the
   closed-surface component has no rejected component to name and must not be
   asked to invent one. A slice that hand-rolled states which component it
   rejected and why it does not fit, on top of the model required by 2.
4. **One execution surface per slice.** Do not land a concurrency/durability
   subsystem inside a PR whose stated purpose is something else (a privacy
   boundary, a feature, a migration). Split it: the feature ships against the
   closed-surface component; the subsystem ships alone if it is genuinely needed.

Reviewer enforcement lives in the open-execution row of the path-trigger table in
`docs/REVIEWER_RULES.md` (R8 + R2). Like the guard row it is deliberately
prose-only -- no path glob identifies a durability protocol, since the same
module paths carry ordinary code -- so `scripts/audit_review_rules_triggered.py`
surfaces it as an explicit advisory finding per 3g rather than deriving it. A
plan whose slice is open-execution work and whose Review Contract omits R8 fails
that surfacing. The reviewer checks 1, 2, and 4 for every such slice, and 3 only
where the slice hand-rolled.

**Why:** #2184 (scoped mailbox binding) is the case. Its stated purpose was an
authorization boundary -- bind each CRM business context to one mailbox. It also
grew `ScopedGmailTokenStore`, a 15-method cross-process durability protocol
(`flock` lease, double fsync, `fstat`-verified no-follow descriptors,
FIFO/socket/symlink rejection, monotonic generation ancestry with cycle
detection, repeated-cancellation draining at three call sites) -- while
the project already runs Postgres with a repository layer, where this state is a
row and the ordering it hand-rolls is a transaction. (That alternative needs new
schema: `atlas_brain/storage/repositories/business_context.py` writes a fixed
business-profile column list with no token or binding field, and its `get` and
`upsert` are separate pool operations, not one compare-and-swap. "Add a column
and a transaction" is still a far smaller and better-tested surface than a
bespoke cross-process lease -- but it is a build, not a drop-in.) Because none of
this is a guard,
3k.1/3k.3 never applied; it reached 12 bot rounds, and its *Intentional* section
now records schedule patches as design intent ("the FIFO nonblocking regression
starts its deadline only after the spawned reader has completed
interpreter/import setup"). Review cost is not predicted by diff size --
#2117 (+2813) took 3 rounds, #2181 (+2569) took 20 -- it is predicted by whether
the slice hand-rolled an open surface.

### 3l. PR fix mode (constrain the fix loop)

A **fix loop** -- iterating on red CI or review comments on an already-open PR
-- is where sessions burn the most time and tokens: broad exploration, edits to
files outside the real failure source, and re-orientation after every
compaction. Before editing in a fix loop, record a **fix baton** in
this session's state file (the `PR Fix Mode` block) capturing the failure source,
the **allowed-files set**, and a **max-files budget**.

- **Stay inside the allowed set.** The allowed files are the failure source you
  identified, not "everything the symptom touches." Touching a file outside the
  set is presumed scope creep. Declare `Max files: N` in the plan's *Scope* to
  have the files-touched audit (`scripts/audit_plan_doc_files_touched.py`)
  enforce the budget at pre-push/CI -- the PR fails if more than N files change.
- **Widening the set is a root-cause decision.** If the fix genuinely needs an
  upstream file, name the upstream reason in the baton and the plan **before**
  editing it (this is the §3k trace, not a drive-by). Do not silently grow the
  diff.
- **One judgment pass, no auto-loop.** Bot findings are advisory inputs you
  disposition deliberately (resolve or waive with a reason); there is no
  "address every comment" reflex (§4a.1).
- **The baton is the compaction handoff.** Keep the current failing
  check/comment, the last useful log finding, the next exact action, and
  do-not-redo notes current, so a post-compaction resume continues instead of
  re-exploring. Update it before and after each push.

---

## 4. Reviewer workflow

The reviewer's job is **not** "review the code" -- it is to **disposition the
review matrix: every acceptance criterion in the PR's Review Contract, and every
rule in `docs/REVIEWER_RULES.md`.** Every rule gets a verdict (pass / fail /
not-verified / n-a-with-reason); the path-trigger table sets how deeply each is
probed, not which appear. Every finding cites a rule ID (R1-R14) and maps to a
verdict level (BLOCKER / MAJOR / NIT). The rule matrix and AI reconciliation go
in the §2a template.

A dispositioned matrix is a **complete** review -- that is the stopping
condition, and "violates none of the rules" is not, being a universal negative
no non-trivial diff can discharge. Complete is not approved: `not-verified` or
`could-not-determine` entries end the search but block LGTM. See **Review
completion** in the pack.

### 4a. Independent verification

**Reconstruct the PR independently from the diff -- never review it against its
description** (`docs/PR_RECONSTRUCTION_PROTOCOL.md`, mandatory for every review).
Build two INDEPENDENT reconstructions, then compare them. They are independent,
not sequential: derive (2) purely from the problem, never from the diff, so the
diff cannot anchor it (the challenger pass -- easiest if you derive it before
opening the diff, but the binding requirement is only that the diff never shapes
it).

1. **What the diff actually does** -- read the diff alone and state it change by
   change, in your own words. The code is ground truth; the description, commit,
   and title are unverified claims.
2. **What a correct fix should do** -- from the problem and the Review Contract
   alone (the challenger pass), derive which files *should* change and which
   tests *should* exist, never letting the diff shape the answer.
3. **Compare** those two against what the description claims and report every
   gap: diff != description; diff != correct fix (wrong / incomplete / symptom
   patch); diff changes unmentioned things.
4. **Cite checkable evidence for every claim**: `file:line` for code/content
   claims, or a named non-file artifact such as command output, CI run/job,
   generated artifact, or PR metadata. Sort each into confirmed / contradicted /
   could-not-determine (never confirmed without evidence), and lead with the
   gaps (record them in the §2a template's Reconstruction block).

Don't trust the PR description's claims; reproduce them. The
reviewer should:

1. Record the exact reviewed head SHA from the checked-out PR head.
2. Re-run the named verification commands.
3. Spot-check the plan's invariants at the actual file:line pointed
   to in the diff.
4. Sweep for missed call sites with grep patterns more reliable than
   the PR's claim (multi-line constructions, kwargs split across
   lines, etc.).
5. Walk EVERY rule R1-R14 and record Pass/Fail/Not-Verified/N-A for each. The
   trigger table in `docs/REVIEWER_RULES.md` sets probe depth, not matrix
   membership -- an untriggered rule is still recorded, and N-A carries a
   reason. A Not-Verified entry ends the search but blocks LGTM.
6. Apply R14: every claim used in the verdict must be backed by checked-out
   code, a caller/test/artifact spot-check, or a "not verified" note with the
   reason. **No LGTM from the PR story alone.**

### 4a.1. AI-finding reconciliation (mandatory before LGTM)

External review bots (Codex, Copilot) post raw comments outside the
BLOCKER/MAJOR/NIT taxonomy. They are **advisory inputs to a judgment session,
never auto-applied** -- a bot false-positive applied blindly turns correct work
into incorrect work, so there is no "auto-address all comments" loop. **A
reviewer may not issue LGTM until every AI finding is either fixed or explicitly
waived with a reason in the PR body.** The machine owns mechanical issues; the
human owns intent mismatch, product logic, architecture, risky assumptions, and
missing tests.

This rule is enforced from both sides. Locally,
`scripts/audit_ai_reconciliation.py` checks the PR body's reconciliation record
is internally resolved. In CI, the `AI Reconciliation (live)` workflow
(`scripts/check_ai_reconciliation_live.py`) reads the live Codex/Copilot review
threads and fails when the body records the findings as all fixed/waived (or
carries no record) while bot threads are still open -- closing the
"omitted a real open finding" loophole the local check cannot see.

### 4b. Verdict frugality

Post **one** review per push. Don't comment on the PR while CI is
in-flight unless asked. Don't rubber-stamp -- a bare "LGTM" with no
verification is worse than no comment.

### 4c. NIT discipline

NITs should be marked skip-worthy explicitly when they are. The
builder applies 1-line NITs; ignores style/naming/comment NITs that
require a follow-up commit unless the reviewer specifically calls
out "this should be fixed."

### 4d. Audit checklist

Before LGTM, the reviewer confirms:

- [ ] CI green (extracted-checks ✅, Vercel ✅).
- [ ] Plan doc has all 7 required sections, including the `### Review
      Contract` block in Scope (acceptance criteria, affected surfaces, risk
      areas, triggered rule IDs).
- [ ] EVERY rule R1-R14 (`docs/REVIEWER_RULES.md`) is recorded
      Pass/Fail/Not-Verified/N-A -- the path-trigger table sets probe depth,
      not which rules appear, so an untriggered rule is still recorded (Pass,
      or N-A with a reason). Every Fail at BLOCKER level cites file:line, and
      no Not-Verified entry remains when issuing LGTM.
- [ ] R14 is recorded Pass/Fail/N-A. The verdict names the reviewed head SHA,
      changed code inspected, caller/test/artifact spot-checks, and any claims
      not verified. No LGTM from PR/body/builder claims alone.
- [ ] Every AI (Codex/Copilot) finding is fixed or explicitly waived with a
      reason in the PR body (§4a.1). No unresolved bot comments at LGTM.
- [ ] Plan and PR body name a `Slice phase`, and the diff matches that
      phase. The squash commit message carries the phase from the PR body
      at merge.
- [ ] Diff size matches the plan's estimate (or the overage is
      justified in **Why**).
- [ ] No regressions in the named test sweep.
- [ ] For shared-function PRs, cross-layer caller hints were inspected
      and the verdict names any caller-layer tests or unaffected
      references.
- [ ] For checker/evaluator/validator/gate PRs, each detection branch
      has a focused negative fixture, OR predicates have one-marker
      fixtures, and false-positive surfaces are covered or explicitly
      deferred with a named future slice. If the PR adds or changes
      denylist/regex/phrase-matcher/pattern-list detection, the coverage
      includes an allowed near-miss fixture or the plan names the future PR
      that will add it.
- [ ] For guard-shaped PRs (guards, validators, caps, classifiers, gates,
      sanitizers, denylists, parser admission rules, or safety checkers), the
      verdict includes `boundary-probe: <what applied + result>` before LGTM.
      Missing proof is BLOCKER for security, billing, data deletion,
      customer-visible output, or CI/release gates; otherwise it is at least
      MAJOR.
- [ ] No drift from the plan's stated scope (no scope creep, no
      "while I was at it" cleanups beyond the slice's contract).
- [ ] Defensible trade-offs are explained in **Intentional**.
- [ ] Deferred items have a clear next-PR home.

---

## 5. Within-session agent routing

**Reasoning stays in main; retrieval goes to a subagent. Synthesis
stays with whoever has to act on the answer (almost always main).**

Applies to both builder and reviewer sessions. The point: stretch
the weekly token budget without pushing judgment work to a model
that can't make judgment calls.

### 5a. The decision

Two questions before opening a file or kicking off a search:

1. *Will I edit this file in-session?* -> Main, direct `Read` (need
   exact line numbers).
2. *Does this need judgment* (quality, design trade-off,
   root-cause)? -> Main only.

If neither, route by shape:

| Shape | Where | Why |
|---|---|---|
| Bounded read-only scouting/checking with a small known surface and no edits planned | Spark subagent (if available), else `Explore` | Lightweight retrieval/checking without pulling raw context into main |
| Read-only, >400 lines, no edits planned | `Explore` subagent | Pure retrieval; summary lands in main context, raw file does not |
| Reading 3+ files just to orient | `Explore` subagent | Width without depth -- the subagent's strength |
| "Find every caller of X" / "where is Y defined" | `grep`/`find` via Bash | Regex match, no LLM needed |
| Scaffold multi-file boilerplate (tests, configs, fixtures) | `general-purpose` subagent | Write-capable, separate context window |
| Architectural decision / debugging / refactor plan | Main only | Needs holistic judgment |
| Code review verdict | Main only | Verdict requires judgment, not a summary |

The boundary that matters most: **judgment vs lookup.** "Where is
the displacement edge schema?" -- lookup, delegate. "Is this
displacement edge schema right?" -- judgment, do it yourself.

### 5b. Parallelism

Independent retrievals run as parallel subagents in a single
message -- the main session waits once for N answers instead of N
times for one each. We used this pattern during the CLAUDE.md
refresh (three `Explore` agents in parallel mapped churn signals,
extracted packages, and planned products); without it the same work
would have cost N round-trips of main-context overhead.

### 5c. Spark and lightweight worker relationships

Spark is the preferred lightweight subagent for bounded read-only scouting and
checking when it is available: known files, narrow searches, review-thread
summaries, plan-vs-diff checks, backlog scans, and other retrieval tasks where
the answer can be a compact list of facts. Do not use Spark for edit-target
reads that need exact file context in main, architectural decisions,
root-cause calls, review verdicts, Git/GitHub mutations, or final user-facing
synthesis. If Spark is unavailable, or the task needs broader multi-file
orientation, use `Explore`.

If a `claude-coworker-model`-style worker LLM is installed locally
(Kimi / DeepSeek / Ollama via OpenRouter), it slots in as a
**cheaper** retrieval channel for cases where an `Explore`
or Spark subagent is overkill -- one big file, no reasoning needed, no other
files to cross-reference. The decision table above is unchanged;
just add a row:

| Shape | Where | Why |
|---|---|---|
| Deep retrieval of one large file, no cross-refs | Worker LLM (if installed), else `Explore` | Worker is cheapest; `Explore` is the in-tree fallback |

The worker never replaces Spark or `Explore` for multi-file orientation, and no
worker replaces the main session for judgment.

### 5d. Routing anti-patterns

- **Asking a subagent for a judgment call** ("is this design
  right?"). The subagent doesn't have full session context and the
  answer is just deferred judgment the main session has to redo.
- **Sequencing N orthogonal `Explore` calls** instead of firing
  them in parallel.
- **Using `Explore` on a <400-line file you're about to edit
  anyway.** Just `Read` it directly.
- **Letting a subagent compose the final user-facing answer.**
  Synthesis is a main-session job.
- **Routing exact-line edits through a worker.** Edits need a
  precise file:line citation; a summary won't have one.

---

## 6. Anti-patterns

Things that should **never** appear in a PR or review:

- **Drive-by formatting changes** unrelated to the slice. Format-only
  diffs ship as their own slice if needed.
- **Plan doc that arrives in a follow-up commit.** Plan and
  implementation ship together.
- **"While I was here..." cleanups** that aren't required for the
  slice to function. Add a `HARDENING.md` entry and move on.
- **Bypassing CI with `--no-verify`** unless the user explicitly
  authorizes.
- **Reviewer running the builder's commands without spot-checking
  the diff.** A green test sweep doesn't prove the diff matches the
  plan.
- **Builder applying every NIT without judgment.** NITs marked
  skip-worthy are skipped. Apply only the 1-line / unambiguous ones.

---

## 7. References

- `AUDITOR_PROMPT.md` -- cross-cutting auditor prompt
  (canonical / integration / scope / debt). Run before any non-trivial
  build session.
- `docs/CURRENT_PRODUCT_DISCIPLINE.md` -- active product discipline, vertical-first,
  hardening-deferral, and product-shape consent rules.
- `BUILD_SPEC.md` -- deprecated historical context; do not use as the current
  product roadmap or definition of done.
- `CANONICAL.md` -- which implementation is the real one.
- `INTEGRATION_MAP.md` -- what's wired to what.
- `CONTEXT.md` -- historical/session notes and known debt; verify before using
  as current product state.
- `CLAUDE.md` -- project-level Claude Code guidance.
- `HARDENING.md` -- parked non-blocking hardening discoveries from
  thin slices.
- `plans/` -- per-slice plan docs (one per PR).

---

## 8. Bootstrapping a fresh reviewer session

When the reviewer session is killed, expired, or otherwise needs to
be re-seeded, paste the block below into a fresh Claude Code session
(adjust the trailing PR list to match what's actually open). The
session will arrive with everything it needs to start auditing
immediately.

```
You are the reviewer Claude session for the canfieldjuan/atlas
repository. A separate builder session opens PRs; you audit them
and post one consolidated review per push with a verdict at:

- BLOCKER: correctness, security, contract break, or CI red.
- MAJOR: architectural / scope / pattern concern, or a proven defect
  whose blast radius does not warrant blocking.
- NIT: style / naming / comment polish (mark explicitly skip-worthy
  when applicable).
- LGTM: all gates green, no remaining concerns.

Read AGENTS.md at the repo root before your first review. It
defines:
- The required plan-doc shape (Why / Scope / Mechanism / Files
  touched / Intentional / Deferred / Verification) at
  plans/PR-<Slice-Name>.md.
- The PR body / commit message conventions that mirror the plan.
- The thin-slice rule and `HARDENING.md` parking contract for
  non-blocking discoveries.
- The reviewer verification template (sections 2a + 4d).
- The 400 LOC diff budget and how to handle overage.
- Anti-patterns that should never appear in a builder PR.

Read `docs/REVIEWER_RULES.md` before your first review. Your job is
not "review the code" -- it is to disposition the review matrix: every
acceptance criterion in the Review Contract, and every rule R1-R14, each
pass / fail / not-verified / n-a-with-reason. That matrix is your stopping
condition; an unresolved entry ends the search but blocks LGTM. Cite a
rule ID on every finding.

Read AUDITOR_PROMPT.md for the cross-cutting audit checks
(canonical / integration / scope / debt). Apply both lenses.

For each PR you review:

1. Reconstruct independently (`docs/PR_RECONSTRUCTION_PROTOCOL.md`,
   mandatory): build two independent reconstructions -- what a correct
   fix should do (from the Review Contract and problem, never shaped by
   the diff) and what the diff actually does (read the diff alone) --
   then report every gap (diff != description; diff != correct fix; diff
   changes unmentioned things), cited `file:line`, sorted confirmed /
   contradicted / could-not-determine, gaps first. Do not anchor on the
   builder's summary.
2. Reproduce the named verification commands from the PR body.
   Don't trust claims; re-run them. Spot-check the plan's
   invariants at the actual file:line cited in the diff.
3. Record the reviewed head SHA and apply R14: every claim in the verdict
   must be backed by checked-out code, a caller/test/artifact spot-check, or a
   "not verified" note with the reason. No LGTM from PR/body/builder claims
   alone.
4. Sweep for missed call sites with grep patterns more reliable
   than the PR's claim (multi-line constructions, kwargs split
   across lines).
5. Record Pass/Fail/Not-Verified/N-A for EVERY rule R1-R14
   (`docs/REVIEWER_RULES.md`) -- the trigger table sets probe depth, not which
   rules appear; cite file:line on any Fail. Do not LGTM with a Not-Verified
   entry outstanding.
6. Reconcile AI findings: do not LGTM until every Codex/Copilot
   finding is fixed or explicitly waived with a reason in the PR body.
7. Confirm CI is green (extracted-checks x2 + Vercel) before
   issuing LGTM.
8. Mark NITs as skip-worthy explicitly when they are. The builder
   applies 1-line / unambiguous NITs; ignores style/naming/comment
   NITs unless you specifically call out "this should be fixed."
9. Sign your review with: `_Generated by [Claude Code](
   https://claude.ai/code)_`.

The package under active iteration is `extracted_content_pipeline`.
Its audit gauntlet is:

```bash
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash scripts/run_extracted_pipeline_checks.sh   # full CI mirror
```

The package's manifest at
`extracted_content_pipeline/manifest.json` distinguishes
package-owned files (entries with only a `target`) from
synced-from-`atlas_brain/` files (entries with both `source` and
`target`). Synced files cannot be edited in
`extracted_content_pipeline/` directly; the source lives in
`atlas_brain/`. The sync script propagates.

Recently merged context (so you're not starting cold):
- PR #396 PR-Blog-Topic-Per-Call: per-call topic kwarg threads from
  request.inputs through the dispatcher into the blog skill prompt.
- PR #397 PR-Describe-Control-Surfaces-Cache: GET
  /content-ops/control-surfaces hot path caches the static
  catalog payload at import; only execution flags are computed
  per request.
- PR #398 PR-Campaign-Config-V2 (BREAKING): drops the legacy
  `channel` field from CampaignGenerationConfig + the
  `or (self._config.channel,)` fallback. 4 in-tree wrappers
  normalize `channel` -> `channels` tuple before dataclass
  construction.
- PR #399 PR-Blog-Reasoning-Parity: brings
  BlogPostGenerationService to constructor parity with the other
  4 generators on the CampaignReasoningContextProvider port.
- PR #400 PR-Agents-Md-Framework: this file.

Open PRs as of this bootstrap: <fill in via the GitHub MCP tools or
mcp__github__list_pull_requests at session start>.

Standby for `<github-webhook-activity>` events; investigate each
in turn.
```

If the user prefers to keep AGENTS.md leaner, the reviewer prompt can
also live at `REVIEWER_BOOTSTRAP.md` and be invoked by a one-liner:
*"Read REVIEWER_BOOTSTRAP.md, then standby."* Either shape works;
pick one and keep it.
