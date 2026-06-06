# AGENTS.md — Atlas multi-agent workflow

Atlas uses **two coordinated Claude Code sessions** for non-trivial
work:

1. **Builder session** — drafts the plan, writes the code, opens the PR.
2. **Reviewer session** — audits each PR independently and posts a
   verdict (BLOCKER / MAJOR / NIT / LGTM).

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

## 1. PR shape

Every non-trivial change ships as a single PR with the following
artifacts:

### 1a. Plan doc (`plans/PR-<Slice-Name>.md`)

Required sections, in this order:

| Section | Purpose |
|---|---|
| **Why this slice exists** | What's broken / what's missing / what audit item this closes. Tie to a prior plan, audit finding, or a concrete user request. |
| **Scope (this PR)** | The narrow surface this PR touches. Start with an `Ownership lane: <lane>` line, then a `Slice phase: <phase>` line, then a numbered list of intent and a "Files touched" subsection. |
| **Mechanism** | Short prose (and code stub if helpful) explaining *how* the change works -- enough that the reviewer doesn't have to reverse-engineer it from the diff. |
| **Intentional** | Things that look wrong but aren't -- explicit trade-offs and rejected alternatives ("no `warnings.warn` shim because ..."). Saves reviewer cycles. |
| **Deferred** | Things explicitly punted to a follow-up slice. Each item should name the future PR or describe what would unlock it. Include "Parked hardening: none" or list the `HARDENING.md` entries added by this slice. |
| **Verification** | The specific commands the builder ran locally + their pass counts. Reviewer reproduces. |
| **Estimated diff size** | LOC budget; flag if approaching 400 LOC. |

### 1b. PR body

Mirror the plan-doc framing in the PR description:

```
Plan: plans/PR-<Slice-Name>.md
Slice phase: <phase>

<one-paragraph why>

## Intentional
- ...

## Deferred
- ...

## Parked hardening
- None. (or: `HARDENING.md` entry title and why it was parked)

## Verification
- ...

## Diff size
N files, +X / -Y
```

### 1c. Commit message

Same `Plan: ...` and `Slice phase: ...` lead lines + Intentional /
Deferred / Parked hardening sections as the PR body. Squash-merge
collapses to one canonical commit at merge time.

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
| **MAJOR** | Architectural / scope / pattern concern. Strong recommendation but not auto-block. | Fix in this PR if the fix is small; otherwise discuss before deferring. |
| **NIT** | Style, naming, comment polish. Skip-worthy. | Apply if 1-line; skip otherwise. The reviewer should mark NITs as skip-worthy explicitly. |
| **LGTM** | All gates green, no remaining concerns. | Merge. |

### 2a. Reviewer's verification template

The reviewer should produce something like:

```
**Verification (independent):**
1. <claim from PR description> -- verified via <command>
2. <invariant from Mechanism> -- confirmed at <file:line>
3. ...

**Plan-doc compliance:** Why / Scope / Mechanism / Files touched /
Intentional / Deferred / Verification -- matches AGENTS.md framework.
Slice phase is named and matches the PR's scope. Parked hardening is
named in Deferred or explicitly marked none.

**Defensible trade-offs (no action needed):**
- <decision> -- <why it's the right call>

**<N> NITs (skip-worthy):**
1. ...

LGTM. (or: BLOCKER -- ...)
```

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

### 3a.1. Session ownership map

Every builder session must maintain a local `SESSION_STATE.local.md`
at the repository root, using `docs/SESSION_STATE_TEMPLATE.md` as the
shape. This file is ignored by git because it is volatile session
state, but it is mandatory working context.

Update the map:

- at session start or after compaction/restart reorientation;
- before opening a PR;
- after pushing a PR update;
- after merging a PR;
- before handing back to the operator.

The map must name the current lane, current task, owned active PR
number/title/branch/plan/head SHA when one exists, PRs this session may
touch, PRs this session must not touch, and the last safe action.

Before inspecting comments, pushing updates, closing, or merging any
PR, the builder must verify all of the following:

1. `gh pr list --state open` has been checked in this resume window.
2. `git log --oneline -15 origin/main` has been checked for already
   landed work.
3. The target PR is listed in `SESSION_STATE.local.md` under "Owned
   Active PR" or "PRs This Session May Touch".
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

If any check fails, stop and ask the operator. A PR in the same lane is
not automatically owned. A PR that "looks abandoned" is not owned. A PR
opened by another session is not owned unless the operator explicitly
reassigns it and the map is updated first.

### 3a.2. PR-prep helpers

Three scripts remove the PR-shape and failed-push friction — use them
rather than hand-formatting:

- `bash scripts/new_pr_plan.sh <Slice> --lane <lane> --phase "<phase>"` —
  scaffolds `plans/PR-<Slice>.md` with the required 7 sections, a
  `### Files touched` placeholder, and a zero diff-size table. Refuses to
  overwrite an existing plan without `--force`.
- `python scripts/sync_pr_plan.py plans/PR-<Slice>.md [base-ref]` —
  rewrites `### Files touched` and `## Estimated diff size` from the actual
  `git diff` (tracked vs merge-base plus untracked). Run after
  implementation; `--check` mode fails if the plan is out of sync
  (CI-gateable).
- `bash scripts/push_pr.sh <pr-body-file> [git-push-args]` — runs
  `local_pr_review.sh` with the body file, then pushes with
  `ATLAS_CURRENT_PR_BODY_FILE` exported so the installed pre-push hook
  validates the same body. The wrapper does **not** add `--no-verify`;
  callers must not pass `--no-verify` through the forwarded push args.

Flow: `bash scripts/new_pr_plan.sh` -> implement ->
`python scripts/sync_pr_plan.py` -> `bash scripts/push_pr.sh`.

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
review bundle:

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

After the mechanical bundle passes, hand the branch to a separate local
reviewer session for judgment review. The reviewer should read the plan
doc and diff locally, run any focused tests needed to verify the claims,
and return a verdict before the builder opens the GitHub PR.

For logic or shared-function PRs, the builder must read the cross-layer
caller hints from local review and either add focused caller-layer tests
or name why the referenced callers are unaffected. The hints are
advisory rather than blocking because outside references can be valid,
but silently ignoring them recreates the diff-only review gap.

GitHub Actions still runs the same wrapper after the PR opens. Treat CI
as the final enforcement layer, not the first reviewer.

After opening or updating a PR, the builder does **not** wait for CI,
automated review, or human review comments. Report the PR URL, the
local verification already run, and any immediately visible PR status,
then stop. The operator will tell the builder when checks are green or
when review comments are ready to inspect. Only resume PR inspection,
comment handling, or merge decisions after that operator signal.

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

For a `Vertical slice`, build the thinnest end-to-end version that
exercises the real flow. A slice is done only when the builder
demonstrates the behavior with a concrete test, script, artifact, or
command output.

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
- **Integration-level**: services + ports with fakes. Live in
  `tests/test_<package>_<service>.py`.
- **Smoke**: thin wrappers that just check imports / wiring.

Locked-in regression tests for deferred follow-ups should name the
future slice in their docstring (e.g. *"after PR-Foo-V2 lands this
test is removed"*) so the test's lifetime is explicit.

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

---

## 4. Reviewer workflow

### 4a. Independent verification

Don't trust the PR description's claims; reproduce them. The
reviewer should:

1. Re-run the named verification commands.
2. Spot-check the plan's invariants at the actual file:line pointed
   to in the diff.
3. Sweep for missed call sites with grep patterns more reliable than
   the PR's claim (multi-line constructions, kwargs split across
   lines, etc.).

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
- [ ] Plan doc has all 7 required sections.
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

### 5c. The Kimi worker model relationship

If a `claude-coworker-model`-style worker LLM is installed locally
(Kimi / DeepSeek / Ollama via OpenRouter), it slots in as a
**cheaper** retrieval channel for cases where an `Explore`
subagent is overkill -- one big file, no reasoning needed, no other
files to cross-reference. The decision table above is unchanged;
just add a row:

| Shape | Where | Why |
|---|---|---|
| Deep retrieval of one large file, no cross-refs | Worker LLM (if installed), else `Explore` | Worker is cheapest; `Explore` is the in-tree fallback |

The worker never replaces `Explore` for multi-file orientation or
the main session for judgment.

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
- `BUILD_SPEC.md` -- what Atlas is, P0/P1/P2 priorities, definition
  of done.
- `CANONICAL.md` -- which implementation is the real one.
- `INTEGRATION_MAP.md` -- what's wired to what.
- `CONTEXT.md` -- session notes, known debt.
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
- MAJOR: architectural / scope / pattern concern.
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

Read AUDITOR_PROMPT.md for the cross-cutting audit checks
(canonical / integration / scope / debt). Apply both lenses.

For each PR you review:

1. Reproduce the named verification commands from the PR body.
   Don't trust claims; re-run them. Spot-check the plan's
   invariants at the actual file:line cited in the diff.
2. Sweep for missed call sites with grep patterns more reliable
   than the PR's claim (multi-line constructions, kwargs split
   across lines).
3. Confirm CI is green (extracted-checks x2 + Vercel) before
   issuing LGTM.
4. Mark NITs as skip-worthy explicitly when they are. The builder
   applies 1-line / unambiguous NITs; ignores style/naming/comment
   NITs unless you specifically call out "this should be fixed."
5. Sign your review with: `_Generated by [Claude Code](
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
