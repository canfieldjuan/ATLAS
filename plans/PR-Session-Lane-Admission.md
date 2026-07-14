# PR-Session-Lane-Admission

## Why this slice exists

PR #2097's accepted plan identifies session-lane admission as the next
merge-ordered workflow slice. The operator instructed this session to continue
after that PR merged. AGENTS.md requires a builder to verify that a new plan's
ownership lane matches its local session state before scaffolding, but the
runtime scaffold does not read session state at all: it accepts any `--lane` or
emits `TODO-ownership-lane`.

### Problem-derived contract

- Root cause: the only pre-PR entrypoint that creates a plan and claims an
  ownership lane (`new_pr_plan.sh`) has no session-state input, while the
  template calls the intended field `Operator-assigned lane` and the existing
  PR ownership guard cannot help because it requires PR metadata. A session can
  therefore scaffold another lane before any later PR ownership check runs.
- Correct fix must touch/change: the plan-scaffold CLI must resolve the same
  session-state-file convention as the ownership guard, require exactly one
  non-empty canonical top-level `Current lane:` value, require an explicit
  `--lane`, and reject any mismatch before creating or overwriting a plan. The
  state template/bootstrap wording and focused scaffold fixtures must name and
  prove that contract.
- Must not change: PR ownership semantics after a PR exists; `--force`'s
  existing overwrite behavior after lane admission succeeds; plan shape,
  product behavior, branch protection, other sessions' state, issue queues,
  watcher behavior, or any ownership lane other than this slice.

## Scope (this PR)

Ownership lane: dev-workflow/session-lane-admission
Slice phase: Workflow/process

1. Fail closed in `scripts/new_pr_plan.sh` before it writes a plan unless the
   caller supplies `--lane` exactly matching one non-empty `Current lane:` in
   the selected local session-state file. Support `--state-file` and otherwise
   follow `ATLAS_SESSION_STATE_FILE`, then legacy `SESSION_STATE.local.md`.
2. Canonicalize the session-state template and bootstrap instruction on
   `Current lane:`, and add focused fixture coverage for match, missing state,
   missing/duplicate lane, mismatch, omitted `--lane`, and `--force` bypass.

### Review Contract

- Acceptance criteria:
  - [ ] The real scaffold accepts only an explicit `--lane` equal to exactly
        one non-empty, non-placeholder canonical `Current lane:` in the
        selected state file.
  - [ ] Missing state, missing or duplicate `Current lane:`, an omitted lane,
        and a mismatch exit non-zero before a normal or `--force` scaffold
        writes/overwrites a plan.
  - [ ] The state-file precedence is explicit flag, then
        `ATLAS_SESSION_STATE_FILE`, then worktree-local legacy state.
  - [ ] Existing valid scaffold shape and successful `--force` replacement are
        preserved after admission succeeds.
- Reachability proof: invoke the real shell scaffold against temporary Git
  repositories and inspect its exit status plus whether a plan file was created
  or changed.
- Affected surfaces: the plan-scaffold CLI, session-state template/bootstrap
  contract, and focused scaffold tests. The existing PR ownership guard is an
  inspected non-target because it only applies after PR metadata exists.
- Risk areas: a state-file fallback or malformed field silently admitting a
  different lane; parsing nested text as a lane; a `--force` bypass; changing
  post-admission overwrite or existing plan shape behavior.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/SESSION_BOOTSTRAP.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `plans/PR-Session-Lane-Admission.md`
- `scripts/new_pr_plan.sh`
- `tests/test_new_pr_plan.py`

## Mechanism

`new_pr_plan.sh` will parse `--state-file` before the existing slice options.
After it resolves the repository root, it selects the explicit file, otherwise
`ATLAS_SESSION_STATE_FILE`, otherwise `$repo_root/SESSION_STATE.local.md`. It
reads only top-level `Current lane:` lines, rejects zero or multiple values and
rejects a blank value. It then requires `--lane` and compares it exactly to the
canonical value before checking an existing plan or creating its temporary
scaffold. This makes both normal and `--force` paths fail before any write.

The template and bootstrap name the canonical field so future state files and
the runtime parser agree. Tests execute the real shell CLI with temporary Git
repositories and state files, covering successful admission and every
failure-before-write boundary.

## Intentional

- This is a pre-PR lane-admission check, not an extension of
  `check_session_pr_ownership.py`: that guard correctly requires PR number,
  branch, and head SHA that do not yet exist while a plan is being scaffolded.
- The lane comparison is exact and case-sensitive. Normalizing or inferring
  lane aliases would create a new cross-session ownership policy.
- `--lane` is required rather than defaulting from session state, so the plan
  records the operator-assigned lane at the call site and an accidental omission
  cannot create a misleading plan.

## Deferred

- Extend post-PR ownership checks with canonical lane parsing only if a
  demonstrated PR-mutation gap remains after this admission gate lands.
- Watcher readiness wording, reviewer-status trust, and branch-protection
  enrollment remain separately deferred by PR #2097.

Parked hardening: none.

## Verification

- Shell syntax and focused scaffold tests — 12 passed.
- Plan-document audit and whitespace check — passed.
- The exact pytest command enrolled in `.github/workflows/pre_push_audit.yml`
  — 492 passed.
- Pending before push: guarded `scripts/push_pr.sh` local review with the final
  PR body.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `docs/SESSION_STATE_TEMPLATE.md` | 2 |
| `plans/PR-Session-Lane-Admission.md` | 128 |
| `scripts/new_pr_plan.sh` | 40 |
| `tests/test_new_pr_plan.py` | 150 |
| **Total** | **322** |
