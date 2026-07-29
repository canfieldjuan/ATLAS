# AGENTS Mechanical Enforcement Audit - 2026-07-29

## Summary

Current `origin/main` is not a Claude two-reviewer gate. AGENTS.md defines one
builder plus the GitHub Codex connector, whose threads are enforced by
`live-reconciliation` (`AGENTS.md:3`, `AGENTS.md:7`, `AGENTS.md:8`). The live
branch-protection payload for `main` currently requires only these contexts:

- `live-reconciliation`
- `Gitleaks PR secret scan`
- `Gitleaks baseline growth guard`

Command evidence:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks
```

The local required-status checker confirms the mismatch between intended and
live protection:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks.json
python scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks.json
# required status check audit: FAIL
# - diff-budget: missing required check
```

## Status Labels

- `ENFORCED_REQUIRED`: live branch protection requires the check.
- `ENFORCED_CI_NOT_REQUIRED`: a CI job exists, but live branch protection does
  not require it.
- `LOCAL_ONLY`: enforced by local wrappers or helpers before push.
- `MANUAL_HELPER`: an enforcing helper exists, but the prescribed PR mutation
  entrypoints do not invoke it automatically.
- `ADVISORY_ONLY`: reports warnings or intentionally does not fail.
- `PROSE_ONLY`: policy exists in docs, but no enforcing artifact was found.
- `CONTRADICTED`: policy or checker expectation is stronger than live config.

## Enforcement Matrix

| Promise | Status | Evidence |
|---|---|---|
| Codex connector threads gate merge readiness. | `ENFORCED_REQUIRED` | AGENTS names Codex review threads as the reviewer gate (`AGENTS.md:7`, `AGENTS.md:8`). `ai_reconciliation_live.yml` defines required context `live-reconciliation` (`.github/workflows/ai_reconciliation_live.yml:42`) and runs `scripts/check_ai_reconciliation_live.py` (`.github/workflows/ai_reconciliation_live.yml:57`, `.github/workflows/ai_reconciliation_live.yml:61`). Live branch protection requires `live-reconciliation`. |
| `live-reconciliation` requires current-head Codex review attestation, except docs-only bypass. | `ENFORCED_REQUIRED` | The live checker fails missing current-head attestation (`scripts/check_ai_reconciliation_live.py:474`, `scripts/check_ai_reconciliation_live.py:484`) and allows docs-only PRs with no open scoped threads (`scripts/check_ai_reconciliation_live.py:469`, `scripts/check_ai_reconciliation_live.py:471`). Live branch protection requires `live-reconciliation`. |
| Gitleaks PR secret scan and baseline growth guard block merge. | `ENFORCED_REQUIRED` | `security_guardrails.yml` job name is `Gitleaks PR secret scan` (`.github/workflows/security_guardrails.yml:19`, `.github/workflows/security_guardrails.yml:20`). `gitleaks_baseline_growth_guard.yml` job name is `Gitleaks baseline growth guard` (`.github/workflows/gitleaks_baseline_growth_guard.yml:12`, `.github/workflows/gitleaks_baseline_growth_guard.yml:13`). Live branch protection requires both contexts. |
| Human non-Markdown PRs add exactly one plan; Markdown-only PRs may use `Docs-only: true`. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires this and says outcomes must surface locally and in CI (`AGENTS.md:96`, `AGENTS.md:101`). Local and CI plan admission run `audit_pr_plan_presence.py` (`scripts/pre_push_audit.sh:116`, `.github/workflows/plan_admission.yml:45`, `.github/workflows/plan_admission.yml:52`). The published job context is `plan-admission` (`.github/workflows/plan_admission.yml:17`). Live branch protection does not require `plan-admission`. |
| Full PR body shape or valid docs-only body is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS defines full and docs-only body shapes (`AGENTS.md:191`, `AGENTS.md:201`, `AGENTS.md:204`). `audit_pr_body.py` validates the shape (`scripts/audit_pr_body.py:4`, `scripts/audit_pr_body.py:9`). `pr_body_contract.yml` runs it in CI (`.github/workflows/pr_body_contract.yml:43`, `.github/workflows/pr_body_contract.yml:49`). The published job context is `pr-body-contract` (`.github/workflows/pr_body_contract.yml:17`). Live branch protection does not require `pr-body-contract`. |
| The local review bundle must pass before opening or updating a PR. | `MANUAL_HELPER` | AGENTS requires the bundle before opening or updating (`AGENTS.md:524`, `AGENTS.md:531`). `push_pr.sh` runs or delegates that bundle before `git push` (`scripts/push_pr.sh:86`, `scripts/push_pr.sh:95`), but `open_pr.sh` runs only the body audit before `gh pr create/edit` (`scripts/open_pr.sh:49`, `scripts/open_pr.sh:53`, `scripts/open_pr.sh:82`, `scripts/open_pr.sh:89`). A direct push followed by `open_pr.sh` can still create an immediate-red PR. |
| CI reruns the local review bundle after the PR exists. | `ENFORCED_CI_NOT_REQUIRED` | `pre_push_audit.yml` runs `local_pr_review.sh` on pull requests from a trusted-base checkout (`.github/workflows/pre_push_audit.yml:64`, `.github/workflows/pre_push_audit.yml:75`) and publishes job context `pre-push-audit` (`.github/workflows/pre_push_audit.yml:21`). Live branch protection does not require `pre-push-audit`. |
| Review Contract shape and triggered-rule agreement are checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires the plan's Review Contract to name acceptance criteria, affected surfaces, risk areas, and triggered rule IDs (`AGENTS.md:118`, `AGENTS.md:122`). `review_contract.yml` defines the published job context `review-contract` (`.github/workflows/review_contract.yml:8`) and runs `audit_plan_doc.py` plus `audit_review_rules_triggered.py` against the PR plan (`.github/workflows/review_contract.yml:53`, `.github/workflows/review_contract.yml:55`). Local review runs the same triggered-rule audit for changed plan docs (`scripts/local_pr_review.sh:220`, `scripts/local_pr_review.sh:222`). Live branch protection does not require `review-contract`. |
| Problem-derived contract is written before code. | `PROSE_ONLY` | AGENTS requires a Problem-derived contract in the plan (`AGENTS.md:104`, `AGENTS.md:110`) and says the plan is written before any code change (`AGENTS.md:381`, `AGENTS.md:385`), but the discovered tooling checks shape/content, not historical authoring order. No timestamp or commit-order gate was found. |
| Commit messages mirror the plan and PR body contract. | `PROSE_ONLY` | AGENTS requires commit messages to include `Plan:`, `Slice phase:`, Intentional, Deferred, Parked hardening, and Cold diff reconstruction (`AGENTS.md:234`, `AGENTS.md:238`). No `scripts/audit_*commit*`/`scripts/check_*commit*` checker or CI workflow was found to validate that canonical commit-message shape before squash merge. |
| `diff-budget` should be a required status check. | `CONTRADICTED` | The workflow exists (`.github/workflows/diff_budget.yml:23`) and the required-status checker expects `diff-budget` (`scripts/check_required_status_checks.py:13`, `scripts/check_required_status_checks.py:15`). Live branch protection omits it, and the checker fails against the live payload. |
| Builder branch names use `claude/pr-<slice-name>`. | `PROSE_ONLY` | AGENTS defines the branch naming convention (`AGENTS.md:246`, `AGENTS.md:249`). `new_pr_plan.sh` validates the lane argument against session state, not the current branch name (`scripts/new_pr_plan.sh:126`, `scripts/new_pr_plan.sh:127`), and no branch-name gate was found in `open_pr.sh`, `push_pr.sh`, or CI. |
| PRs open ready for review by default, with draft only on explicit operator request. | `PROSE_ONLY` | AGENTS forbids draft PRs unless explicitly requested (`AGENTS.md:251`, `AGENTS.md:256`). `open_pr.sh` rejects body arguments only (`scripts/open_pr.sh:55`, `scripts/open_pr.sh:61`) and forwards other `gh pr create` args, including `--draft`, without an operator-consent check (`scripts/open_pr.sh:89`). |
| Whole unit suite execution should happen on every PR. | `CONTRADICTED` | `unit_gate.yml` still says the whole unit suite runs on every PR (`.github/workflows/unit_gate.yml:3`, `.github/workflows/unit_gate.yml:9`), but the implemented job now selects impacted tests (`.github/workflows/unit_gate.yml:77`, `.github/workflows/unit_gate.yml:94`) and runs `--growth-only` when no tests are reachable from changed files (`.github/workflows/unit_gate.yml:107`, `.github/workflows/unit_gate.yml:115`). |
| Unit Gate impacted-test or growth-only execution runs on every PR. | `ENFORCED_CI_NOT_REQUIRED` | `unit_gate.yml` triggers on every pull request (`.github/workflows/unit_gate.yml:17`, `.github/workflows/unit_gate.yml:19`) and executes either the selected impacted test files or the baseline growth guard (`.github/workflows/unit_gate.yml:99`, `.github/workflows/unit_gate.yml:125`). The published job context is `unit-gate` (`.github/workflows/unit_gate.yml:26`). Live branch protection does not require `unit-gate`. |
| Session lane drift is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires session ownership/lane checks before PR mutation (`AGENTS.md:425`, `AGENTS.md:454`). `new_pr_plan.sh` refuses a lane mismatch (`scripts/new_pr_plan.sh:126`) and `audit_pr_session_drift.py` fails ownership or current-body phase errors (`scripts/audit_pr_session_drift.py:112`, `scripts/audit_pr_session_drift.py:119`). `session_lane.yml` runs the drift audit in CI (`.github/workflows/session_lane.yml:60`, `.github/workflows/session_lane.yml:67`). The published job context is `session-lane` (`.github/workflows/session_lane.yml:17`). Live branch protection does not require `session-lane`. |
| PR ownership before mutation is guarded. | `MANUAL_HELPER` | AGENTS says run `check_session_pr_ownership.py` before PR mutation when metadata is known (`AGENTS.md:436`, `AGENTS.md:440`). The script fails when the PR is not owned or is listed must-not-touch (`scripts/check_session_pr_ownership.py:92`, `scripts/check_session_pr_ownership.py:95`), but no invocation was found in `open_pr.sh`, `push_pr.sh`, or `local_pr_review.sh`; `open_pr.sh` proceeds directly to `gh pr edit/create` (`scripts/open_pr.sh:70`, `scripts/open_pr.sh:89`). |
| Push wrapper prevents skipped local review. | `LOCAL_ONLY` | AGENTS says `push_pr.sh` wires the body file and rejects `--no-verify` (`AGENTS.md:485`, `AGENTS.md:490`). The wrapper rejects `--no-verify` (`scripts/push_pr.sh:55`, `scripts/push_pr.sh:57`) and runs local review when the managed hook will not (`scripts/push_pr.sh:86`, `scripts/push_pr.sh:89`). GitHub cannot know if a user pushed another way. |
| Worktree/branch teardown after merge is mandatory. | `PROSE_ONLY` | AGENTS requires removing the dedicated worktree, deleting the branch, then archiving the plan from synced `main` after merge (`AGENTS.md:260`, `AGENTS.md:283`). No post-merge gate was found that fails lingering worktrees or local branches; GitHub branch protection cannot observe local teardown. |
| Plans archive backlog nudge matches teardown cleanup. | `ADVISORY_ONLY` | AGENTS requires archiving only the merged plan by name on a local `main` synced to `origin/main` (`AGENTS.md:270`, `AGENTS.md:283`). `archive_plans.py archive` is bulk and still moves every root `PR-*.md` plan (`scripts/archive_plans.py:109`, `scripts/archive_plans.py:130`), but this PR repairs `archive_plans.py check` and local review advisory output so the nudge names the synced-main, single-plan `git mv` plus `archive_plans.py index` flow (`scripts/archive_plans.py:183`, `scripts/archive_plans.py:185`; `scripts/local_pr_review.sh:240`, `scripts/local_pr_review.sh:242`). |
| Cross-layer caller hints are printed for logic/shared-function review. | `ADVISORY_ONLY` | `local_pr_review.sh` runs `audit_cross_layer_callers.py` as a non-blocking hint label (`scripts/local_pr_review.sh:185`, `scripts/local_pr_review.sh:186`), matching AGENTS' statement that outside references can be valid (`AGENTS.md:555`, `AGENTS.md:559`). |
| Caller-hint disposition is mandatory for logic/shared-function PRs. | `PROSE_ONLY` | AGENTS requires the builder to either add caller-layer tests or name why referenced callers are unaffected (`AGENTS.md:555`, `AGENTS.md:559`). The local review output cannot detect whether the builder read or dispositioned those hints, and no PR-body or plan gate was found that requires this record. |
| Guard class closure, seam convergence, boundary enumeration, and deployed config probing are structural tripwires. | `ADVISORY_ONLY` | These workflows self-identify as advisory or never-blocking: guard class closure (`.github/workflows/guard_class_closure.yml:3`, `.github/workflows/guard_class_closure.yml:6`), seam convergence (`.github/workflows/seam_convergence.yml:3`, `.github/workflows/seam_convergence.yml:8`), boundary enumeration (`.github/workflows/boundary_change_enumeration.yml:3`, `.github/workflows/boundary_change_enumeration.yml:4`), and deployed config probing (`.github/workflows/deployed_config_probing.yml:3`, `.github/workflows/deployed_config_probing.yml:4`). Live branch protection requires none of them. |
| Auditor/checker changes ship fixture tests. | `PROSE_ONLY` | AGENTS requires fixture tests for new `scripts/audit_*.py` files and negative coverage for checkers (`AGENTS.md:897`, `AGENTS.md:914`, `AGENTS.md:932`). Existing local review runs many tooling tests from `pre_push_audit.yml` (`.github/workflows/pre_push_audit.yml:80`, `.github/workflows/pre_push_audit.yml:83`), but no generic changed-file gate was found that fails every new or changed checker lacking fixture coverage. |

## Follow-Up Slices

1. `PR-Require-Diff-Budget-Branch-Protection`: make live branch protection match
   `scripts/check_required_status_checks.py` by requiring `diff-budget`, or
   intentionally remove `diff-budget` from the required-context expectation.
2. `PR-Required-Workflow-Enrollment-Audit`: decide which existing CI-only
   process checks must become branch-protection required, starting with
   `pre-push-audit`, `pr-body-contract`, `plan-admission`, `review-contract`,
   `session-lane`, and the impacted-test/growth-only `unit-gate`.
3. `PR-Pre-Open-Local-Review-Guard`: make `open_pr.sh` require proof that
   `local_pr_review.sh` already passed for the pushed head, or downgrade the
   AGENTS pre-open timing claim to manual-only.
4. `PR-PR-Mutation-Ownership-Wrapper`: wire `check_session_pr_ownership.py` into
   the PR mutation helpers, or downgrade the AGENTS wording to manual-only.
5. `PR-Commit-Message-Contract-Gate`: validate canonical commit-message
   sections before PR publication or explicitly limit the requirement to the
   squash-merge message.
6. `PR-Open-Ready-Draft-Consent-Guard`: reject `--draft` in `open_pr.sh` unless
   an explicit operator-consent flag or recorded session-state field is present.
7. `PR-Branch-Naming-Gate`: enforce `claude/pr-<slice-name>` for PR branches in
   the open/push/session-lane path, or document it as a convention only.
8. `PR-Caller-Hint-Disposition-Gate`: require logic/shared-function PRs to
   record caller-hint disposition in the plan or PR body, then audit that record.
9. `PR-Audit-Checker-Fixture-Enforcement`: add a changed-file gate for new or
   materially changed `scripts/audit_*.py`, `scripts/check_*.py`, and
   `scripts/evaluate_*.py` files so fixture coverage is mechanical instead of
   prose-only.
10. `PR-Unit-Gate-Prose-Implementation-Alignment`: either restore true whole-suite
   unit execution or update the workflow prose to match impacted-test selection.
11. `PR-Teardown-Archive-Enforcement-Decision`: decide whether plan archive
   backlog should remain advisory after this PR's safer nudge, or become a
   required cleanup gate with a concurrent-session-safe exception.

## Bottom Line

The current hard merge gate is much narrower than the local/CI review bundle:
branch protection blocks Codex thread reconciliation and Gitleaks only. The
AGENTS rules that keep PR shape sane split into three buckets: a narrow hard
merge gate, several CI-only checks that are visible but not required, and several
local/manual promises that no current gate can prove were respected. The biggest
concrete branch-protection contradiction is `diff-budget`: repository code
expects it to be required, while live protection does not require it. The biggest
pre-open contradiction is local review timing: CI can rerun the bundle after a PR
exists, but it cannot prove the bundle passed before GitHub first saw the PR.
