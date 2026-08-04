# AGENTS Mechanical Enforcement Audit - 2026-07-29

## 2026-08-04 Update

The required-status contradiction found in this audit has been closed in live
branch protection. `main` now requires the `ci/gates.yml` `branch_required`
contexts from the GitHub Actions app source: `live-reconciliation`,
`diff-budget`, `plan-admission`, `session-lane`, `review-contract`,
`pr-body-contract`, `Gitleaks PR secret scan`, and
`Gitleaks baseline growth guard`.

Verification:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks \
  > /tmp/atlas-required-status-checks-live-after.json
python scripts/check_required_status_checks.py \
  --payload-file /tmp/atlas-required-status-checks-live-after.json
# required status check audit: PASS
```

## Summary

Current `origin/main` is not a Claude two-reviewer gate. AGENTS.md defines one
builder plus the GitHub Codex connector, whose threads are enforced by
`live-reconciliation` (`AGENTS.md:3`, `AGENTS.md:7`, `AGENTS.md:8`). The live
branch-protection payload for `main` required only these contexts at the time
of the original 2026-07-29 audit:

- `live-reconciliation`
- `Gitleaks PR secret scan`
- `Gitleaks baseline growth guard`

Command evidence:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks
```

The local required-status checker confirmed the mismatch between intended and
live protection at that time:

```bash
gh api repos/canfieldjuan/ATLAS/branches/main/protection/required_status_checks > /tmp/atlas-required-status-checks.json
python scripts/check_required_status_checks.py --payload-file /tmp/atlas-required-status-checks.json
# required status check audit: FAIL
# - diff-budget: missing required check
# - plan-admission: missing required check
# - session-lane: missing required check
# - review-contract: missing required check
# - pr-body-contract: missing required check
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
| `live-reconciliation` requires current-head Codex review attestation, except docs-only bypass. | `ENFORCED_REQUIRED` | The live checker fails a current-head changes-requested review or a fresh head with no scoped Codex activity inside the review window (`scripts/check_ai_reconciliation_live.py:551`, `scripts/check_ai_reconciliation_live.py:570`) and allows docs-only PRs with no open scoped threads (`scripts/check_ai_reconciliation_live.py:575`, `scripts/check_ai_reconciliation_live.py:579`). Live branch protection requires `live-reconciliation`. |
| Gitleaks PR secret scan and baseline growth guard block merge. | `ENFORCED_REQUIRED` | `security_guardrails.yml` job name is `Gitleaks PR secret scan` (`.github/workflows/security_guardrails.yml:19`, `.github/workflows/security_guardrails.yml:20`). `gitleaks_baseline_growth_guard.yml` job name is `Gitleaks baseline growth guard` (`.github/workflows/gitleaks_baseline_growth_guard.yml:12`, `.github/workflows/gitleaks_baseline_growth_guard.yml:13`). Live branch protection requires both contexts. |
| Human non-Markdown PRs add exactly one plan; Markdown-only PRs may use `Docs-only: true`. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires this and says outcomes must surface locally and in CI (`AGENTS.md:96`, `AGENTS.md:101`). Local and CI plan admission run `audit_pr_plan_presence.py` (`scripts/pre_push_audit.sh:116`, `.github/workflows/plan_admission.yml:45`, `.github/workflows/plan_admission.yml:52`). The published job context is `plan-admission` (`.github/workflows/plan_admission.yml:17`). Live branch protection does not require `plan-admission`. |
| Full PR body shape or valid docs-only body is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS defines full and docs-only body shapes (`AGENTS.md:191`, `AGENTS.md:201`, `AGENTS.md:204`). `audit_pr_body.py` validates the shape (`scripts/audit_pr_body.py:4`, `scripts/audit_pr_body.py:9`). `pr_body_contract.yml` runs it in CI (`.github/workflows/pr_body_contract.yml:43`, `.github/workflows/pr_body_contract.yml:49`). The published job context is `pr-body-contract` (`.github/workflows/pr_body_contract.yml:17`). Live branch protection does not require `pr-body-contract`. |
| The local review bundle must pass before opening or updating a PR through the prescribed wrapper. | `LOCAL_ONLY` | AGENTS requires the bundle before opening or updating (`AGENTS.md:524`, `AGENTS.md:531`). `push_pr.sh` runs or delegates that bundle before `git push` (`scripts/push_pr.sh:86`, `scripts/push_pr.sh:95`). Current `open_pr.sh` also runs final local review before both `gh pr edit` and `gh pr create` (`scripts/open_pr.sh:321`, `scripts/open_pr.sh:330`, `scripts/open_pr.sh:345`, `scripts/open_pr.sh:352`). GitHub cannot prove a user avoided direct `gh` commands outside the wrapper. |
| CI reruns the local review bundle after the PR exists. | `ENFORCED_CI_NOT_REQUIRED` | `pre_push_audit.yml` runs `local_pr_review.sh` on pull requests from a trusted-base checkout (`.github/workflows/pre_push_audit.yml:64`, `.github/workflows/pre_push_audit.yml:75`) and publishes job context `pre-push-audit` (`.github/workflows/pre_push_audit.yml:21`). Live branch protection does not require `pre-push-audit`. |
| Review Contract shape and triggered-rule agreement are checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires the plan's Review Contract to name acceptance criteria, affected surfaces, risk areas, and triggered rule IDs (`AGENTS.md:118`, `AGENTS.md:122`). `review_contract.yml` defines the published job context `review-contract` (`.github/workflows/review_contract.yml:8`) and runs `audit_plan_doc.py` plus `audit_review_rules_triggered.py` against the PR plan (`.github/workflows/review_contract.yml:53`, `.github/workflows/review_contract.yml:55`). Local review runs the same triggered-rule audit for changed plan docs (`scripts/local_pr_review.sh:220`, `scripts/local_pr_review.sh:222`). Live branch protection does not require `review-contract`. |
| Problem-derived contract is written before code. | `PROSE_ONLY` | AGENTS requires a Problem-derived contract in the plan (`AGENTS.md:104`, `AGENTS.md:110`) and says the plan is written before any code change (`AGENTS.md:381`, `AGENTS.md:385`), but the discovered tooling checks shape/content, not historical authoring order. No timestamp or commit-order gate was found. |
| Commit messages mirror the plan and PR body contract. | `PROSE_ONLY` | AGENTS requires commit messages to include `Plan:`, `Slice phase:`, Intentional, Deferred, Parked hardening, and Cold diff reconstruction (`AGENTS.md:234`, `AGENTS.md:238`). No `scripts/audit_*commit*`/`scripts/check_*commit*` checker or CI workflow was found to validate that canonical commit-message shape before squash merge. |
| `diff-budget`, `plan-admission`, `session-lane`, `review-contract`, and `pr-body-contract` should be required status checks according to the repo-side registry/checker. | `CONTRADICTED` | The gate registry marks these contexts `branch_required` (`ci/gates.yml:21`, `ci/gates.yml:56`), and the required-status checker derives its default contexts from that registry (`scripts/check_required_status_checks.py:169`, `scripts/check_required_status_checks.py:184`). Live branch protection omits them, and the checker fails against the live payload. |
| Builder branch names use `claude/pr-<slice-name>`. | `PROSE_ONLY` | AGENTS defines the branch naming convention (`AGENTS.md:246`, `AGENTS.md:249`). `new_pr_plan.sh` validates the lane argument against session state, not the current branch name (`scripts/new_pr_plan.sh:126`, `scripts/new_pr_plan.sh:127`), and no branch-name gate was found in `open_pr.sh`, `push_pr.sh`, or CI. |
| PRs open ready for review by default, with draft only on explicit operator request. | `PROSE_ONLY` | AGENTS forbids draft PRs unless explicitly requested (`AGENTS.md:251`, `AGENTS.md:256`). `open_pr.sh` rejects body arguments only (`scripts/open_pr.sh:55`, `scripts/open_pr.sh:61`) and forwards other `gh pr create` args, including `--draft`, without an operator-consent check (`scripts/open_pr.sh:89`). |
| Unit Gate impacted-test, full-suite fallback, or growth-only execution runs on every PR. | `ENFORCED_CI_NOT_REQUIRED` | `unit_gate.yml` triggers on every pull request (`.github/workflows/unit_gate.yml:17`, `.github/workflows/unit_gate.yml:19`), selects impacted tests or FULL (`.github/workflows/unit_gate.yml:77`, `.github/workflows/unit_gate.yml:94`), and executes either the selected tests, the full-suite fallback, or the baseline growth guard (`.github/workflows/unit_gate.yml:99`, `.github/workflows/unit_gate.yml:125`). The published job context is `unit-gate` (`.github/workflows/unit_gate.yml:26`). Live branch protection does not require `unit-gate`. |
| Session lane drift is checked. | `ENFORCED_CI_NOT_REQUIRED` | AGENTS requires session ownership/lane checks before PR mutation (`AGENTS.md:425`, `AGENTS.md:454`). `new_pr_plan.sh` refuses a lane mismatch (`scripts/new_pr_plan.sh:126`) and `audit_pr_session_drift.py` fails ownership or current-body phase errors (`scripts/audit_pr_session_drift.py:112`, `scripts/audit_pr_session_drift.py:119`). `session_lane.yml` runs the drift audit in CI (`.github/workflows/session_lane.yml:60`, `.github/workflows/session_lane.yml:67`). The published job context is `session-lane` (`.github/workflows/session_lane.yml:17`). Live branch protection does not require `session-lane`. |
| PR ownership before mutation is guarded. | `MANUAL_HELPER` | AGENTS says run `check_session_pr_ownership.py` before PR mutation when metadata is known (`AGENTS.md:436`, `AGENTS.md:440`). The script fails when the PR is not owned or is listed must-not-touch (`scripts/check_session_pr_ownership.py:92`, `scripts/check_session_pr_ownership.py:95`), but no invocation was found in `open_pr.sh`, `push_pr.sh`, or `local_pr_review.sh`; after final local review, `open_pr.sh` mutates GitHub with `gh pr edit` or `gh pr create` (`scripts/open_pr.sh:321`, `scripts/open_pr.sh:330`, `scripts/open_pr.sh:345`, `scripts/open_pr.sh:352`). |
| Push wrapper prevents skipped local review. | `LOCAL_ONLY` | AGENTS says `push_pr.sh` wires the body file and rejects `--no-verify` (`AGENTS.md:485`, `AGENTS.md:490`). The wrapper rejects `--no-verify` (`scripts/push_pr.sh:55`, `scripts/push_pr.sh:57`) and runs local review when the managed hook will not (`scripts/push_pr.sh:86`, `scripts/push_pr.sh:89`). GitHub cannot know if a user pushed another way. |
| Worktree/branch teardown after merge is mandatory. | `PROSE_ONLY` | AGENTS requires removing the dedicated worktree, deleting the branch, then archiving the plan from synced `main` after merge (`AGENTS.md:260`, `AGENTS.md:283`). No post-merge gate was found that fails lingering worktrees or local branches; GitHub branch protection cannot observe local teardown. |
| Plans archive backlog nudge matches teardown cleanup. | `ADVISORY_ONLY` | AGENTS requires archiving only the merged plan by name on a local `main` synced to `origin/main` (`AGENTS.md:270`, `AGENTS.md:283`). `archive_plans.py archive` is bulk and still moves every root `PR-*.md` plan (`scripts/archive_plans.py:109`, `scripts/archive_plans.py:130`), but this PR repairs `archive_plans.py check` and local review advisory output so the nudge names the synced-main, single-plan `git mv` plus `archive_plans.py index` flow (`scripts/archive_plans.py:183`, `scripts/archive_plans.py:185`; `scripts/local_pr_review.sh:240`, `scripts/local_pr_review.sh:242`). |
| Cross-layer caller hints are printed for logic/shared-function review. | `ADVISORY_ONLY` | `local_pr_review.sh` runs `audit_cross_layer_callers.py` as a non-blocking hint label (`scripts/local_pr_review.sh:185`, `scripts/local_pr_review.sh:186`), matching AGENTS' statement that outside references can be valid (`AGENTS.md:555`, `AGENTS.md:559`). |
| Caller-hint disposition is mandatory for logic/shared-function PRs. | `PROSE_ONLY` | AGENTS requires the builder to either add caller-layer tests or name why referenced callers are unaffected (`AGENTS.md:555`, `AGENTS.md:559`). The local review output cannot detect whether the builder read or dispositioned those hints, and no PR-body or plan gate was found that requires this record. |
| Guard class closure, seam convergence, boundary enumeration, and deployed config probing are structural tripwires. | `ADVISORY_ONLY` | These workflows self-identify as advisory or never-blocking: guard class closure (`.github/workflows/guard_class_closure.yml:3`, `.github/workflows/guard_class_closure.yml:6`), seam convergence (`.github/workflows/seam_convergence.yml:3`, `.github/workflows/seam_convergence.yml:8`), boundary enumeration (`.github/workflows/boundary_change_enumeration.yml:3`, `.github/workflows/boundary_change_enumeration.yml:4`), and deployed config probing (`.github/workflows/deployed_config_probing.yml:3`, `.github/workflows/deployed_config_probing.yml:4`). Live branch protection requires none of them. |
| Auditor/checker changes ship fixture tests. | `PROSE_ONLY` | AGENTS requires fixture tests for new `scripts/audit_*.py` files and negative coverage for checkers (`AGENTS.md:897`, `AGENTS.md:914`, `AGENTS.md:932`). Existing local review runs many tooling tests from `pre_push_audit.yml` (`.github/workflows/pre_push_audit.yml:80`, `.github/workflows/pre_push_audit.yml:83`), but no generic changed-file gate was found that fails every new or changed checker lacking fixture coverage. |

## Follow-Up Slices

1. `PR-Required-Status-Check-Alignment`: make live branch protection match
   `scripts/check_required_status_checks.py` by requiring `diff-budget`,
   `plan-admission`, `session-lane`, `review-contract`, and
   `pr-body-contract`, or intentionally remove those contexts from the
   required-context expectation.
2. `PR-Required-Workflow-Enrollment-Audit`: decide which existing CI-only
   process checks must become branch-protection required, starting with
   `pre-push-audit` and the impacted-test/growth-only `unit-gate`.
3. `PR-PR-Mutation-Ownership-Wrapper`: wire `check_session_pr_ownership.py` into
   the PR mutation helpers, or downgrade the AGENTS wording to manual-only.
4. `PR-Commit-Message-Contract-Gate`: validate canonical commit-message
   sections before PR publication or explicitly limit the requirement to the
   squash-merge message.
5. `PR-Open-Ready-Draft-Consent-Guard`: reject `--draft` in `open_pr.sh` unless
   an explicit operator-consent flag or recorded session-state field is present.
6. `PR-Branch-Naming-Gate`: enforce `claude/pr-<slice-name>` for PR branches in
   the open/push/session-lane path, or document it as a convention only.
7. `PR-Caller-Hint-Disposition-Gate`: require logic/shared-function PRs to
   record caller-hint disposition in the plan or PR body, then audit that record.
8. `PR-Audit-Checker-Fixture-Enforcement`: add a changed-file gate for new or
   materially changed `scripts/audit_*.py`, `scripts/check_*.py`, and
   `scripts/evaluate_*.py` files so fixture coverage is mechanical instead of
   prose-only.
9. `PR-Teardown-Archive-Enforcement-Decision`: decide whether plan archive
   backlog should remain advisory after this PR's safer nudge, or become a
   required cleanup gate with a concurrent-session-safe exception.

## Bottom Line

As of 2026-08-04, the biggest branch-protection contradiction identified here
has been closed: live protection requires the registry's `branch_required`
contexts. The remaining mechanical gaps are local/manual promises that GitHub
cannot prove directly, plus follow-up decisions about whether `pre-push-audit`
and `unit-gate` should stay visible-but-not-required or become branch-required.
