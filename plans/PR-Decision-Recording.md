# PR-Decision-Recording

## Why this slice exists

Juan asked for the fourth priority codification rule: plan docs may use an
umbrella re-scope decision as authority only when that decision is recorded on
GitHub with a URL. I verified the example issue comment directly with
gh api repos/canfieldjuan/ATLAS/issues/comments/5085162036; the current Atlas
plan scaffold has no Decision recording slot, and no current advisory detector
checks for unlinked re-scope decisions in changed plans.
This slice is over the 400 LOC target because the single rule is required to
ship as law, template, tripwire, generated-scaffold coverage, and both-direction
detector tests together; splitting it would leave either an unenforced rule or a
checker without the required authoring prompt/evidence.

### Problem-derived contract

- Root cause: an agent can cite a real chat/local-memory decision as scope
  authority even though reviewers and future compacted sessions cannot inspect
  it from the repository or the umbrella issue.
- Correct fix must touch/change: add the rule, scaffold prompt, advisory
  tripwire, and violating/compliant tests.
- Must not change: product/runtime behavior, required CI, workflow write
  permissions, reviewer strength, review caps, open PR branches, or EOM repos.

## Scope (this PR)

Ownership lane: dev-workflow/codification
Slice phase: Workflow/process

1. Add the Decision recording rule to Atlas builder law.
2. Add the matching plan-template prompt and advisory plan detector.
3. Add both-direction detector tests.

### Review Contract

- Acceptance criteria:
  - `AGENTS.md` contains the verbatim rule; `scripts/new_pr_plan.sh` scaffolds
    URL, umbrella, and scope-effect prompts.
  - `.github/workflows/decision_recording.yml` runs read-only on `pull_request`.
  - `tests/test_check_decision_recording.py` proves unlinked re-scope decisions
    warn while recorded comment URLs stay silent.
- Reachability proof: CI/workflow-only surface; the observable gate is the new
  workflow invoking python scripts/check_decision_recording.py.
- Affected surfaces: builder docs, plan scaffold, advisory CI, detector tests.
- Risk areas: false positives, silent false negatives, wrong URL scope.
- Reviewer rules triggered: R1, R2, R10, R12.

### Boundary-change enumeration

N/A - this PR changes workflow/process enforcement only, not an Atlas product
guard, validator, resolver, or admission boundary.

- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.

### Deployed-config probing

N/A - this PR does not change a product guard/config boundary.

- Deployed/default config values: N/A.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Decision recording

- Recorded decision URL: https://github.com/canfieldjuan/ATLAS/issues/2188#issuecomment-5085162036
- Umbrella issue: https://github.com/canfieldjuan/ATLAS/issues/2188
- Scope effect: future re-scoping decisions land on the umbrella at decision
  time.

### Files touched

- `AGENTS.md`
- `scripts/new_pr_plan.sh`
- `scripts/check_decision_recording.py`
- `tests/test_check_decision_recording.py`
- `tests/test_new_pr_plan.py`
- `.github/workflows/decision_recording.yml`
- `plans/PR-Decision-Recording.md`

## Mechanism

`AGENTS.md` gains §3k.8. `new_pr_plan.sh` adds URL/umbrella/scope-effect
prompts. `check_decision_recording.py` treats the structured Decision recording
section as the closed seam: any authored recorded-decision URL, umbrella issue,
or scope-effect value declares reliance on a re-scope decision, and the plan
warns unless the recorded-decision field contains a GitHub issue/discussion
comment URL whose owner, repository, resource type, and number match the cited
umbrella URL. The workflow runs advisory-only. `tests/test_new_pr_plan.py`
proves the scaffold keeps emitting the required decision-recording fields.

## Intentional

- The detector is heuristic and advisory-first; it exits 0 unless `--strict` is
  supplied.
- N/A remains valid for plans that do not cite a re-scoping operator decision.
- The URL must be an issue/discussion comment URL in the Decision recording
  section.

## Deferred

- Promotion to required is deferred to a later operator decision.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_decision_recording.py tests/test_new_pr_plan.py -q --noconftest - 30 passed.
- python scripts/check_decision_recording.py --base claude/pr-enforced-set-verification - OK.
- python scripts/audit_plan_doc.py plans/PR-Decision-Recording.md - OK.
- python scripts/audit_plan_doc_files_touched.py plans/PR-Decision-Recording.md claude/pr-enforced-set-verification - OK.
- python scripts/audit_plan_doc_diff_size.py plans/PR-Decision-Recording.md claude/pr-enforced-set-verification - OK.
- CI-equivalent command copied from enforcing workflow: python -m pytest tests/test_check_decision_recording.py -q --noconftest plus python scripts/check_decision_recording.py --base <base> from `.github/workflows/decision_recording.yml`.
- Copied from enforcing workflow: `.github/workflows/decision_recording.yml`.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | +12 |
| `scripts/new_pr_plan.sh` | +9 |
| `scripts/check_decision_recording.py` | +144 |
| `tests/test_check_decision_recording.py` | +147 |
| `tests/test_new_pr_plan.py` | +4 |
| `.github/workflows/decision_recording.yml` | +46 |
| `plans/PR-Decision-Recording.md` | +131 |
| **Total** | **~493** |
