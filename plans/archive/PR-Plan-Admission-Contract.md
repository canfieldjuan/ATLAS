# PR-Plan-Admission-Contract

## Why this slice exists

The AGENTS contract says non-trivial PRs carry a plan and a Review Contract, but
the runtime workflow does not consistently enforce either claim. A human
non-Markdown diff with no changed plan makes plan and reviewer-rule audits skip;
the plan scaffold omits the required Review Contract; and the PR-body audit
rejects every human planless PR, making the intended Markdown-only exemption
impossible. The current pre-push-audit fixture suite also has three stale
failures because its plan fixture lacks the now-required problem-derived
contract and its copied scripts omit the watcher-safety auditor.

The first CI repair exposed the remaining root gaps: a full plan body is only
checked for existence rather than against the sole plan the branch added;
`push_pr.sh` audits a Docs-only body before it refreshes the base that audit
requires; the shared policy imports Python-3.11-only `StrEnum`; and unit-gate
correctly rejects its three now-passing pre-push-audit tests as stale baseline
entries. These are direct review/CI findings in this PR's admission path, not
new workflow scope.

The subsequent current-head review found the same full-body binding omitted for
Markdown-only diffs: a planless documentation PR could cite an unrelated
existing plan instead of using the explicit `Docs-only: true` marker. This is
the same admission root cause and requires only the body guard and its boundary
fixtures.

The latest review exposed the exemption-side mirror of that guard: path suffix
alone classified symlinks and compound shell-script Markdown names as docs only,
while branch-added plans already required regular Git blobs. It also found
that `open_pr.sh` used a base-dependent body audit before refreshing `origin/main`.
Both are fail-closed admission-path defects within this slice; branch-protection
enrollment remains an external operator action.

The final independent review found that the new strict exemption checked every
changed path only in the PR head. A deleted regular Markdown file therefore
became plan-required, blocking the contract-required merged-plan archive flow.
Docs-only admission must validate deleted Markdown blobs against the merge base
and added or modified non-executable Markdown blobs against the PR head.

This workflow/process slice repairs those admission paths. It is justified by a
real safety risk: an unplanned code/workflow change can receive a green
mechanical bundle without the plan, rule, and body contracts the repository
claims to enforce.

The slice intentionally exceeds the 400-LOC soft target because the shared
classifier, every public admission entrypoint, and both sides of each
exemption/failure path must land together. Splitting the wrapper/body/CI wiring
from the guard would recreate a bypass in at least one real PR path.

### Problem-derived contract

- Root cause: plan admission is split across independent scripts with no shared
  changed-path classification or single authoritative branch-added plan. The
  full-body path consequently validates any existing plan instead of the plan
  that admits this diff, and its push wrapper uses the dependent base ref before
  refreshing it. The shared classifier also assumes Python 3.11 despite the
  repository's Python 3.10 support signals.
- Correct fix must touch/change: one shared diff/author classifier; plan
  presence, PR-body, push/open, and local-review entrypoints; the plan scaffold
  and structural plan audit; reviewer-rule extraction; aligned workflow docs;
  focused failure-branch tests and their explicit pre-push CI enrollment; the
  full-body/branch-plan binding; regular-blob, single-suffix docs-only
  classification; wrapper refresh ordering; Python-3.10-safe enum behavior; and
  the unit-gate baseline shrink earned by passing tests.
- Must not change: product routes, schemas, customer-visible behavior,
  migrations, GitHub branch-protection settings, Dependabot's existing body
  exemption, reviewer-status/R14 live-comment enforcement, or any maturity
  baseline.

## Scope (this PR)

Ownership lane: dev-workflow/plan-admission
Slice phase: Workflow/process

Max files: 30

1. Require exactly one branch-added plan document named under `plans/` with a
   `PR-` prefix and an MD filename suffix for human diffs with any
   non-Markdown path, while exempting only non-executable regular Git blobs
   with the Markdown extension as their sole suffix (at the merge base for
   deletions and at the PR head otherwise) and Dependabot.
2. Make a planless Markdown-only human PR use `Docs-only: true` as its first
   non-empty body line; a normal full body remains valid only when its branch
   adds the plan it names.
3. Scaffold and structurally enforce one non-empty `### Review Contract` inside
   `## Scope (this PR)`, then keep reviewer-rule declarations inside that block.
4. Repair the affected fixture harness and enroll all added/changed workflow
   tooling tests in both explicit pre-push-audit pytest commands.
5. Bind full human plan bodies to the one branch-added plan, refresh the base
   in both wrappers before any base-dependent body audit, preserve Python 3.10 compatibility,
   remove only the three newly passing unit-gate baseline nodes, and make the
   existing open-wrapper fixture establish the base the real push-then-open
   flow supplies.

### Review Contract

- Acceptance criteria:
  - [ ] A human non-Markdown diff without exactly one branch-added plan fails
        both local review and trusted-base CI with an actionable admission error.
  - [ ] A planless docs-only human diff passes only when each added or modified
        path is a non-executable regular PR-head blob and each deleted path is
        a non-executable regular merge-base blob, with the Markdown extension
        as its sole suffix, and its body begins `Docs-only: true`; symlinks,
        executable Markdown, and compound shell-script Markdown names require
        a plan.
  - [ ] Dependabot keeps its explicit exemption, and every exemption is printed
        rather than silently skipped.
  - [ ] A full human plan body must name the single plan added by its branch;
        an existing but unrelated plan is rejected by the local and trusted-base
        body gates.
  - [ ] `push_pr.sh` and `open_pr.sh` fetch `origin/main` before evaluating a
        Docs-only body, and policy imports work on Python 3.10-compatible runtimes.
  - [ ] Unit-gate's baseline shrinks by exactly its three newly passing
        pre-push-audit nodes, without changing a maturity baseline.
  - [ ] New plan scaffolds contain a Review Contract; plan audit rejects a
        missing, duplicate, empty, or out-of-Scope contract.
  - [ ] Reviewer-rule declarations are read only from the Review Contract and
        still recognize established label variants.
- Reachability proof: run the real `scripts/local_pr_review.sh` bundle against
  temporary Git repositories for human code, docs-only, and Dependabot cases;
  observe its pre-push and PR-body results. Exercise `push_pr.sh` and
  `open_pr.sh` in dry-run fixture tests to prove they pass the same policy.
- Affected surfaces: plan/body admission scripts, builder wrappers, local and
  trusted-base review wiring, plan/reviewer workflow docs, and their tests.
- Risk areas: accidentally treating a non-Markdown deletion/rename as docs
  only; letting a stale or unresolved base ref create an exemption; diverging
  bot handling across entrypoints; breaking current freeform Review Contract
  prose while adding structural enforcement; accepting an unrelated existing
  plan; or silently dropping Python 3.10 compatibility.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/pr_body_contract.yml`
- `.github/workflows/pre_push_audit.yml`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/REVIEWER_RULES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/ai_dev_operating_model.md`
- `plans/PR-Plan-Admission-Contract.md`
- `scripts/_pr_change_policy.py`
- `scripts/audit_plan_doc.py`
- `scripts/audit_pr_body.py`
- `scripts/audit_pr_plan_presence.py`
- `scripts/audit_review_rules_triggered.py`
- `scripts/local_pr_review.sh`
- `scripts/new_pr_plan.sh`
- `scripts/open_pr.sh`
- `scripts/pre_push_audit.sh`
- `scripts/push_pr.sh`
- `tests/test_audit_plan_doc.py`
- `tests/test_audit_pr_body.py`
- `tests/test_audit_pr_plan_presence.py`
- `tests/test_audit_review_rules_triggered.py`
- `tests/test_local_pr_review.py`
- `tests/test_new_pr_plan.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_pr_body_contract_workflow.py`
- `tests/test_pre_push_audit.py`
- `tests/test_pre_push_audit_workflow.py`
- `tests/test_push_pr_wrapper.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

`scripts/_pr_change_policy.py` will resolve the merge base, list every changed
path (including both sides of a rename), and classify it as Dependabot-exempt,
docs-only, no-change, or plan-required. Docs-only requires each changed path to
be a non-executable regular Git blob with the Markdown extension as its only
suffix: deleted paths are checked at the merge base and every other path at the
PR head.
`scripts/audit_pr_plan_presence.py`
consumes that result in `pre_push_audit.sh`; `scripts/audit_pr_body.py` consumes
the same result when its caller provides a base ref and, in a trusted-base gate,
the fetched PR-head ref. The body checker accepts an explicit docs-only marker
only for a non-empty Markdown-only human diff and otherwise requires a full
body to name the sole branch-added plan. This keeps a plan-backed documentation
PR valid while preventing a planless one from citing an unrelated existing plan.

`push_pr.sh` and `open_pr.sh` refresh `origin/main` before the base-dependent
body audit so Docs-only validation never observes a missing or stale local base. The shared
policy keeps string-valued enum behavior with Python-3.10-compatible standard
library types. The unit baseline removes only the three nodes CI reports as
passing, so the existing ratchet shrinks rather than being expanded. The
open-wrapper fixture creates its `origin/main` base before exercising the
wrapper, matching the documented push-then-open route instead of weakening the
new body audit.

The plan scaffold will add the documented Review Contract template. The plan
auditor will inspect its exact Scope section for one non-empty contract, while
the rule trigger auditor reads accepted rule-label aliases only inside that
contract. Wrapper and local-review callers will forward the base ref and PR
author so local, wrapper, and trusted-base behavior agree.

## Intentional

- Only non-executable regular Git blobs with the Markdown extension as their
  sole suffix are human plan-exempt: deletions are proven at the merge base and
  all other paths at the PR head. Symlinks, executable or compound names,
  `.txt`, configuration, workflow, lockfile, test, asset, and source changes
  require a plan.
- The Review Contract audit is structural, not a natural-language completeness
  parser, so existing freeform acceptance/reachability prose remains valid.
- `Docs-only: true` is deliberately explicit; freeform planless human bodies
  remain invalid.
- Existing Dependabot handling remains centralized through the same author
  helper rather than broadening exemptions to other bots.
- The CI follow-up avoids ratchet-baseline expansion: equivalent iterator
  access removes newly flagged fixed-index patterns, and the admission test
  proves the shared policy's fail-closed exception directly.
- The unit-gate baseline change is an earned three-node shrink reported by CI;
  it does not waive or hide a failing test.

## Deferred

- Session-lane admission is the next merge-ordered slice; it will add canonical
  `Current lane:` enforcement to the plan scaffold.
- Watcher readiness wording, `claude-review` trust clarification, branch
  protection reconciliation, and live R14 review-body enforcement belong to
  later status-truthfulness work.
- Admission code is advisory until an operator enrolls `pr-body-contract` or
  `pre-push-audit` in branch protection; this PR must not mutate that global
  configuration.

Parked hardening: none; the deferred work is a planned workflow sequence, not
newly discovered product hardening.

## Verification

- Before implementation, the initial focused fixture command reported 29 passed
  and 3 stale fixture failures; this slice repairs those fixtures.
- The focused policy/body/wrapper/scaffold/plan-audit test group — 59 passed
  after the post-review repair.
- Current-head review repair: `python -m pytest tests/test_audit_pr_plan_presence.py tests/test_audit_pr_body.py tests/test_open_pr_wrapper.py tests/test_push_pr_wrapper.py tests/test_local_pr_review.py -q` — 90 passed.
- The exact two pytest commands enrolled in `.github/workflows/pre_push_audit.yml`, including the new admission and fixture tests — 486 passed.
- Python compilation passed for `scripts/_pr_change_policy.py`,
  `scripts/audit_pr_plan_presence.py`, `scripts/audit_plan_doc.py`,
  `scripts/audit_pr_body.py`, and `scripts/audit_review_rules_triggered.py`.
  Shell syntax passed for `scripts/local_pr_review.sh`, `scripts/new_pr_plan.sh`,
  `scripts/open_pr.sh`, `scripts/pre_push_audit.sh`, and `scripts/push_pr.sh`;
  `git diff --check` passed.
- `scripts/push_pr.sh` ran the real local-review bundle with the final PR body;
  plan/body admission, plan-shape, consistency, rule-trigger, and drift checks
  must all pass before the branch is pushed.
- The scripts-lane maturity ratchet must pass with the existing baseline;
  the diff-budget PR-body override must describe why the shared admission path
  cannot be split safely.
- CI repair verification: the focused plan-audit and plan-admission suites
  passed 19 tests, and the scripts-lane maturity ratchet passed against
  `tests/maturity_sweep/baseline_scripts.json` without a baseline update.
- Post-review repair verification proved mismatched branch-plan rejection across
  six unrelated existing plan paths, refresh-before-body-audit ordering,
  string-enum behavior, and the three now-passing unit-gate nodes before the
  guarded push. Current-head review repair also proves that a planless
  Markdown-only full body fails and a branch-plan-backed Markdown-only full
  body passes. The focused body-audit suite and scripts-lane maturity ratchet
  pass without a baseline update after checked singleton unpacking replaced the
  newly flagged fixed index.
- Latest CI repair proves regular-blob and single-suffix docs-only admission,
  rejects a Markdown symlink, a compound shell-script Markdown name, and a
  symlinked branch plan, and
  proves `open_pr.sh` repopulates a missing `origin/main` before the body audit.
- Final independent-review repair proves both the plan-admission and PR-body
  routes accept a deleted regular Markdown file and the documented completed-
  plan archive move using merge-base blobs, while a Markdown rename remains
  exempt and symlink/executable PR-head paths remain plan-required.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pr_body_contract.yml` | 5 |
| `.github/workflows/pre_push_audit.yml` | 4 |
| `AGENTS.md` | 32 |
| `CLAUDE.md` | 7 |
| `docs/REVIEWER_RULES.md` | 9 |
| `docs/SESSION_BOOTSTRAP.md` | 4 |
| `docs/ai_dev_operating_model.md` | 10 |
| `plans/PR-Plan-Admission-Contract.md` | 304 |
| `scripts/_pr_change_policy.py` | 206 |
| `scripts/audit_plan_doc.py` | 79 |
| `scripts/audit_pr_body.py` | 128 |
| `scripts/audit_pr_plan_presence.py` | 79 |
| `scripts/audit_review_rules_triggered.py` | 36 |
| `scripts/local_pr_review.sh` | 7 |
| `scripts/new_pr_plan.sh` | 10 |
| `scripts/open_pr.sh` | 16 |
| `scripts/pre_push_audit.sh` | 19 |
| `scripts/push_pr.sh` | 8 |
| `tests/test_audit_plan_doc.py` | 52 |
| `tests/test_audit_pr_body.py` | 371 |
| `tests/test_audit_pr_plan_presence.py` | 261 |
| `tests/test_audit_review_rules_triggered.py` | 25 |
| `tests/test_local_pr_review.py` | 33 |
| `tests/test_new_pr_plan.py` | 6 |
| `tests/test_open_pr_wrapper.py` | 81 |
| `tests/test_pr_body_contract_workflow.py` | 15 |
| `tests/test_pre_push_audit.py` | 84 |
| `tests/test_pre_push_audit_workflow.py` | 7 |
| `tests/test_push_pr_wrapper.py` | 102 |
| `tests/unit_gate_baseline.txt` | 3 |
| **Total** | **2003** |
