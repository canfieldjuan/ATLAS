# PR-Enforced-Set-Verification

## Why this slice exists

Juan asked for the third priority codification rule: a plan's Verification
section must cite the CI-equivalent command copied from the enforcing workflow,
not a hand-picked subset. The current Atlas code confirms
`.github/workflows/atlas_eom_lead_pipeline_checks.yml` spells out its pytest
list inline and no `scripts/run_eom_lead_pipeline_checks.sh` mirror exists.
The qualifying blocker is the observed partial-local-gauntlet class from ATLAS
#2216 round-2 CI and eom-timetracker #63: builders can claim verification while
missing the workflow/deployed enforced set. For EOM lead-pipeline surfaces this
is a privacy/customer-communications risk because tenant scoping and outbound
mail paths are in the enforced workflow. This slice is over the 400 LOC target
because the single rule is required to ship as law, template, tripwire, mirror
script, and both-direction/negative-path tests together; splitting it would
leave either an unenforced law or an unprompted tripwire.

### Problem-derived contract

- Root cause: a builder can run focused tests and describe that as verification
  even when the enforcing workflow runs a wider set. Without a mirror command,
  the correct local command is easy to miss after compaction.
- Correct fix must touch/change: add the rule to `AGENTS.md`, add the prompt to
  the Verification template, add an advisory CI tripwire whose warning quotes
  the rule, add both-direction tests, and mirror the EOM lead workflow pytest
  invocation into one local script while deferring workflow rewiring because the
  in-flight #2216 PR also touches that workflow file.
- Must not change: product/runtime behavior, pytest file membership, required
  CI status checks, workflow write permissions, reviewer strength, review-round
  caps, existing open PR branches, or EOM/timetracker working copies.

## Scope (this PR)

Ownership lane: dev-workflow/codification
Slice phase: Workflow/process

1. Add the enforced-set verification rule to Atlas builder law.
2. Add the matching Verification prompt and advisory plan detector.
3. Add a local mirror script for the EOM lead pipeline workflow's existing
   pytest set; do not edit the in-flight #2216 workflow in this PR.
4. Record the required durable-mechanism retirement review.

### Review Contract

- Acceptance criteria:
  - `AGENTS.md` contains the verbatim enforced-set verification rule and allows
    an explicit no-enforcing-workflow statement when applicable.
  - `scripts/new_pr_plan.sh` scaffolds Verification lines for the
    CI-equivalent command and copied enforcing workflow.
  - `scripts/run_eom_lead_pipeline_checks.sh` contains the same pytest files
    and PostgreSQL preconditions that `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
    enforces on current main; workflow rewiring is deferred because #2216
    touches that workflow.
  - `tests/test_check_enforced_set_verification.py` proves a plan missing the
    CI-equivalent citation warns and a compliant plan stays silent.
  - `tests/test_new_pr_plan.py` proves newly generated plan scaffolds keep the
    enforced-set Verification prompts.
- Reachability proof: CI/workflow-only surface; the observable gate is the new
  workflow invoking python scripts/check_enforced_set_verification.py.
- Affected surfaces: builder workflow docs, plan scaffold, advisory CI,
  detector tests, and the EOM lead pipeline local mirror command.
- Risk areas: false positives, silent false negatives, workflow command drift,
  accidental pytest-set mismatch, accidental required/blocking gate.
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

### Files touched

- `AGENTS.md`
- `scripts/new_pr_plan.sh`
- `scripts/check_enforced_set_verification.py`
- `tests/test_check_enforced_set_verification.py`
- `tests/test_new_pr_plan.py`
- `scripts/run_eom_lead_pipeline_checks.sh`
- `.github/workflows/enforced_set_verification.yml`
- `REVIEW_MISSES.md`
- `plans/PR-Enforced-Set-Verification.md`

## Mechanism

`AGENTS.md` gains a short imperative §3k.7 rule. `new_pr_plan.sh` adds
Verification placeholders for the CI-equivalent command and the workflow it was
copied from. `check_enforced_set_verification.py` inspects changed plan docs and
warns unless the Verification section includes non-placeholder command/source
markers plus affirmative execution evidence tied to the selected command. The
EOM lead pipeline workflow's existing pytest list and PostgreSQL preconditions are mirrored in
`scripts/run_eom_lead_pipeline_checks.sh`, making the enforced set a single
local command without editing #2216's in-flight workflow file. `REVIEW_MISSES.md`
records the required retirement-review pass for this durable mechanism addition.

## Intentional

- The detector is heuristic and advisory-first; it exits 0 unless `--strict` is
  supplied.
- The mirror script preserves the existing pytest list; this PR does not edit
  the workflow file because #2216 also touches it.
- The detector checks for affirmative run/pass evidence in plan text; it cannot
  independently prove a human executed the command.

## Deferred

- Promotion to a required check is deferred to a later operator decision after
  advisory evidence exists.
- Additional workflow mirror scripts are deferred until those workflows are
  touched by future slices.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_enforced_set_verification.py tests/test_new_pr_plan.py -q --noconftest - 32 passed.
- bash scripts/run_eom_lead_pipeline_checks.sh - 197 passed, 1 warning.
- python scripts/check_enforced_set_verification.py --base claude/pr-deployed-config-probing - OK.
- python scripts/audit_plan_doc.py plans/PR-Enforced-Set-Verification.md - OK.
- python scripts/audit_plan_doc_files_touched.py plans/PR-Enforced-Set-Verification.md claude/pr-deployed-config-probing - OK.
- python scripts/audit_plan_doc_diff_size.py plans/PR-Enforced-Set-Verification.md claude/pr-deployed-config-probing - OK.
- CI-equivalent command copied from enforcing workflow: bash scripts/run_eom_lead_pipeline_checks.sh.
- Copied from enforcing workflow: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | +12 |
| `scripts/new_pr_plan.sh` | +4 |
| `.github/workflows/enforced_set_verification.yml` | +46 |
| `REVIEW_MISSES.md` | +1 |
| `plans/PR-Enforced-Set-Verification.md` | +152 |
| `scripts/check_enforced_set_verification.py` | +204 |
| `tests/test_check_enforced_set_verification.py` | +165 |
| `tests/test_new_pr_plan.py` | +4 |
| `scripts/run_eom_lead_pipeline_checks.sh` | +62 |
| **Total** | **~650** |
