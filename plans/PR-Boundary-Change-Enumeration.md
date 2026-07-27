# PR-Boundary-Change-Enumeration

## Why this slice exists

Juan asked this codification session to turn observed builder failure patterns
into repo-enforced rules that survive context compaction. The first priority is
boundary-change enumeration: when a guard, validator, resolver, or admission
boundary changes, the plan must enumerate replaced-path behaviors,
guard-relevant fields, and every caller x input shape before code. The evidence
claims in the handoff remain claims, but the current code confirms there is no
existing AGENTS subsection or CI tripwire requiring this broader enumeration.
The qualifying blocker is the observed boundary-ripple class from ATLAS #2216
rounds 1-7 and Website #70 rounds 1-4: a builder can change a decision seam and
only discover replaced-path/caller/input-shape fallout after repeated review
rounds. This detector is open-input work over free-form diffs, so the closure
mechanism is a bounded recognizer: code suffix filtering is the choke point,
boundary-shaped paths or declarations warn by default, and tests cover each
detection branch plus section-scoped plan parsing as bounded evidence rather
than an exhaustive language parser. This slice is over the 400 LOC target
because the single rule is required to ship as law, template, tripwire,
scaffold regression, and both-direction detector tests together; splitting would
leave either an unenforced rule or a checker without the required authoring
prompt/evidence.

### Problem-derived contract

- Root cause: Atlas has guard class-closure and open-input method rules, but a
  boundary rewrite can still reach code review without a plan-time inventory of
  the behavior it replaces, the fields that affect its verdict, and the callers
  that feed it.
- Correct fix must touch/change: add the rule to `AGENTS.md`, add the prompt to
  the plan scaffold, add an advisory CI tripwire whose warning quotes the rule,
  and prove the tripwire fires and stays silent with unit fixtures plus the
  generated scaffold fields.
- Must not change: product/runtime behavior, reviewer strength, review-round
  caps, required branch protection checks, existing open PR branches, or
  EOM/timetracker working copies.

## Scope (this PR)

Ownership lane: dev-workflow/codification
Slice phase: Workflow/process

1. Add the boundary-change enumeration rule to Atlas builder law.
2. Add the corresponding plan-template prompt and advisory detector.
3. Add both-direction detector tests and generated-scaffold coverage.

### Review Contract

- Acceptance criteria:
  - `AGENTS.md` contains the verbatim boundary-change rule and states that it
    does not weaken the existing open-input evidence-gate.
  - `scripts/new_pr_plan.sh` scaffolds a `### Boundary-change enumeration`
    subsection with replaced-path behavior, guard-relevant field, and caller x
    input-shape prompts.
  - `.github/workflows/boundary_change_enumeration.yml` runs on
    `pull_request`, has read-only permissions, and runs the detector advisory.
  - `tests/test_check_boundary_change_enumeration.py` proves a violating
    boundary-shaped diff warns and a compliant plan stays silent.
  - `tests/test_new_pr_plan.py` proves the generated scaffold keeps the
    boundary-change enumeration heading, applicability instruction, and rows.
- Reachability proof: CI/workflow-only surface; the observable gate is the new
  workflow invoking python scripts/check_boundary_change_enumeration.py.
- Affected surfaces: builder workflow docs, plan scaffold, advisory CI,
  detector tests, and scaffold regression.
- Risk areas: false positives, silent false negatives, contradiction with
  open-input 3k.3, accidental required/blocking gate.
- Reviewer rules triggered: R1, R2, R10, R12.

### Boundary-change enumeration

N/A - this PR adds the boundary-change rule and detector, but it does not change
an Atlas product guard, validator, resolver, or admission boundary.

- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.

### Files touched

- `AGENTS.md`
- `scripts/new_pr_plan.sh`
- `scripts/check_boundary_change_enumeration.py`
- `tests/test_check_boundary_change_enumeration.py`
- `tests/test_new_pr_plan.py`
- `.github/workflows/boundary_change_enumeration.yml`
- `plans/PR-Boundary-Change-Enumeration.md`

## Mechanism

`AGENTS.md` gains a short imperative §3k.5 rule. `new_pr_plan.sh` adds a
matching optional subsection so the requirement appears while the plan is being
written. `check_boundary_change_enumeration.py` scans changed Python,
JavaScript, TypeScript-family, and shell files for added or removed
boundary-shaped path/function/method/class signals, including normalizing and
routing seams, and warns unless the `### Boundary-change enumeration` section
carries non-placeholder rows for replaced-path behaviors, guard-relevant fields,
and caller x input shape. Every duplicate enumeration row must be independently
dispositioned so a later TODO cannot hide behind an earlier valid row. The
self-bootstrap exemption is limited to this detector; other checker/admission
files still scan. The workflow runs the detector on PRs and emits warnings only.

## Intentional

- The detector is heuristic and advisory-first; it exits 0 unless `--strict` is
  supplied.
- The detector requires section-scoped, non-placeholder rows rather than
  attempting to verify the semantic quality of each enumeration row.
- The rule explicitly preserves 3k.3 so open-input work still needs an
  evidence-gated/defaulted mechanism.

## Deferred

- Promotion to a required check is deferred to a later operator decision after
  advisory evidence exists.
- Porting or enforcing this rule in EOM repos is deferred unless Juan asks for
  separate law-only repo PRs.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_boundary_change_enumeration.py -q --noconftest - 22 passed.
- python -m pytest tests/test_new_pr_plan.py -q --noconftest - 16 passed.
- python scripts/check_boundary_change_enumeration.py --base origin/main - OK.
- python scripts/check_boundary_change_enumeration.py --base origin/main --strict - OK.
- python scripts/audit_plan_doc.py plans/PR-Boundary-Change-Enumeration.md - OK.
- python scripts/audit_plan_doc_files_touched.py plans/PR-Boundary-Change-Enumeration.md origin/main - OK.
- python scripts/audit_plan_doc_diff_size.py plans/PR-Boundary-Change-Enumeration.md origin/main - OK.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | +20 |
| `scripts/new_pr_plan.sh` | +9 |
| `scripts/check_boundary_change_enumeration.py` | +210 |
| `tests/test_check_boundary_change_enumeration.py` | +175 |
| `tests/test_new_pr_plan.py` | +5 |
| `.github/workflows/boundary_change_enumeration.yml` | +46 |
| `plans/PR-Boundary-Change-Enumeration.md` | +139 |
| **Total** | **~650** |
