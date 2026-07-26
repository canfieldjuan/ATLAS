# PR-Deployed-Config-Probing

## Why this slice exists

Juan asked for the second priority codification rule: guard PRs must name
deployed/default config values, probe explicit/absent/default-session shapes,
and avoid side effects before all admissions pass. The handoff evidence is a
claim; the current Atlas code confirms the existing plan scaffold and advisory
checks do not enforce this exact deployed-config probe shape.
The qualifying blocker is the observed deployed/default mismatch class from
Website CORS #15, eom-timetracker #63, and ATLAS #2216's default-context
claim-before-reject issue: builders can verify only a local explicit path while
the deployed/default admission shape still mutates or admits incorrectly. This
detector is open-input work over free-form diffs, so the closure mechanism is a
bounded recognizer: code suffix filtering is the choke point, ambiguous
boundary/config-looking changes warn by default, and tests cover repo-native
Python, TypeScript, shell, removed-line, hunk-context, and substring-collision
forms as bounded evidence rather than an exhaustive language parser. This slice
is over the 400 LOC target because the single rule is required to ship as law,
template, tripwire, scaffold regression, retirement review, and both-direction
detector tests together; splitting would leave either an unenforced rule or a
checker without the required authoring prompt/evidence.

### Problem-derived contract

- Root cause: Atlas can warn about some guard-class and plan-shape issues, but a
  guard/config-boundary PR can still verify only an explicit or local-default
  path without recording deployed/default facts, absent/default-session probes,
  or side-effect ordering in the plan.
- Correct fix must touch/change: add the rule to `AGENTS.md`, add the prompt to
  the plan scaffold, add an advisory CI tripwire whose warning quotes the rule,
  and test violating and compliant examples plus the generated scaffold fields.
- Must not change: product/runtime behavior, required CI status checks,
  workflow write permissions, reviewer strength, review-round caps, existing
  open PR branches, or EOM/timetracker working copies.

## Scope (this PR)

Ownership lane: dev-workflow/codification
Slice phase: Workflow/process

1. Add the deployed-config probing rule to Atlas builder law.
2. Add the matching plan-template prompt and advisory detector.
3. Add both-direction detector tests and generated-scaffold coverage.

### Review Contract

- Acceptance criteria:
  - `AGENTS.md` contains the verbatim deployed-config probing rule and includes
    a code-grounded/could-not-determine path for deployed values not present in
    repo-owned config.
  - `scripts/new_pr_plan.sh` scaffolds a `### Deployed-config probing`
    subsection with deployed/default values, explicit, absent,
    default-session/default-context, and side-effect-ordering prompts.
  - `.github/workflows/deployed_config_probing.yml` runs on `pull_request`, has
    read-only permissions, and runs the detector advisory.
  - `tests/test_check_deployed_config_probing.py` proves env fallback and guard
    changes warn, while a plan carrying the required probe markers stays silent.
  - `tests/test_new_pr_plan.py` proves the generated scaffold keeps the
    deployed-config probing heading and fields.
- Reachability proof: CI/workflow-only surface; the observable gate is the new
  workflow invoking python scripts/check_deployed_config_probing.py.
- Affected surfaces: builder workflow docs, plan scaffold, advisory CI,
  detector tests, and scaffold regression.
- Risk areas: false positives, silent false negatives, overstating deployed
  values that code cannot prove, accidental required/blocking gate.
- Reviewer rules triggered: R1, R2, R10, R11, R12.

### Boundary-change enumeration

N/A - this PR adds a process detector but does not change an Atlas product
guard, validator, resolver, or admission boundary.

- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.

### Deployed-config probing

N/A - this PR adds the deployed-config probing rule/detector but does not change
a product guard/config boundary.

- Deployed/default config values: N/A.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: N/A.
- Side-effect ordering: N/A.

### Files touched

- `AGENTS.md`
- `scripts/new_pr_plan.sh`
- `scripts/check_deployed_config_probing.py`
- `tests/test_check_deployed_config_probing.py`
- `tests/test_new_pr_plan.py`
- `.github/workflows/deployed_config_probing.yml`
- `REVIEW_MISSES.md`
- `plans/PR-Deployed-Config-Probing.md`

## Mechanism

`AGENTS.md` gains a short imperative §3k.6 rule. `new_pr_plan.sh` adds the
matching subsection so builders see the probe checklist while writing the plan.
`check_deployed_config_probing.py` scans changed Python, JavaScript,
TypeScript, and shell files for env/config fallbacks and boundary-shaped
guard/resolver/admission signals, including removed fallback lines and hunk
context, then warns unless a changed plan contains non-placeholder
deployed/default, explicit, absent, default-session/default-context, and
side-effect-ordering dispositions. The workflow runs the detector as
advisory-only. The recognizer deliberately favors warning on bounded
boundary/config signals instead of claiming a complete parser for every
language. `tests/test_new_pr_plan.py` proves the generated scaffold carries the
deployed-config fields. `REVIEW_MISSES.md` records the required retirement-review
pass for this durable mechanism addition.

## Intentional

- The detector is heuristic and advisory-first; it exits 0 unless `--strict` is
  supplied.
- The rule allows `could-not-determine` for deployed values not recorded in
  repo-owned config, rather than inviting invented deployment claims.
- This PR does not inspect live deployed config; it codifies the plan-time duty
  for future guard/config-boundary PRs.

## Deferred

- Promotion to a required check is deferred to a later operator decision after
  advisory evidence exists.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_deployed_config_probing.py -q --noconftest - 14 passed.
- python -m pytest tests/test_new_pr_plan.py -q --noconftest - 16 passed.
- python scripts/check_deployed_config_probing.py --base claude/pr-boundary-change-enumeration - OK.
- python scripts/audit_plan_doc.py plans/PR-Deployed-Config-Probing.md - OK.
- python scripts/audit_plan_doc_files_touched.py plans/PR-Deployed-Config-Probing.md claude/pr-boundary-change-enumeration - OK.
- python scripts/audit_plan_doc_diff_size.py plans/PR-Deployed-Config-Probing.md claude/pr-boundary-change-enumeration - OK.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | +15 |
| `scripts/new_pr_plan.sh` | +11 |
| `scripts/check_deployed_config_probing.py` | +201 |
| `tests/test_check_deployed_config_probing.py` | +130 |
| `tests/test_new_pr_plan.py` | +6 |
| `.github/workflows/deployed_config_probing.yml` | +46 |
| `REVIEW_MISSES.md` | +6 |
| `plans/PR-Deployed-Config-Probing.md` | +132 |
| **Total** | **~568** |
