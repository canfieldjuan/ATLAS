# PR-Frontend-Test-CI-Enrollment-Rule

## Why this slice exists

The Atlas Intel UI CI workflow (`.github/workflows/atlas_intel_ui_checks.yml`)
runs an **explicit per-test list**, not a glob. So adding a `test:<name>` script
to `atlas-intel-ui/package.json` does not make CI run it — the matching
`run: npm run test:<name>` step must be added to the workflow by hand. That
manual step has been dropped repeatedly: flagged in #1223, enrolled correctly in
#1226/#1227, dropped again in #1228, and re-added in a dedicated follow-up #1229.
Each recurrence ships a test that silently doesn't run in CI (green for the wrong
reason) and costs a follow-up PR. Unlike `extracted-checks`, the intel-ui
workflow has no automated enrollment check, so the discipline has to be
codified.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add an AGENTS.md §3e rule: adding/renaming a `test:*` script requires the
   matching workflow `run` step in the same PR; a script entry in
   `atlas-intel-ui/package.json` is not CI execution.
2. Add the same rule to the SESSION_BOOTSTRAP.md §1 recurring-lapse checklist so
   builder sessions inherit it next to the existing test-placement guidance.

### Files touched

- `plans/PR-Frontend-Test-CI-Enrollment-Rule.md`
- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`

## Mechanism

Docs/process only. AGENTS.md §3e ("Tests") gains a paragraph stating the
explicit-per-test-list fact and the same-PR enrollment requirement, with the
reviewer/self check (grep the workflow run list for the new test name).
SESSION_BOOTSTRAP.md §1.3 gains a matching bullet right after the existing
"Test placement" item, phrased for a fresh builder session.

## Intentional

- Docs only; no workflow logic or test changes.
- Names the asymmetry explicitly: `extracted-checks` auto-checks enrollment, the
  intel-ui workflow does not — that asymmetry is why the gap recurs only on the
  frontend.

## Deferred

- An automated enrollment check for the intel-ui workflow (fail CI if a
  `test:*` script in `atlas-intel-ui/package.json` has no matching `run` step), mirroring the
  `extracted-checks` enrollment audit. That is the durable fix and would make
  this doc rule unnecessary, but it is a tooling slice of its own.

## Parked hardening

None.

## Verification

- `git diff` limited to the two docs + this plan.
- AGENTS.md §3e renders the new paragraph before §3f; SESSION_BOOTSTRAP.md §1
  renders the new bullet after "Test placement".
- No code touched; the local PR review hook (plan/code consistency, PR drift)
  passes.

## Estimated diff size

| Area | LOC |
|---|---:|
| AGENTS.md §3e rule | ~16 |
| SESSION_BOOTSTRAP.md bullet | ~1 |
| Plan doc | ~70 |
| **Total** | ~87 |

Docs/process only; well under the 400-LOC soft cap.
