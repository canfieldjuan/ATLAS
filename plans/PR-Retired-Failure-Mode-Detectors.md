# PR-Retired-Failure-Mode-Detectors

## Why this slice exists

The operator rejected the previous Fable 5 lessons note as too narrow and asked
for an investigation brief on a persistent, non-blocking detection layer for
retired failure modes. Atlas already has blocking gates for plan-as-contract,
open-thread reconciliation, lane/scope enforcement, brittle-code checks, and
root-cause tracing, but a retired failure mode can still recur under merge
pressure, longer arcs, ambiguous plans, or newer model behavior. This slice
captures candidate detector architectures before any implementation guard is
chosen.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Workflow/process

1. Replace the previous Fable-specific artifact with an investigation document
   focused on persistent retired-mode detectors.
2. Inspect and cite existing repo hooks the detector layer can reuse without
   changing critical-path gate behavior.
3. Present 2-4 candidate architectures without choosing one.

### Files touched

- `docs/retired_failure_mode_detection_layer.md`
- `plans/PR-Retired-Failure-Mode-Detectors.md`

## Mechanism

The document maps current CI and local-review seams, defines cheap signatures for
plan-weakening, test-weakening, scope drift, and symptom patching, then describes
four non-blocking recording architectures: CI summary/artifact, sticky PR
comment, labels plus issue ledger, and scheduled/offline ledger. This is a
documentation-only investigation; no runtime gates or workflows are changed.

## Intentional

- The slice does not pick a winning architecture because the operator explicitly
  asked for tradeoffs and a later choice.
- The slice does not implement detectors or guards. The brief distinguishes
  detector signals from blocking guards and keeps this PR additive.
- The prior Fable 5 lesson note is removed because the replacement brief is the
  requested durable artifact.

## Deferred

- Implementing a detector script and GitHub recording backend is deferred until
  the operator chooses one of the candidate architectures.
- Posting the issue-ready content to a GitHub issue remains deferred to an
  environment with GitHub issue access.

Parked hardening: none.

## Verification

- Sync plan helper completed against `HEAD~1`.
- Sync plan check completed against `HEAD~1`.
- Local PR review helper completed against `HEAD~1` with environment warnings
  noted in the final report.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/retired_failure_mode_detection_layer.md` | 262 |
| `plans/PR-Retired-Failure-Mode-Detectors.md` | 69 |
| **Total** | **331** |
