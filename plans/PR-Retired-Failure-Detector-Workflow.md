# PR-Retired-Failure-Detector-Workflow

## Why this slice exists

#1965 added the JSON-only retired failure-mode detector, but no PR automation
runs it yet. #1964 needs an always-on, cheap detector signal before any ledger
collector can group or post recurrence events. This slice adds the advisory PR
workflow and artifact contract only.

The detector remains a detector, not a guard: recurrence signals are uploaded
for review and do not fail the workflow. Operational failures still fail so a
broken detector cannot silently disappear.

## Scope (this PR)

Ownership lane: workflow/retired-failure-detectors
Slice phase: Workflow/process

1. Add a PR workflow that runs `scripts/detect_retired_failure_modes.py` against
   the PR diff.
2. Write the detector JSON report to an artifact directory and upload it with a
   pinned artifact action.
3. Add workflow contract tests for advisory behavior, artifact upload, and
   PR/test enrollment.
4. Enroll the workflow contract tests in the existing maturity-sweep PR command.

### Review Contract

- Acceptance criteria:
  - [ ] The workflow runs on `pull_request`, not `pull_request_target`.
  - [ ] The workflow fetches the base ref and runs the detector with `--json-out`.
  - [ ] The artifact upload uses a SHA-pinned `actions/upload-artifact`.
  - [ ] Detector signals stay advisory: there is no signal-count failure step.
  - [ ] The new workflow contract test is enrolled in PR CI.
- Affected surfaces: GitHub Actions workflow / advisory detector automation /
  workflow contract tests.
- Risk areas: PR-code execution posture, artifact drift, accidental hard gate.
- Reviewer rules triggered: R2, R10, R12, R14.

### Files touched

- `.github/workflows/maturity_sweep_advisory.yml`
- `.github/workflows/retired_failure_detector.yml`
- `plans/PR-Retired-Failure-Detector-Workflow.md`
- `tests/test_retired_failure_detector_workflow.py`

## Mechanism

The new workflow checks out the PR, fetches the base branch into
`refs/remotes/origin/<base>`, creates `artifacts/retired-failure-detector`, and
runs:

```bash
python scripts/detect_retired_failure_modes.py \
  --base "origin/${BASE_REF}" \
  --json-out artifacts/retired-failure-detector/retired-failure-signals.json
```

The workflow uploads the artifact directory even when the JSON report contains
signals. That is the core detector/guard boundary: signals are data for the
future ledger, not merge blockers.

`tests/test_retired_failure_detector_workflow.py` parses the workflow YAML and
checks the execution posture, detector command, artifact upload, and CI
enrollment. The maturity-sweep workflow runs that test beside the detector
fixtures so workflow drift is caught on PRs.

## Intentional

- No GitHub issue comment, sticky PR comment, label, or disposition write is
  added here.
- No 30-minute watcher/timer is added here. That belongs to the separate
  long-running PR watcher operating model, not this detector workflow.
- No `pull_request_target` is used; this workflow needs no secrets and should
  not run with trusted-base token posture.

## Deferred

- Slice 3: scheduled/manual collector that reads uploaded detector artifacts and
  posts grouped signals to #1964.
- Slice 4: disposition documentation for true recurrence / false positive /
  needs guard / needs detector tuning.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_retired_failure_detector_workflow.py -q` — 4 passed.
- `python -m pytest tests/test_maturity_sweep.py tests/test_detect_retired_failure_modes.py tests/test_retired_failure_detector_workflow.py --noconftest -q` — 38 passed.
- `python scripts/audit_workflow_security_posture.py .github/workflows` — passed with existing repo-wide warnings only.
- `python scripts/detect_retired_failure_modes.py --base origin/main --json-out /tmp/retired-signals-s2.json` — exited 0 and emitted zero signals for this planned diff.
- `git diff --check` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/maturity_sweep_advisory.yml` | 1 |
| `.github/workflows/retired_failure_detector.yml` | 50 |
| `plans/PR-Retired-Failure-Detector-Workflow.md` | 102 |
| `tests/test_retired_failure_detector_workflow.py` | 85 |
| **Total** | **238** |
