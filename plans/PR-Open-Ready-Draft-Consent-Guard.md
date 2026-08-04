# PR-Open-Ready-Draft-Consent-Guard

## Why this slice exists

The AGENTS mechanical-enforcement audit found that PRs are supposed to open
ready for review by default, but `open_pr.sh` still forwards `--draft` without
any explicit operator-consent signal. Draft PRs delay automated review and have
already cost this lane manual attention.

### Problem-derived contract

- Root cause: The PR mutation wrapper rejects target-changing args, but it does
  not classify draft mode as a consent-gated create option, so a builder can
  accidentally open a draft PR even though AGENTS says ready-for-review is the
  default.
- Correct fix must touch/change: `scripts/open_pr.sh` must reject `--draft` /
  `-d` unless an explicit operator-consent environment flag is present.
  `tests/test_open_pr_wrapper.py` must prove reject-by-default happens before
  any GitHub mutation and prove the explicit flag forwards draft mode when the
  operator intentionally allows it.
- Must not change: Do not alter PR body validation, target repo/base/head
  safety, local review ordering, existing PR edit behavior, branch protection,
  or any product/runtime code.

## Scope (this PR)

Ownership lane: dev-workflow/open-ready-draft-consent
Slice phase: Workflow/process

1. Add an explicit draft-consent gate to `open_pr.sh` create-argument parsing.
2. Add wrapper tests for rejected draft creation by default and allowed draft
   creation with the explicit consent flag.

### Review Contract

- Acceptance criteria:
  - `scripts/open_pr.sh` rejects `--draft` and `-d` before GitHub mutation when
    `ATLAS_OPEN_PR_DRAFT_CONSENT` is not set to `1`.
  - `scripts/open_pr.sh` forwards `--draft` to `gh pr create` when
    `ATLAS_OPEN_PR_DRAFT_CONSENT=1` is set.
  - Existing safe ready-for-review create and edit flows continue to pass.
- Reachability proof: `tests/test_open_pr_wrapper.py` invokes the real wrapper
  script against a fake `gh`; observable effects are process exit code, stderr,
  and captured `gh pr create` argv/stdin.
- Affected surfaces: `scripts/open_pr.sh` create-argument boundary and
  `tests/test_open_pr_wrapper.py` fixtures.
- Risk areas: create-argument parsing, accidental draft mode, consent flag
  misuse, existing PR edit/create wrapper regression.
- Reviewer rules triggered: R1, R2, R6, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `scripts/open_pr.sh` create-argument admission.
- Replaced-path behaviors: `--draft` / `-d` no longer pass through by default.
- Guard-relevant fields: wrapper argv and `ATLAS_OPEN_PR_DRAFT_CONSENT`.
- Caller x input shape: local builder running `bash scripts/open_pr.sh
  BODY_FILE [gh-pr-create-args...]`.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `ATLAS_OPEN_PR_DRAFT_CONSENT` defaults absent.
- Explicit value probe: test sets `ATLAS_OPEN_PR_DRAFT_CONSENT=1` and proves
  draft mode is forwarded.
- Absent value probe: tests omit the flag for `--draft` and `-d` and prove the
  wrapper exits before fake `gh` is invoked.
- Default-session/default-context probe: existing ready create/edit wrapper
  tests continue to run without the flag.
- Side-effect ordering: draft rejection happens in argument admission before
  base refresh, local review, or GitHub mutation.

### Files touched

- `plans/PR-Open-Ready-Draft-Consent-Guard.md`
- `scripts/open_pr.sh`
- `tests/test_open_pr_wrapper.py`

## Mechanism

`reject_target_overrides` treats `--draft` and `-d` as consent-gated create
arguments. Without `ATLAS_OPEN_PR_DRAFT_CONSENT=1`, it prints a targeted error
and exits before any GitHub call. With the flag set, the wrapper leaves the
argument in place so the existing `gh pr create` call can intentionally create a
draft PR.

## Intentional

- The consent signal is an environment flag, not a new persistent session-state
  parser; it keeps this slice small and makes the exceptional draft request
  explicit at the command boundary.
- The wrapper still opens ready PRs by default; no extra flag is needed for the
  normal path.

## Deferred

None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_open_pr_wrapper.py` - 28 passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-open-ready-draft-consent.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-body-open-ready-draft-consent.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Open-Ready-Draft-Consent-Guard.md` | 118 |
| `scripts/open_pr.sh` | 10 |
| `tests/test_open_pr_wrapper.py` | 15 |
| **Total** | **143** |
