# PR-Atlas-Local-PR-Body-Preflight

## Why this slice exists

Reviewer feedback flagged a repeated workflow miss: code quality is staying
high, but first pushes keep failing metadata/CI contracts after the PR is
already open. Recent misses include PR-body `Slice phase` omissions and extracted
test enrollment drift. The scanners catch the problems, but the builder path
needs to run the same checks before the first PR push.

This slice makes the existing local PR review bundle enforce the two recurring
metadata gates directly: current PR body phase validation and extracted pipeline
test enrollment.

## Scope (this PR)

Ownership lane: atlas-workflow

Slice phase: Workflow/process.

1. Let the drift audit validate a local PR body file before a GitHub PR exists.
2. Require current PR body validation from `scripts/local_pr_review.sh`, using
   either the open PR body or a supplied local body file.
3. Run the extracted pipeline CI-enrollment scanner from the local review bundle.
4. Add focused negative/positive fixtures for the new pre-open PR body path.

### Files touched

- `plans/PR-Atlas-Local-PR-Body-Preflight.md`
- `scripts/audit_pr_session_drift.py`
- `scripts/local_pr_review.sh`
- `tests/test_audit_pr_session_drift.py`

## Mechanism

`audit_pr_session_drift.py` gets two options:

```bash
--current-pr-body-file <path>
--require-current-pr-body
```

When a branch adds a plan doc with a `Slice phase`, `--require-current-pr-body`
fails unless either the open PR body is checked through GitHub metadata or a
local body file is supplied. The same slice-phase parser validates both paths,
so the pre-open body check and GitHub body check cannot diverge.

`local_pr_review.sh` passes `--require-current-pr-body` to the drift audit and
accepts `--current-pr-body-file` / `--pr-body-file` for the pre-open path. The
installed pre-push hook can use `ATLAS_CURRENT_PR_BODY_FILE=...` for the same
body file because hook invocations cannot receive ad hoc CLI flags. The wrapper
also runs `scripts/audit_extracted_pipeline_ci_enrollment.py` when present.

## Intentional

- This does not generate PR bodies. It enforces that the body exists and matches
  the plan phase before the builder treats local review as green.
- The current GitHub PR body remains authoritative after a PR exists. The local
  body file is for the pre-open review path.
- The enrollment scanner runs unconditionally when available. It is already a
  repo-wide audit and is fast enough for the local mechanical bundle.

## Deferred

- Future PR: consider a body-render helper if builders still hand-write PR
  descriptions incorrectly after this gate.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_pr_session_drift.py -q` - 26 passed.
- `python -m py_compile scripts/audit_pr_session_drift.py` - passed.
- `bash scripts/local_pr_review.sh --help` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file <temp body>` - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=<temp body> bash scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Drift audit body-file gate | ~97 |
| Local review wiring | ~35 |
| Tests | ~85 |
| Plan doc | ~85 |
| **Total** | **~302** |
