# PR-Pre-Push-Audit-Httpx

## Why this slice exists

The #2117 merge added `tests/test_eval_local_mcp_models.py` to the
pre_push_audit tooling-test bundle; that suite imports the eval script
whose live-runner path does a function-level `import httpx`
(`scripts/eval_local_mcp_models.py:1453`). The workflow runs under
`pull_request_target` (trusted-base execution, #1949), so #2117's own CI
ran the pre-merge bundle and never exercised the new test — every PR
since the merge now fails `pre-push-audit` with ModuleNotFoundError.
Both requirements files already pin httpx==0.28.1; the workflow env is
the only gap. Same class as #2117's in-PR jsonschema fix — which also
only took effect post-merge for the same trusted-base reason.

### Problem-derived contract

- Root cause: workflow tooling-test env missing a dependency the merged
  bundle needs; trusted-base execution hid it until after merge.
- Correct fix must: add httpx to both tooling-test install lines.
- Must not change: anything else.

## Scope (this PR)

Ownership lane: mcp/local-model-qualification
Slice phase: workflow/process

1. `.github/workflows/pre_push_audit.yml`: both tooling-test pip lines
   gain `httpx`.

### Review Contract

- Acceptance criteria: the tooling-test bundle imports
  `eval_local_mcp_models` end-to-end on a clean runner (post-merge, any
  PR's `pre-push-audit` passes the import).
- Reachability proof: `pre-push-audit` check on every PR after merge.
- Affected surfaces: CI env only.
- Risk areas: none (pinned dep already in requirements).
- Reviewer rules triggered: R11, R12, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Pre-Push-Audit-Httpx.md`

## Mechanism

One package name on two install lines.

## Intentional

- httpx unpinned on the install line, matching the bundle's existing
  style (jsonschema etc.); the version contract lives in requirements.

## Deferred

- Nothing.

Parked hardening: none new.

## Verification

- Local: `python -m pip install pytest pytest-asyncio pyyaml jsonschema
  httpx` + the bundle's eval tests pass (103 in the harness suite).
- The real proof is post-merge CI (trusted-base: this PR's own
  pre-push-audit still runs the OLD line and still fails — expected).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/PR-Pre-Push-Audit-Httpx.md` | 60 |
| **Total** | **~62** |
