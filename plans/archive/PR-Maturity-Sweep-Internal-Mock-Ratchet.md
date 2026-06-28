# PR-Maturity-Sweep-Internal-Mock-Ratchet

## Why this slice exists

Issue #1879 is the second real-adapters slice after #1885. The root cause is
that the repo now has a policy saying "mock the external edge, not first-party
code," but no blocking detector prevents new first-party mocks from entering
tests. Existing maturity-sweep ratchets can block score increases, but they
currently analyze production Python files and skip tests, so internal mocks in
tests would remain invisible unless they are attached back to the mocked
production module.

This PR fixes the root by adding an `INTERNAL_MOCK` finding to the existing
maturity-sweep model. The detector keys on the mocked target, not the test file
where the mock appears, so the existing per-file baseline ratchet can fail when
a PR adds a new mock of `extracted_content_pipeline.*`, `atlas_brain.*`, or
`scripts.*` while allowing explicitly external seams such as Stripe/httpx/LLM
clients.

This slice is over the 400 LOC target because the new signal must land with
refreshed baselines for every currently gated maturity lane that already has
internal-mock debt. Splitting the detector from its baselines would make CI red
on arrival; splitting the baselines from the detector would record meaningless
counts.

## Scope (this PR)

Ownership lane: real-adapters/test-quality
Slice phase: Workflow/process

1. Add target-aware internal-mock detection to `scripts/maturity_sweep.py`.
2. Prove internal targets fail and external targets pass with focused fixtures.
3. Refresh the affected maturity-sweep baselines so existing mock debt is
   tracked and only new internal mocks fail.

### Files touched

- `plans/PR-Maturity-Sweep-Internal-Mock-Ratchet.md`
- `scripts/maturity_sweep.py`
- `scripts/maturity_sweep_file_lane.py`
- `tests/maturity_sweep/baseline_ai_content_ops_lane.json`
- `tests/maturity_sweep/baseline_atlas_brain_agents.json`
- `tests/maturity_sweep/baseline_atlas_brain_alerts.json`
- `tests/maturity_sweep/baseline_atlas_brain_api.json`
- `tests/maturity_sweep/baseline_atlas_brain_auth.json`
- `tests/maturity_sweep/baseline_atlas_brain_autonomous.json`
- `tests/maturity_sweep/baseline_atlas_brain_comms.json`
- `tests/maturity_sweep/baseline_atlas_brain_discovery.json`
- `tests/maturity_sweep/baseline_atlas_brain_jobs.json`
- `tests/maturity_sweep/baseline_atlas_brain_mcp.json`
- `tests/maturity_sweep/baseline_atlas_brain_memory.json`
- `tests/maturity_sweep/baseline_atlas_brain_modes.json`
- `tests/maturity_sweep/baseline_atlas_brain_orchestration.json`
- `tests/maturity_sweep/baseline_atlas_brain_pipelines.json`
- `tests/maturity_sweep/baseline_atlas_brain_reasoning.json`
- `tests/maturity_sweep/baseline_atlas_brain_security.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_b2b.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_llm.json`
- `tests/maturity_sweep/baseline_atlas_brain_services_scraping.json`
- `tests/maturity_sweep/baseline_atlas_brain_skills.json`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/maturity_sweep/baseline_atlas_brain_tools.json`
- `tests/maturity_sweep/baseline_atlas_brain_voice.json`
- `tests/maturity_sweep/baseline_deflection_lane.json`
- `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json`
- `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
- `tests/maturity_sweep/baseline_scripts.json`
- `tests/test_maturity_sweep.py`

### Review Contract

Acceptance criteria:
- `mock.patch` / `@patch` / `monkeypatch.setattr` with a first-party target is
  counted as `INTERNAL_MOCK` on the mocked production module.
- External targets such as Stripe, httpx, urllib, and provider SDKs do not
  count.
- Sanctioned wall-clock/randomness seams such as `time`, `datetime`, `random`,
  `monotonic`, and `perf_counter` do not count.
- Blocking extracted-package lanes are first-party roots, not blind spots.
- Dotted no-asname imports and package `__init__.py` exports resolve to the
  intended target module.
- Ratchet mode fails when a new internal mock increases a file's
  `INTERNAL_MOCK` count, and `--update-baseline` remains the visible escape
  hatch for intentional debt.
- The changed detector has negative fixtures for internal, external, and
  ratchet-increase behavior.

Affected surfaces:
- `scripts/maturity_sweep.py`
- `tests/test_maturity_sweep.py`
- Maturity-sweep baseline JSON files touched by the new finding.
- `scripts/maturity_sweep_file_lane.py`

Risk areas:
- False positives from string targets that look first-party but do not map to a
  scanned module.
- False negatives for dynamic monkeypatch targets. This PR handles the common
  static target forms; broad dynamic inference is deferred.

Reviewer rules: R1, R2, R9, R10, R13, R14.

## Mechanism

- Index tests as ASTs in addition to source text.
- Extract mock targets from static forms:
  - `mock.patch("first.party.module.attr")`
  - `patch("first.party.module.attr")` and `@patch("first.party.module.attr")`
  - `patch.object(first_party_module, "attr")`
  - `monkeypatch.setattr("first.party.module.attr", ...)`
  - `monkeypatch.setattr(first_party_module, "attr", ...)`
- Resolve those targets to scanned production files when the target module is
  under `atlas_brain`, `scripts`, or one of the gated `extracted_*` packages.
- Suppress known external seams imported through a first-party module, such as
  `socket`, `httpx`, `stripe`, `asyncpg`, `find_spec`, wall-clock, and
  randomness seams, so edge fakes do not count as first-party mocks.
- Resolve dotted no-asname imports by keeping the package root mapped to the
  root, so `import extracted_content_pipeline.api.control_surfaces` does not
  double-prefix the target path.
- Index `__init__.py` both as `package.__init__` and `package`, so package
  export patches such as `patch("atlas_brain.auth.require_auth")` attach to the
  package file instead of disappearing.
- Add `INTERNAL_MOCK` findings to the target file's `FileResult`, so existing
  baseline score/count comparisons catch new first-party mocks without changing
  the lane CLI.
- Apply the same attachment step in `maturity_sweep_file_lane.py`, because the
  deflection lane is an explicit-file sweep rather than a directory sweep.

## Intentional

- `MagicMock()` is counted when it is part of a target-aware
  `monkeypatch.setattr` call. Bare `MagicMock()` with no first-party target is
  not counted in this slice because it cannot be safely assigned to a mocked
  module without data-flow analysis.
- Existing internal mocks are accepted by refreshing baselines. The value of
  this slice is preventing new debt; burn-down belongs to follow-up cleanup.
- Baseline accounting: the refreshed baseline diff has 227 changed entries.
  195 are `INTERNAL_MOCK`-only. The remaining 32 are disclosed, intentional
  ratchet refresh rather than hidden detector scope:
  - `scripts/maturity_sweep.py` records the new detector's own added AST
    complexity in the scripts baseline.
  - The newly enforced extracted-package root adds current internal-mock debt to
    `baseline_extracted_competitive_intelligence.json`; this is the intended
    current floor for the formerly blind blocking lane.
  - Several deflection/script entries were already present production files
    from prior landed slices that were not yet represented in the broad
    maturity baselines; refreshing them here keeps the ratchet green while
    preserving their current scores.
  - A few atlas `security.py` entries decrease because existing
    HAPPY_PATH/NO_RAISES attribution now sees the matched tests; those are
    floor-lowering fixes, not accepted new brittleness.
  The review-visible escape hatch is still the baseline JSON diff; future PRs
  fail on score increases or new `INTERNAL_MOCK` counts.
- The explicit deflection file-lane only attributes mocks whose production
  target is in the swept file list. A new internal mock against an unchanged
  production file outside that explicit list is caught by the broad maturity
  sweep, not by the file-lane wrapper alone.

## Deferred

- Dynamic target inference for variables that build patch strings at runtime.
- Baseline burn-down for existing internal mocks after the ratchet is in place.
- Bare `MagicMock()` attribution when no static first-party target is present.

Parked hardening: none.

## Verification

- Maturity sweep unit tests plus explicit-file lane tests - 28 passed.
- Extracted content pipeline ratchet gate against
  `tests/maturity_sweep/baseline_extracted_content_pipeline.json` - passed.
- Deflection explicit-file ratchet gate against
  `tests/maturity_sweep/baseline_deflection_lane.json` - passed.
- Scripts ratchet gate against `tests/maturity_sweep/baseline_scripts.json` -
  passed.
- AI content ops explicit-file ratchet gate against
  `tests/maturity_sweep/baseline_ai_content_ops_lane.json` - passed.
- Refreshed atlas-brain baselines were rechecked against the workflow lane
  matrix, including the `atlas_brain/skills` lane; all atlas-brain ratchets
  passed.
- Extracted package ratchet gates, including the newly enforced C1-C3 roots -
  passed.
- Deflection explicit-file and AI content-ops explicit-file ratchet gates -
  passed.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.
- Pending before push: blocking local review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Maturity-Sweep-Internal-Mock-Ratchet.md` | 222 |
| `scripts/maturity_sweep.py` | 214 |
| `scripts/maturity_sweep_file_lane.py` | 1 |
| `tests/maturity_sweep/baseline_ai_content_ops_lane.json` | 12 |
| `tests/maturity_sweep/baseline_atlas_brain_agents.json` | 7 |
| `tests/maturity_sweep/baseline_atlas_brain_alerts.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_api.json` | 112 |
| `tests/maturity_sweep/baseline_atlas_brain_auth.json` | 8 |
| `tests/maturity_sweep/baseline_atlas_brain_autonomous.json` | 183 |
| `tests/maturity_sweep/baseline_atlas_brain_comms.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_discovery.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_jobs.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 56 |
| `tests/maturity_sweep/baseline_atlas_brain_memory.json` | 15 |
| `tests/maturity_sweep/baseline_atlas_brain_modes.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_orchestration.json` | 3 |
| `tests/maturity_sweep/baseline_atlas_brain_pipelines.json` | 12 |
| `tests/maturity_sweep/baseline_atlas_brain_reasoning.json` | 35 |
| `tests/maturity_sweep/baseline_atlas_brain_security.json` | 9 |
| `tests/maturity_sweep/baseline_atlas_brain_services_b2b.json` | 41 |
| `tests/maturity_sweep/baseline_atlas_brain_services_llm.json` | 14 |
| `tests/maturity_sweep/baseline_atlas_brain_services_scraping.json` | 50 |
| `tests/maturity_sweep/baseline_atlas_brain_skills.json` | 9 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 20 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 10 |
| `tests/maturity_sweep/baseline_atlas_brain_voice.json` | 6 |
| `tests/maturity_sweep/baseline_deflection_lane.json` | 108 |
| `tests/maturity_sweep/baseline_extracted_competitive_intelligence.json` | 18 |
| `tests/maturity_sweep/baseline_extracted_content_pipeline.json` | 108 |
| `tests/maturity_sweep/baseline_scripts.json` | 52 |
| `tests/test_maturity_sweep.py` | 201 |
| **Total** | **1541** |
