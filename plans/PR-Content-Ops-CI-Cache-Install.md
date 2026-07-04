# PR-Content-Ops-CI-Cache-Install

## Why this slice exists

Issue #1962's CI runtime audit ranks product package installs as the next
safe speedup after the Maturity Sweep matrix split. Four content-ops workflows
spend 123-130 seconds installing dependencies before running 15-73 seconds of
targeted tests, and they all install the repo-wide `requirements.txt`.

Root cause: the slow workflows use the full Atlas runtime dependency file even
though their targeted content-ops checks do not exercise ASR, voice, video,
browser automation, or audio packages. The existing `cache: "pip"` only caches
downloaded artifacts; each clean runner still resolves and installs the whole
runtime, so unrelated heavy dependencies dominate these checks and unrelated
changes to `requirements.txt` invalidate their cache key.

This slice fixes the root for the sampled content-ops checks by giving them a
dedicated CI dependency subset and a cache key tied to that subset. It does not
claim the content-ops tests are torch-free: `tests/test_atlas_content_ops_infrastructure.py`
intentionally import-skips on `torch`, and current `atlas_brain.services`
imports still pull torch at collection.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add a `requirements.content_ops_ci.txt` subset for the four measured
   slow content-ops workflows.
2. Switch those workflows from `requirements.txt` to the content-ops CI
   subset and tie `actions/setup-python` pip caching to that file.
3. Add a workflow contract test proving the targeted workflows use the subset
   without removing their existing product pytest targets.
4. Archive the merged #1984 plan doc as same-lane teardown housekeeping.

### Review Contract

Acceptance criteria:

- The four sampled slow workflows still run their existing product pytest
  targets for Stripe paid, macro writeback, input provider, and deflection
  report, plus the new stdlib-only workflow contract test.
- The workflows install `requirements.content_ops_ci.txt`, not
  `requirements.txt`, and their `cache-dependency-path` points at the subset.
- The subset keeps dependencies required by current collection/runtime probes,
  including `torch`, `asyncpg`, `mcp`, `stripe`, `fpdf2`, `markdown`,
  `curl_cffi`, `beautifulsoup4`, and `feedparser`.
- The subset deliberately excludes heavyweight packages outside these workflow
  targets: `torchaudio`, `transformers`, `accelerate`, `bitsandbytes`,
  `nemo_toolkit[asr]`, `opencv-python`, `sounddevice`, `soundfile`,
  `webrtcvad`, `piper-tts`, `kokoro`, `playwright`, and `playwright-stealth`.
- A local blocked-import collection probe proves the targeted content-ops tests
  collect while those excluded packages are unavailable.

Affected surfaces:

- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `requirements.content_ops_ci.txt`

Risk areas:

- False speedup by weakening test coverage.
- Hidden import dependency on an excluded package that only appears at test
  runtime.
- Future drift back to `requirements.txt` or cache keys tied to the wrong file.

Triggered reviewer rules:

- R1 Requirements match.
- R2 Test evidence.
- R12 Scope discipline.
- R14 Codebase verification.

Reachability proof:

- Run the workflow contract test.
- Run a blocked-import pytest collection probe against the exact workflow test
  targets with the excluded packages unavailable.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `.github/workflows/atlas_content_ops_macro_writeback_checks.yml`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-CI-Cache-Install.md`
- `plans/archive/PR-Maturity-Sweep-Matrix.md`
- `requirements.content_ops_ci.txt`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_ci_requirements_workflows.py`

## Mechanism

Create a content-ops CI requirements file that keeps the runtime/test packages
these four workflows currently need but omits packages only used by voice,
ASR, video, browser scraping, or local audio playback. The workflows install
that file and set:

```yaml
cache: "pip"
cache-dependency-path: requirements.content_ops_ci.txt
```

This keeps the current pip-cache mechanism but prevents unrelated changes in
the repo-wide runtime file from invalidating these content-ops checks. The
workflow test reads the YAML as text and asserts both the install command and
cache dependency path, plus a canary that the existing product pytest target
names are still present.

## Intentional

- No torch-free split in this slice. The current input-provider workflow has
  an intentional `pytest.importorskip("torch")`, and `atlas_brain.services`
  still imports torch-backed service base classes during collection. Removing
  torch would skip or break coverage.
- No workflow trigger narrowing in this slice. That is candidate rank 3 in the
  audit and needs a separate proof that each workflow does not read the broad
  path it currently watches.
- No dependency deletion from the main runtime `requirements.txt`. Production
  and other workflows may still need the ASR, voice, video, browser, and
  scraping dependencies.

## Deferred

- Import-boundary hardening: make `atlas_brain.api` and
  `atlas_brain.services.llm` avoid eager optional-package imports so future
  content-ops checks can become torch-free where appropriate.
- Candidate rank 3 from the audit: narrow broad workflow path filters only
  after proving the workflow does not consume the excluded paths.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_ci_requirements_workflows.py -q`
  -- 3 passed.
- `python -m pytest --collect-only -q <content-ops workflow targets>` with
  heavyweight excluded packages blocked by an import hook -- 538 tests
  collected.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  -- OK: 196 matching tests are enrolled.
- `bash scripts/push_pr.sh <pr-body-file> -u origin HEAD`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 11 |
| `.github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml` | 11 |
| `.github/workflows/atlas_content_ops_input_provider_checks.yml` | 11 |
| `.github/workflows/atlas_content_ops_macro_writeback_checks.yml` | 11 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-CI-Cache-Install.md` | 162 |
| `plans/archive/PR-Maturity-Sweep-Matrix.md` | 0 |
| `requirements.content_ops_ci.txt` | 41 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_ci_requirements_workflows.py` | 121 |
| **Total** | **372** |
