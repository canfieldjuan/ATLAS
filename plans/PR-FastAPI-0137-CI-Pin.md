# PR-FastAPI-0137-CI-Pin

## Why this slice exists

PR #1556's generated-assets CI lane went red even after the deflection streaming
code passed its own extracted checks. The review traced the root cause to
dependency drift: `requirements.txt` declares unpinned `fastapi`, so CI installed
FastAPI 0.137.0 on the day it changed `router.routes` from a flat list of
`APIRoute` objects into a tree of intermediate router objects. The generated
assets host tests inspect mounted routes and therefore started failing on every
PR that triggers that workflow. Follow-up review found the same drift class in a
direct extracted-checks workflow install and two sibling service requirement
files. A downstream test-helper patch in #1556 was the wrong layer; this slice
pins FastAPI below the breaking release across the repo surfaces that install it
so main and PR CI are deterministic again.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Pin FastAPI below 0.137 in the root runtime requirements used by CI.
2. Pin the same FastAPI ceiling in ASR and sibling service requirement files so
   the same dependency-drift root is not left elsewhere in the repo.
3. Pin the direct extracted-checks workflow install that bypasses requirement
   files.
4. Update the extracted-checks CI-contract test so it asserts the pinned
   FastAPI package spec, not the old unbounded bare package name.
5. Trigger generated-assets checks when the requirement files that feed that
   workflow change.
6. Add no product code changes; #1556 keeps its deflection-streaming scope
   clean.

### Review Contract

- Acceptance criteria:
  - `requirements.txt` no longer allows FastAPI 0.137.0+.
  - `requirements.asr.txt` no longer allows FastAPI 0.137.0+.
  - Other repo requirement files that declare FastAPI keep their existing lower
    bounds but also reject FastAPI 0.137.0+.
  - Extracted-checks CI no longer installs unbounded FastAPI directly.
  - The extracted-checks route CI contract test fails on a bare unpinned
    `fastapi` install and passes on the quoted `fastapi<0.137` spec.
  - Generated-assets CI runs on root requirement-file changes.
  - The generated-assets workflow command still passes in the current local
    environment.
- Affected surfaces: requirement files and CI workflow dependency install/path
  triggers.
- Risk areas: dependency ceiling too broad or too narrow, and accidental changes
  to unrelated runtime packages.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `.github/workflows/extracted_pipeline_checks.yml`
- `atlas_video-processing/requirements.txt`
- `graphiti-wrapper/requirements.txt`
- `plans/PR-FastAPI-0137-CI-Pin.md`
- `requirements.asr.txt`
- `requirements.txt`
- `tests/test_extracted_pipeline_route_ci_contract.py`

## Mechanism

Requirement files change their FastAPI declarations to include `<0.137`. Root
requirements use `fastapi<0.137`; service-specific files preserve their existing
lower bounds and add the same ceiling. The extracted-checks workflow direct pip
install also quotes `fastapi<0.137` so the shell treats it as a package spec and
not redirection. Its route CI contract test parses the shell command with
`shlex` and asserts the pinned package spec is present while the bare `fastapi`
token is absent. That preserves the currently green FastAPI 0.136.x series while
excluding the 0.137.0 router internals change that made CI route-introspection
tests fail. The pin is a ceiling rather than a hard `==0.136.3` so
patch-compatible versions below the breaking release remain installable.

## Intentional

- This PR does not change `tests/test_atlas_content_ops_generated_assets_api.py`.
  FastAPI explicitly treats direct `router.routes` iteration as an internal
  detail after 0.137, so changing one test helper would be a symptom fix rather
  than the dependency determinism root.
- This PR does not loosen the generated-assets workflow. The lane should stay
  red when the installed dependency set drifts into incompatible behavior.
- The generated-assets workflow now listens to root requirement files because
  that workflow installs `requirements.txt`; without the path trigger, the next
  dependency-only PR could skip the exact lane it affects.

## Deferred

- Future dependency-upgrade slice: deliberately evaluate FastAPI 0.137+ and
  migrate route-introspection tests to a public API such as OpenAPI route
  discovery if we choose to upgrade.

Parked hardening: none.

## Verification

- python -m pytest tests/test_atlas_content_ops_generated_assets_api.py tests/test_content_ops_brand_voice_profiles.py tests/test_content_ops_brand_voice_profiles_api.py tests/test_content_ops_zendesk_credentials.py tests/test_content_ops_zendesk_export_api.py -q (62 passed)
- python inline requirement-line assertion for all FastAPI requirement
  declarations (all include `<0.137`)
- rg verification that extracted-checks installs `'fastapi<0.137'` directly.
- python -m pytest tests/test_extracted_pipeline_route_ci_contract.py -q
  (3 passed)
- bash scripts/run_extracted_pipeline_checks.sh (4183 passed, 10 skipped)
- Pending before push: local PR review bundle.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 4 |
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `atlas_video-processing/requirements.txt` | 2 |
| `graphiti-wrapper/requirements.txt` | 2 |
| `plans/PR-FastAPI-0137-CI-Pin.md` | 120 |
| `requirements.asr.txt` | 2 |
| `requirements.txt` | 2 |
| `tests/test_extracted_pipeline_route_ci_contract.py` | 6 |
| **Total** | **140** |
