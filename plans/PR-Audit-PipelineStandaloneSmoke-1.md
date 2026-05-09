# PR-Audit-PipelineStandaloneSmoke-1: manifest-driven standalone smoke for `extracted_content_pipeline`

## Why this slice exists

PRs #422 (`brand_registry` shim) and #425 (`_b2b_witnesses` shim)
both closed standalone-import failures that **slipped through CI**
even though the failure mode was deterministic and reproducible:

```
$ EXTRACTED_PIPELINE_STANDALONE=1 python3 -c \
    "from extracted_content_pipeline.autonomous.tasks import competitive_intelligence"
ModuleNotFoundError: No module named 'extracted_content_pipeline.services.brand_registry'
```

Root cause: `extracted_content_pipeline` has **no standalone-mode
smoke**. The default-mode smoke
(`scripts/smoke_extracted_pipeline_imports.py`) imports a hardcoded
list of modules in default mode (no env var), and `competitive_intelligence`
was never added to that list when it was mirrored on 2026-04-30 (commit
`d3b0cab1`). Even if it had been, the default-mode import path doesn't
exercise the standalone-substrate paths -- atlas_brain happens to be
on `sys.path`, so relative imports that *would* fail under the
standalone toggle silently resolve to atlas_brain modules instead.

`extracted_competitive_intelligence` already has a standalone smoke
(`scripts/smoke_extracted_competitive_intelligence_standalone.py`), but
it too uses a hardcoded MODULES list -- the same drift hazard.

## Scope (this PR)

One new script:
`scripts/smoke_extracted_pipeline_standalone.py`. Walks
`extracted_content_pipeline/manifest.json`, builds the import list
from `mappings + owned` (filtering out migrations and `__init__.py`),
sets `EXTRACTED_PIPELINE_STANDALONE=1`, imports every target.

Plus a one-line add to `scripts/run_extracted_pipeline_checks.sh`
wiring it into the CI gate.

## Manifest-driven, not hardcoded

The existing `smoke_extracted_pipeline_imports.py` and
`smoke_extracted_competitive_intelligence_standalone.py` use
hardcoded MODULES lists. New mirrors require a manual addition to
each smoke script -- the failure mode that bit #422 and #425 was
exactly this drift.

This script reads the manifest at runtime. Any new entry under
`mappings` or `owned` is automatically covered. The hardcoded-list
pattern is preserved for the existing default-mode smokes
(refactoring them is a separate slice -- bigger blast radius, not
required to close the standalone gap).

## Failure classification

The script distinguishes two import failure shapes:

1. **Real decoupling failure** -- `ModuleNotFoundError` referencing
   `extracted_*` or `atlas_brain`. These are bugs the PR is meant
   to catch.
2. **3rd-party package missing** -- `ModuleNotFoundError` for a name
   that is not `extracted_*` or `atlas_brain` (e.g., `httpx`,
   `pydantic`). These are environment issues, not decoupling issues.

The script reports both categories but only counts the first toward
the exit status (gate). 3rd-party-missing failures get a warning
line so CI surfaces are still readable, but they don't fail the
build -- a missing 3rd-party package is a separate concern handled
by `validate_extracted_content_pipeline.sh`.

## Intentional (looks wrong but is deliberate)

- **Reads manifest directly** rather than walking the filesystem.
  Filesystem walks would also pick up untracked .py files (50+ in
  this package today, including standalone shims and product-native
  modules). The manifest is the canonical "what should import
  cleanly under standalone" list. Untracked files are a separate
  methodology gap.
- **No refactor of `smoke_extracted_pipeline_imports.py`.** That
  script imports in default mode (without the standalone toggle) and
  has its own hardcoded list. Converting it to manifest-driven is
  desirable but doesn't fit "close the standalone-mode gap" framing
  -- separate slice.
- **No refactor of
  `smoke_extracted_competitive_intelligence_standalone.py`.** Same
  reasoning -- it works today, just with the same drift hazard.
  Converting both is a separate consolidation slice.
- **Only fails the build on real decoupling failures.** A 3rd-party
  package missing in the test env is not a decoupling debt; treating
  it as a gate failure would conflate concerns.
- **Imports each module in a subprocess.** Some modules execute
  module-level code that isn't idempotent across reimports (e.g.,
  module-level `os.environ.setdefault` calls). Subprocess-per-module
  isolates side effects and matches the
  `smoke_extracted_competitive_intelligence_standalone.py`
  precedent.

## Deferred (looks missing but is on purpose)

- **Refactor existing smoke scripts to be manifest-driven.** Real
  cleanup but bigger blast radius -- separate slice.
- **Catch the methodology gap of 50+ untracked .py files.** Different
  concern -- the manifest itself is the contract; whether files
  exist on disk that aren't on the manifest is a separate question
  about the methodology.
- **Default-mode smoke parity.** `extracted_competitive_intelligence`
  has a Phase-1 default-mode smoke
  (`smoke_extracted_competitive_intelligence_imports.py`) and a
  standalone smoke. `extracted_content_pipeline` has only a
  default-mode smoke. This PR adds the standalone smoke;
  whether the existing default-mode smoke evolves is separate.

## Verification

- `EXTRACTED_PIPELINE_STANDALONE=1 python3 scripts/smoke_extracted_pipeline_standalone.py`
  -> reports per-module OK/FAIL, summary, exits 0 (after #425 lands;
  pre-#425 it would report `_b2b_pool_compression` failure).
- New script invocation in `run_extracted_pipeline_checks.sh` runs as
  part of the standard CI gate.
- `bash scripts/check_ascii_python.sh` -> passed.
- New regression test:
  `tests/test_extracted_pipeline_standalone_smoke.py` asserts the
  smoke script returns exit code 0 in the current state, and that
  a synthetic missing-module failure causes a non-zero exit.

## Conflict check

- No file overlap with any open PR.
- The new script reads the existing manifest and runs imports; no
  state change to any existing file beyond a one-line
  `run_extracted_pipeline_checks.sh` add.

## Diff size

- New script: ~80 LOC
- New regression test: ~50 LOC
- One-line add to `run_extracted_pipeline_checks.sh`
- Plan doc: ~110 LOC

Source-only ~80 LOC. Smallest scope that closes the gap that bit
#422 and #425.

## After this lands

New mirrors automatically get standalone-import coverage. The class
of failures that bit #422 and #425 cannot slip through CI again --
any future mirror with a missing relative-import target will fail the
gate before review.
