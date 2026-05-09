# PR-Audit-ManifestDrivenSmokes-1: convert default-mode smokes to manifest-driven

## Why this slice exists

Follow-up to PR-Audit-PipelineStandaloneSmoke-1 (#427). That PR added a
new manifest-driven **standalone-mode** smoke for
`extracted_content_pipeline`. The two existing **default-mode** smokes
still use hardcoded MODULES lists -- the same drift hazard that bit
#422 (`brand_registry`) and #425 (`_b2b_witnesses`):

```
$ wc -l scripts/smoke_extracted_pipeline_imports.py
62  # hardcoded MODULES list, 26 entries
$ wc -l scripts/smoke_extracted_competitive_intelligence_imports.py
78  # hardcoded MODULES list, 23 entries
```

Neither was updated when `competitive_intelligence.py` was mirrored on
2026-04-30 (commit `d3b0cab1`). Same gap, different mode.

## Scope (this PR)

Convert both default-mode smokes to **manifest-driven** discovery.
Walk `manifest.json`, build the import list at runtime, import every
target. New mirrors get auto-coverage.

Two scripts refactored:

1. `scripts/smoke_extracted_pipeline_imports.py` -> manifest-driven
   import sweep over `extracted_content_pipeline`, default mode.
2. `scripts/smoke_extracted_competitive_intelligence_imports.py` ->
   manifest-driven import sweep over
   `extracted_competitive_intelligence`, default mode (Phase-1 with
   `atlas_brain` on `sys.path`).

Each script preserves its existing CI integration and call site. Only
the source of the MODULES list changes.

Failure classification matches #427: a `ModuleNotFoundError`
referencing `extracted_*` or `atlas_brain` is a gate-breaking
**decoupling failure**; any other `ModuleNotFoundError` (e.g.,
`httpx`, `pydantic`) is a 3rd-party env failure that gets a warning
but doesn't break the gate. Other exception types (SyntaxError,
TypeError raised at import) always break the gate.

## Why `smoke_extracted_competitive_intelligence_standalone.py` is NOT refactored here

That script (209 LOC) interleaves three concerns:

- **Import sweep** -- hardcoded MODULES list (the drift hazard).
- **Owner verification** -- asserts certain symbols resolve to the
  expected standalone substrate (`_standalone.config`, etc.) rather
  than falling back to atlas_brain.
- **Atlas-fallback probes** -- asserts certain modules don't
  silently succeed when atlas_brain is unavailable.
- **"Still imports atlas_brain" file scan** -- string match on
  owned files.

Refactoring just the import sweep portion would leave the other
three concerns coupled to the now-removed list. Cleanly separating
them is a meaningful restructure, not a 1:1 refactor. Deferred to a
follow-up slice that focuses specifically on that script.

The standalone-mode coverage gap for `extracted_content_pipeline`
was closed by #427, which is the more important hole. The compintel
standalone smoke remains hardcoded but is exercised on every CI run
of `run_extracted_competitive_intelligence_checks.sh` -- it works
today, just with the same drift hazard as before.

## Intentional (looks wrong but is deliberate)

- **Manifest-driven, not filesystem walk.** Same reasoning as #427:
  filesystem walks pick up untracked .py files; the manifest is the
  canonical contract.
- **Default-mode preserved.** These smokes import without setting any
  standalone env var, matching their pre-refactor behavior.
- **Failure classification copied from #427**, not extracted to a
  shared helper. ~15 LOC per script. Pulling out a shared helper
  prematurely couples 3 unrelated callers; if the pattern propagates
  further, that's the right time to share.
- **No changes to script invocations or CI gates.** The wrapper
  scripts (`run_extracted_pipeline_checks.sh`,
  `run_extracted_competitive_intelligence_checks.sh`) already invoke
  these by name; manifest-driven is a behavior change, not an API
  change.
- **No deletion of MODULES constants in source.** Each script's
  hardcoded list is replaced by a manifest walk. The lists go away
  entirely -- no dead-code residue.

## Deferred (looks missing but is on purpose)

- **Refactor `smoke_extracted_competitive_intelligence_standalone.py`.**
  Separate slice (see above).
- **Extract shared manifest-walk helper.** Wait for a 4th caller to
  emerge.
- **Catch untracked .py files.** Methodology question -- separate
  from the drift hazard this PR closes.

## Verification

- `python3 scripts/smoke_extracted_pipeline_imports.py` -> exits 0
  with all manifest targets reporting OK or 3rd-party-env warning.
- `python3 scripts/smoke_extracted_competitive_intelligence_imports.py`
  -> exits 0 with all manifest targets reporting OK or 3rd-party-env
  warning.
- New regression test
  `tests/test_extracted_default_mode_smokes.py` asserts both smokes
  pass in the current state and that the failure classifier
  correctly distinguishes decoupling failures from 3rd-party env
  failures.
- `bash scripts/check_ascii_python.sh` -> passed.

## Conflict check

- No file overlap with #425 (`_b2b_witnesses` shim) or #427
  (manifest-driven standalone smoke).
- No dependency: this PR's smokes run in default mode, which already
  passes for `competitive_intelligence` (since `atlas_brain` is on
  `sys.path` and the relative import resolves to atlas_brain's
  brand_registry, not the missing extracted shim). The standalone
  smoke from #427 is what catches the missing-shim failure.

## Diff size

- 2 script refactors (~30 LOC each = ~60 LOC source)
- 1 regression test (~80 LOC)
- 1 plan doc (~110 LOC)

Per-script source diff stays under 30 LOC each because the manifest
walk replaces the entire MODULES constant.

## After this lands

All 3 smoke scripts that we control become drift-resistant in spirit:
2 are manifest-driven (this PR + #427), 1 (compintel-standalone) is
queued for separate restructure. The class of failures that bit #422
and #425 -- new mirror added but not added to a hardcoded smoke list
-- can no longer slip through the default-mode gate.
