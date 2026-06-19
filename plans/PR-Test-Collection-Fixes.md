# PR-Test-Collection-Fixes

## Why this slice exists

Two test modules cannot be collected in a base `requirements.txt` environment,
which aborts any whole-suite run (e.g. a repo-wide unit sweep) at collection
time rather than reporting a clean skip:

- `tests/test_cloud_latency.py` does a module-level `from openai import OpenAI`,
  but `openai` is not a brain runtime dependency -- it is an external
  Fireworks/Together latency benchmark that also needs live API keys.
- `tests/test_graphiti_wrapper_health.py` `exec_module`s the standalone
  `graphiti-wrapper/main.py` at import time, whose deps (`graphiti_core`,
  `neo4j`) are not brain runtime deps.

Both are legitimately external/cross-service tests; they should be marked `e2e`
and skip cleanly when their optional deps are absent, instead of erroring
collection.

## Scope (this PR)

Ownership lane: ci/coverage
Slice phase: Production hardening

1. `tests/test_cloud_latency.py`: add module-level `pytest.importorskip("openai")`
   and `pytestmark = pytest.mark.e2e` so it skips cleanly without the SDK/keys.
2. `tests/test_graphiti_wrapper_health.py`: add `pytestmark = pytest.mark.e2e`
   and guard the cross-service `exec_module` with a module-level
   `pytest.skip(allow_module_level=True)` narrowed to `ImportError`, so an
   absent service dep skips but a real syntax/runtime regression still fails.

### Files touched

- `tests/test_cloud_latency.py`
- `tests/test_graphiti_wrapper_health.py`
- `plans/PR-Test-Collection-Fixes.md`

### Review Contract

Acceptance criteria:

- [ ] Both modules collect as clean skips under `not integration and not e2e`
      when their optional deps are absent.
- [ ] The graphiti guard only skips on `ImportError`, not arbitrary exceptions.
- [ ] No production code changes; test classification only.

Affected surfaces: tests only.

Risk areas: over-broad skip masking a real failure -- mitigated by narrowing
the graphiti guard to `ImportError`.

Reviewer rules triggered: R1.

## Mechanism

`pytest.importorskip("openai")` raises `Skipped` at module import when `openai`
is not installed, so collection records a skip instead of an `ImportError`. The
`e2e` marker keeps both modules out of the unit lane (`not integration and not
e2e`). For graphiti, `exec_module` is wrapped so that an `ImportError` (the
wrapper's `graphiti_core`/`neo4j` deps missing in a brain-only env) becomes a
module-level skip, while any other exception (a real regression in
`graphiti-wrapper/main.py`) propagates and fails the e2e test.

## Intentional

- Test classification only; the test bodies and assertions are unchanged.
- The graphiti guard is deliberately `ImportError`-only (not `except
  Exception`) so startup-breaking regressions in the wrapper are not silently
  skipped in the environment where the e2e test should run.

## Deferred

- A repo-wide unit backstop (separate slice) that benefits from these clean
  skips at whole-suite collection.

## Parked hardening

None.

## Verification

- `python -m pytest tests/test_cloud_latency.py -m "not integration and not e2e"`
  -- collects as a clean skip (no `openai`), no collection error.
- Both test modules pass `py_compile`.

## Estimated diff size

| File | LOC |
|---|---:|
| `tests/test_cloud_latency.py` | ~7 |
| `tests/test_graphiti_wrapper_health.py` | ~12 |
| `plans/PR-Test-Collection-Fixes.md` | ~80 |
| **Total** | **~99** |
