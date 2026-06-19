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

They should skip cleanly when their optional deps are absent instead of
erroring collection. `test_cloud_latency` genuinely calls live external APIs,
so it is `e2e`; `test_graphiti_wrapper_health`'s tests are fully mocked, so
they stay in the unit lane and only the cross-service module load is gated on
the optional deps.

## Scope (this PR)

Ownership lane: ci/coverage
Slice phase: Production hardening

1. `tests/test_cloud_latency.py`: add module-level `pytest.importorskip("openai")`
   and `pytestmark = pytest.mark.e2e` so it skips cleanly without the SDK/keys
   (it is an external live-API benchmark).
2. `tests/test_graphiti_wrapper_health.py`: preflight the named optional service
   deps with `pytest.importorskip("neo4j")` / `pytest.importorskip("graphiti_core")`
   before `exec_module`, and run `exec_module` unguarded. No `e2e` marker: the
   health/readiness tests are fully mocked, so when the deps are present they
   run in the unit lane, and a real import regression in `main.py` or its local
   modules still fails (only the named absent deps skip).

### Files touched

- `tests/test_cloud_latency.py`
- `tests/test_graphiti_wrapper_health.py`
- `plans/PR-Test-Collection-Fixes.md`

### Review Contract

Acceptance criteria:

- [ ] Both modules collect as clean skips in a brain-only env (no `openai`,
      no `neo4j`/`graphiti_core`) instead of erroring collection.
- [ ] `test_graphiti_wrapper_health` skips only on the named absent optional
      deps; a real import regression in `main.py` or its local modules fails.
- [ ] `test_graphiti_wrapper_health` carries no `e2e` marker, so its mocked
      tests run in the unit lane when the deps are present.
- [ ] No production code changes; test classification only.

Affected surfaces: tests only.

Risk areas: a skip masking a real failure -- mitigated by preflighting only the
named optional deps (`neo4j`, `graphiti_core`) and running `exec_module`
unguarded so other import regressions propagate.

Reviewer rules triggered: R1.

## Mechanism

`pytest.importorskip(name)` raises `Skipped` at module import when `name` is not
installed, recording a clean skip instead of an `ImportError`. `test_cloud_latency`
preflights `openai` and is marked `e2e` (it makes live Fireworks/Together
calls). `test_graphiti_wrapper_health` preflights the named service deps
`neo4j` and `graphiti_core`, then `exec_module`s `graphiti-wrapper/main.py`
**unguarded**: in a brain-only env the preflight skips before the load is
attempted, while if those deps are present a genuine import regression in
`main.py` or one of its local modules (e.g. `embedder_factory`) propagates as a
failure rather than a silent skip. The module is not marked `e2e` because its
health/readiness tests patch `AsyncGraphDatabase`, the embedder preload, and
the readiness gate -- they need no live services and belong in the unit lane.

## Intentional

- Test classification only; the test bodies and assertions are unchanged.
- The graphiti skip is preflight-on-named-deps (not a broad `except`) so
  regressions in the wrapper's own modules still fail in the environment where
  the test can run -- per reviewer feedback (Codex + Copilot) on the prior
  combined PR.
- No `e2e` marker on the mocked graphiti tests, so they execute in the
  repo-wide unit lane when `neo4j`/`graphiti_core` are installed.

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
