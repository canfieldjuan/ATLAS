# Route reflection LLM analysis through port adapter

## Why this slice exists

After PR #570 and PR #571, the reactive graph nodes use
`AtlasLLMClient` through the extracted reasoning-core port contract. Reflection
still calls the legacy `atlas_brain.reasoning.graph._llm_generate` helper. That
keeps a second LLM call path alive after the graph nodes have moved to the port
adapter seam.

## Scope (this PR)

Move reflection's LLM analysis onto `AtlasLLMClient` plus
`extracted_reasoning_core.graph_helpers.complete_with_json`, then remove the
now-obsolete `_llm_generate` helper from `atlas_brain.reasoning.graph`.

### Files touched

- `atlas_brain/reasoning/graph.py`
- `atlas_brain/reasoning/reflection.py`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/queue.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Reasoning-Core-Reflection-Port-Adapter-2026-05-17.md`
- `tests/test_anthropic_timeout.py`
- `tests/test_atlas_reasoning_reflection_tracing.py`
- `tests/test_reasoning_graph_routing.py`

## Mechanism

- Replace reflection's `_llm_generate` / `_parse_llm_json` import with
  `AtlasLLMClient` and `complete_with_json`.
- Pass `json_mode=True` for reflection analysis so the port adapter requests a
  JSON object from providers that support structured output. This is an
  intentional reliability upgrade over the old best-effort parse-only path.
- Keep reflection's existing configured workload and fallback behavior.
- Remove `_llm_generate` from `atlas_brain.reasoning.graph` once reflection no
  longer imports it.
- Update tests to fake the current `chat(...)` seam instead of the removed
  helper.
- Advance coordination docs from merged #571 to this PR-C8 slice.

## Intentional

- No change to graph node behavior.
- No change to reflection's notification/action policy.
- No change to provider routing or configured workload names.
- Intentional LLM call-shape change: reflection analysis now requests
  structured JSON output through the same metadata path as graph nodes.

## Deferred

- Further graph/state wrapper split work remains after reflection uses the same
  LLM port seam as the graph nodes.

## Verification

- Focused Python compile for edited production and test files.
- Focused pytest for reflection, routing, graph, and Anthropic timeout tests.
- Git whitespace check.
- Local PR review script.

## Estimated diff size

| Area | LOC |
|---|---:|
| Production routing cleanup | ~70 |
| Tests | ~70 |
| Coordination + plan | ~80 |
| **Total** | ~220 |
