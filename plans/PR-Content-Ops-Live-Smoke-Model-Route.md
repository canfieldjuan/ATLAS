# PR: Content Ops Live Smoke Model Route

## Why this slice exists

Issue #1299 correctly gates more Content Ops feature work behind a real live
smoke, but its example wording mentioned local Ollama/qwen. That sent the
builder toward the wrong model path. Content Ops generation is sold and tested
as a cloud-generated product surface, and the host wiring routes generation
through the pipeline LLM client with OpenRouter preferred.

This slice makes the live-smoke contract explicit so future sessions validate
the same route the product actually uses: configured cloud/OpenRouter/Claude,
with local Ollama fallback disabled for validation.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Workflow/process

1. Update the builder bootstrap recurring-lapse checklist with the Content Ops
   live-smoke model route.
2. Add a focused validation note under `docs/extraction/validation/` for the
   #1299 gate.
3. Do not change generation code, model adapters, live-smoke execution logic,
   #1300 PNG code, or #1268 output variations.

### Files touched

- `plans/PR-Content-Ops-Live-Smoke-Model-Route.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/extraction/validation/content_ops_live_smoke_model_route.md`

## Mechanism

The docs state the exact operational rule:

```bash
EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false \
python scripts/smoke_content_ops_live_generation.py ...
```

The run must use the configured OpenRouter route and fail closed when the cloud
route is unavailable. Local Ollama/qwen is not acceptable evidence for Content
Ops generated-content validation.

## Intentional

- This PR is process/docs only. It does not attempt the #1299 live smoke.
- The exact OpenRouter model is described as configuration-owned; the current
  observed default is `anthropic/claude-sonnet-4-5`, but the invariant is the
  cloud route, not a hardcoded model in code.
- No #1300 changes. The held PNG PR should be resumed only after the live gate
  proves real browser rendering and real cloud generation.

## Deferred

- The #1299 live-adapter smoke: real Postgres migrations through 334, real
  OpenRouter/Claude generation, real browser PNG render, tenant isolation,
  review/export UI, and validation doc from the actual run.

## Parked hardening

None.

## Verification

- `git diff --check` -- passed.
- `rg -n "qwen|Ollama|ollama|OpenRouter|EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA|claude-sonnet" docs/SESSION_BOOTSTRAP.md docs/extraction/validation/content_ops_live_smoke_model_route.md plans/PR-Content-Ops-Live-Smoke-Model-Route.md` -- verified route wording.
- `gh issue view 1299 --json body,comments -q '{bodyHasOpenRouter: (.body | contains("configured cloud/OpenRouter model route")), bodyHasNoOllamaFallback: (.body | contains("EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false")), bodyHasPng: (.body | contains("real PNG/Chromium")), commentHasOpenRouter: (.comments[] | select(.id=="IC_kwDOQ5Uhrs8AAAABE0G38Q") | .body | contains("configured cloud/OpenRouter/Claude"))}'` -- all fields true.

## Estimated diff size

| Area | LOC |
|---|---:|
| **Total** | **~140** |
