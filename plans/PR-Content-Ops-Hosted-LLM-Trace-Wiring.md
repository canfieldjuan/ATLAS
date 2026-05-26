# PR: Content Ops Hosted LLM Trace Wiring

## Why this slice exists

PR-Content-Ops-LLM-Usage-Tracing made the product PipelineLLMClient emit
content_ops.llm.complete rows, and PR-Content-Ops-Usage-Summary added the
operator usage read path over those rows. The hosted Atlas Content Ops factory
still returns _HostLLMClient, which routes through Atlas LLM services but
drops per-call metadata and does not use the product tracing client.

That means the read path can exist while hosted /content-ops/execute calls do
not reliably write the Content Ops usage rows the read path summarizes. This
slice fixes that integration at the LLM factory source before UI cost cards,
tenant-scoped usage, budget gates, or cache controls build on top of it.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Make build_content_ops_llm_client() return the product
   PipelineLLMClient for both pipeline-routed and active-registry Atlas LLMs.
2. Preserve the existing factory contract: return None when no provider is
   routable, so execution slots stay disabled instead of failing later.
3. Keep the existing _HostLLMClient adapter available for direct adapter tests
   and any future non-tracing host-only use, but stop using it as the hosted
   production factory return.
4. Add a hosted-factory test proving Content Ops metadata reaches the
   content_ops.llm.complete trace instead of being dropped.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Hosted-LLM-Trace-Wiring.md` | Plan doc for closing the hosted LLM trace wiring gap. |
| `atlas_brain/_content_ops_infrastructure.py` | Return PipelineLLMClient from the hosted Content Ops LLM factory. |
| `tests/test_atlas_content_ops_infrastructure.py` | Update factory tests and pin metadata-preserving trace behavior. |

## Mechanism

build_content_ops_llm_client() keeps the current preflight behavior by asking
the host resolver whether an LLM is routable. When the resolver returns a
provider, the factory returns PipelineLLMClient configured with the same
resolver and the same OpenRouter-oriented routing kwargs already used today.
`PipelineLLMClient.complete()` then resolves the provider per call and emits the
existing content_ops.llm.complete trace with the metadata supplied by the
generation services.

For the active-registry fallback, the factory still checks get_active() first.
If a provider exists, it returns a PipelineLLMClient with a small resolver that
returns that provider. This preserves the previous fallback shape while moving
the call through the same tracing path.

## Intentional

- This does not add tenant-scoped usage yet. It makes hosted calls write the
  trace rows first; account metadata can be added as the next safe tenant-card
  prerequisite.
- This does not change model routing defaults. The factory keeps the current
  OpenRouter workload and auto_activate_ollama=False routing shape.
- _HostLLMClient remains in place because its direct adapter behavior is still
  tested, but it is no longer the production Content Ops factory path.
- This does not add UI, budget gates, or cache controls.

## Deferred

- Future PR: add account_id to Content Ops LLM trace metadata so tenant-facing
  usage cards can filter by metadata ->> 'account_id'.
- Future PR: add UI/control-surface cards against the usage summary route.
- Future PR: wire BudgetGate once hosted usage rows are known to be emitted.
- Future PR: add exact-cache integration with explicit support-ticket privacy
  policy and account scoping.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_atlas_content_ops_infrastructure.py tests/test_extracted_campaign_llm_client.py -q — 27 passed.
- python -m compileall -q atlas_brain/_content_ops_infrastructure.py tests/test_atlas_content_ops_infrastructure.py — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Hosted LLM factory | ~35 |
| Tests | ~70 |
| **Total** | **~185** |

Under the 400 LOC soft cap.
