# PR: Content Ops Usage Account Metadata

## Why this slice exists

The operator usage summary can now read Content Ops LLM usage rows, and the
hosted LLM factory slice wires hosted generation into the product tracing
client. The next prerequisite for tenant-facing spend cards is account scope in
the trace metadata. Without account metadata, any tenant-facing route or UI card
would either be impossible to filter correctly or would risk exposing global
spend.

This slice adds the account metadata at the execution boundary where TenantScope
is already known, instead of patching every individual generator call site.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a scoped Content Ops LLM trace metadata context to the product
   PipelineLLMClient.
2. Set that context around each Content Ops execution step from the resolved
   TenantScope.
3. Preserve explicit per-call metadata precedence for existing generator
   labels such as asset_type, request_id, and run_id, while keeping scoped
   account_id and user_id authoritative.
4. Add focused tests proving account_id and user_id reach successful and failed
   LLM traces without leaking across calls.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Usage-Account-Metadata.md` | Plan doc for tenant/account usage metadata. |
| `extracted_content_pipeline/campaign_llm_client.py` | Add scoped trace metadata and merge it into Content Ops trace rows. |
| `extracted_content_pipeline/content_ops_execution.py` | Set scoped trace metadata around each executed generation step. |
| `tests/test_extracted_campaign_llm_client.py` | Pin scoped metadata merge and reset behavior. |
| `tests/test_extracted_content_ops_execution.py` | Pin execution-scope account metadata around service calls. |

## Mechanism

PipelineLLMClient reads a ContextVar-backed metadata mapping at trace time. The
executor sets that context for each step using the resolved TenantScope:
account_id and user_id are included only when non-empty. The context is reset in
a finally block after the step finishes, so concurrent steps do not leave stale
scope behind for later calls.

Trace metadata merge order is base Content Ops metadata, scoped execution
metadata, then explicit generator metadata after stripping account_id and
user_id from the per-call mapping. That lets the execution boundary add trusted
account_id and user_id while preserving existing per-call labels such as
asset_type, request_id, run_id, skill_name, and attempt_no.

## Intentional

- This does not add a tenant-facing usage summary route yet. It only makes the
  underlying usage rows filterable by metadata ->> 'account_id'.
- This does not change the operator-only global usage route from the previous
  slice.
- This does not modify every generator. The executor and tracing client are the
  shared source of truth for execution-scope metadata.
- Per-call metadata cannot override scoped account_id or user_id. Those fields
  are the tenant attribution anchor for future tenant-facing usage filters.
- This does not add UI, budget gates, or cache controls.

## Deferred

- Future PR: add a tenant-scoped usage summary path that filters by metadata ->>
  'account_id' from the authenticated TenantScope.
- Future PR: add UI/control-surface cards against the correct operator or tenant
  usage route.
- Future PR: wire BudgetGate once account-scoped usage is queryable.
- Future PR: add exact-cache integration with explicit support-ticket privacy
  policy and account scoping.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_campaign_llm_client.py tests/test_extracted_content_ops_execution.py -q — 69 passed, 1 warning.
- python -m compileall -q extracted_content_pipeline/campaign_llm_client.py extracted_content_pipeline/content_ops_execution.py tests/test_extracted_campaign_llm_client.py tests/test_extracted_content_ops_execution.py — passed.
- git diff --check — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Pipeline LLM trace context | ~55 |
| Executor context wiring | ~30 |
| Tests | ~115 |
| **Total** | **~285** |

Under the 400 LOC soft cap.
