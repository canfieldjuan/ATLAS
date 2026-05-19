# PR-Live-Provider-Circular-Import

## Why this slice exists

The live provider smoke for AI Content Ops works only when
`EXTRACTED_LLM_INFRA_STANDALONE=1` is set. In the normal Atlas bridge mode,
`PipelineLLMClient` imports the extracted LLM infrastructure registry, which
imports `atlas_brain.services.registry`. That path initializes Atlas config.
During config validation, `atlas_brain.config` late-imports
`atlas_brain.auth.encryption`, but `atlas_brain.auth.__init__` eagerly imports
auth dependencies that import `settings` from the still-initializing config
module. The result is a circular import before a live provider can resolve.

This is not a smoke-script problem. The source issue is auth package
initialization doing settings-dependent work before config has finished
constructing `settings`.

## Scope (this PR)

1. Make package-level `atlas_brain.auth` exports lazy so importing
   `atlas_brain.auth.encryption` during config validation does not import
   `atlas_brain.auth.dependencies` or `atlas_brain.auth.jwt`.
2. Preserve the existing package-level auth API for callers that import
   `AuthUser`, `require_auth`, token helpers, or password helpers from
   `atlas_brain.auth`.
3. Add a subprocess regression test for the real `PipelineLLMClient` default
   resolver in Atlas bridge mode with `EXTRACTED_LLM_INFRA_STANDALONE` unset.
4. Register this slice in the coordination ledger.

### Files touched

- `atlas_brain/auth/__init__.py`
- `tests/test_live_provider_circular_import.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Live-Provider-Circular-Import.md`

## Mechanism

`atlas_brain.auth.__init__` keeps the same `__all__` but resolves exported
symbols through `__getattr__`. Imports that need `settings` stay behind the
lazy boundary until a caller asks for them. Password helpers remain safe to
load lazily as well for one consistent package pattern.

The regression test runs a fresh Python subprocess, unsets
`EXTRACTED_LLM_INFRA_STANDALONE`, sets a fake OpenRouter key, creates the
product `PipelineLLMClient`, and calls its default resolver. The resolver
constructs the provider adapter but does not call the network.

## Intentional

- No changes to `extracted_llm_infrastructure.services.registry`: bridge mode
  is still allowed to re-export the Atlas registry. The failing import was
  caused by auth package initialization during config validation, not by the
  provider resolver needing standalone mode.
- No environment-variable workaround in the smoke scripts: the default bridge
  mode should import cleanly.
- The test uses a subprocess instead of in-process monkeypatching so already
  imported modules cannot mask the circular import.

## Deferred

- Broader `atlas_brain.services.__init__` import slimming is deferred. It may
  still be valuable, but it is a wider API cleanup than needed to fix this
  provider-resolution failure.

## Verification

Local checks:

```bash
pytest tests/test_live_provider_circular_import.py
# 1 passed

python -m py_compile atlas_brain/auth/__init__.py tests/test_live_provider_circular_import.py
# passed

pytest tests/test_auth_dependencies.py tests/test_extracted_campaign_llm_client.py tests/test_live_provider_circular_import.py
# 21 passed

python - <<'PY'
# Default resolver smoke with EXTRACTED_LLM_INFRA_STANDALONE unset
PY
# OpenRouterLLM openrouter anthropic/claude-sonnet-4-5

bash scripts/local_pr_review.sh --allow-dirty
# passed
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `atlas_brain/auth/__init__.py` | 22 |
| `tests/test_live_provider_circular_import.py` | 51 |
| `docs/extraction/coordination/inflight.md` | 1 |
| `plans/PR-Live-Provider-Circular-Import.md` | 110 |
| **Total** | **184** |

Below the 400 LOC review budget.
