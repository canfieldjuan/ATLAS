# PR: wire reasoning context provider into host route mount (Host-Wire-1 of N)

## Why this slice exists

PR #402 shipped the route-level `reasoning_context_provider`
seam in `extracted_content_pipeline/api/control_surfaces.py`
plus the bundle's per-request `with_reasoning_context()`
derivation, but the host's content-ops route mount in
`atlas_brain/api/__init__.py` doesn't pass that kwarg yet.
Result: every `/api/v1/content-ops/execute` request runs
through the bundle without enriched reasoning context.

The extracted package ships `FileCampaignReasoningContextProvider`
(`extracted_content_pipeline/campaign_reasoning_data.py`) as
a reference implementation: read pre-computed contexts from a
JSON file and serve them per-call. Operators that already
have offline reasoning pipelines can populate that file and
get reasoning enrichment immediately for all 5 LLM-using
outputs (`landing_page`, `email_campaign`, `report`,
`sales_brief`, `blog_post`).

This slice wires the missing seam. It does NOT introduce a
new live reasoning chain (separate slice if desired) -- it
unblocks the file-backed path and establishes the wiring
pattern any future provider (database-backed, host
intervention-pipeline-backed) plugs into without touching
the route mount again.

## Scope (this PR)

Three files plus the plan doc; tightly bounded:

1. **`atlas_brain/_content_ops_reasoning.py`** (new): the
   factory.
   - `build_content_ops_reasoning_context_provider()` reads
     `ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH` from the
     environment.
   - When set + the file exists, returns
     `load_campaign_reasoning_context_provider(path)` (the
     file-backed adapter from
     `extracted_content_pipeline.campaign_reasoning_data`).
   - When unset OR the file is missing, returns `None` --
     same behavior as today, no regression for hosts that
     haven't opted in.
   - Loader exceptions are logged at WARN and resolve to
     `None` so a malformed file doesn't kill the route
     mount during host startup.
   - DI kwargs (`path_factory`, `loader_factory`) for
     test isolation -- same pattern as PRs #453 / #455 /
     #460 / #461.
   - Lives at `atlas_brain/` root with underscore prefix
     (matches the established host-internal pattern).

2. **`atlas_brain/api/__init__.py`** (modified, ~5 LOC
   delta): pass
   `reasoning_context_provider=build_content_ops_reasoning_context_provider`
   to `create_content_ops_control_surface_router(...)`.
   The route layer's existing
   `with_reasoning_context` derivation handles per-request
   rebinding from there.

3. **`tests/test_atlas_content_ops_reasoning.py`** (new):
   ~6 tests pinning env-var-unset path returns `None`,
   missing-file path returns `None`, malformed-file path
   logs + returns `None`, valid-file path returns the
   loaded provider, DI kwargs short-circuit env-var
   resolution, and provider-loaded round-trip via the
   exposed reader method.

4. **`plans/PR-Content-Ops-Reasoning-Provider-Host-Wire-1.md`**
   (this file).

### What's NOT in this slice

- **A live reasoning chain.** The file-backed provider is the
  reference adapter; richer providers (database-backed,
  intervention-pipeline-backed) are separate slices.
- **Per-tenant reasoning context isolation.** The
  file-backed provider is host-wide -- all authenticated
  tenants see the same contexts. A tenant-scoped provider
  is a follow-up; the file-backed one is a meaningful
  unblock for single-tenant deployments / staging.
- **Pydantic settings nesting.** Reading the env var
  directly via `os.environ.get` keeps this slice from
  touching the 5000+ line `config.py`. A future settings
  refactor can fold the env var into a
  `ContentOpsSettings` block.
- **Operator UI for managing context contents.** Operators
  edit the file directly today; admin tooling is a follow-up.

## Mechanism

```python
# atlas_brain/_content_ops_reasoning.py
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH"


def build_content_ops_reasoning_context_provider(
    *,
    path_factory: Callable[[], str | None] | None = None,
    loader_factory: Callable[[str], Any] | None = None,
) -> Any | None:
    """Return the configured reasoning context provider, or
    None when the host hasn't opted in.

    Hosts opt in by setting ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH
    to a JSON file readable by FileCampaignReasoningContextProvider.
    Failures (missing file, parse errors) resolve to None with
    a warning logged -- a bad file must not crash the route mount.
    """

    path = (path_factory or _read_env_path)()
    if not path:
        return None
    if not Path(path).is_file():
        logger.warning(
            "Content Ops reasoning context path %s does not exist; "
            "provider stays unwired.",
            path,
        )
        return None

    loader = loader_factory or _default_loader
    try:
        return loader(path)
    except Exception as exc:
        logger.warning(
            "Failed to load Content Ops reasoning context from %s: %s",
            path, exc,
        )
        return None


def _read_env_path() -> str | None:
    return os.environ.get(_ENV_VAR) or None


def _default_loader(path: str) -> Any:
    from extracted_content_pipeline.campaign_reasoning_data import (
        load_campaign_reasoning_context_provider,
    )
    return load_campaign_reasoning_context_provider(path)
```

`atlas_brain/api/__init__.py` adds one kwarg:

```python
content_ops_router = create_content_ops_control_surface_router(
    dependencies=[Depends(_capture_content_ops_auth_user)],
    execution_services_provider=lambda: (
        build_content_ops_execution_services(enable_db_services=True)
    ),
    scope_provider=build_content_ops_scope,
    reasoning_context_provider=build_content_ops_reasoning_context_provider,
)
```

## Intentional

- **Default unwired (`None`).** Matches today's behavior;
  hosts that haven't set the env var see no change.
- **Lazy import of the file-backed provider.** Keeps the
  factory module light enough to import in dependency-light
  dev envs; the heavy `extracted_content_pipeline.campaign_reasoning_data`
  module loads only when the env var is set.
- **WARN-and-fall-back on errors rather than raise.** The
  reasoning provider is enrichment; a malformed JSON file
  must not crash the route mount or block the entire
  Content Ops surface for all tenants.
- **Env var rather than Pydantic settings.** The host's
  `config.py` is 5000+ LOC and folding into it would
  balloon this slice. The env-var pattern is symmetric
  with PR #453's `_content_ops_infrastructure.py`. A
  future settings refactor can fold the var in.
- **DI kwargs (`path_factory`, `loader_factory`).** Same
  testability pattern as the rest of the Content Ops
  wiring slices.
- **Module at `atlas_brain/` root with underscore prefix.**
  Matches `_content_ops_infrastructure.py` (PR #453),
  `_content_ops_scope.py` (PR #455), etc. -- avoids the
  heavy `atlas_brain.services` import chain.

## Deferred

- Database-backed reasoning provider (per-tenant contexts).
- Intervention-pipeline-backed provider that surfaces
  `intelligence/autonomous_narrative_architect` outputs as
  per-opportunity reasoning contexts.
- Pydantic settings integration.
- Admin tooling for editing the context file.
- Per-output reasoning policies (e.g. blog_post requires
  reasoning, email_campaign optional).

## Verification

- `pytest tests/test_atlas_content_ops_reasoning.py`
  -- new tests pass.
- AST + ASCII gates clean.
- Smoke: `python -c "from atlas_brain._content_ops_reasoning
  import build_content_ops_reasoning_context_provider; print(
  build_content_ops_reasoning_context_provider())"` -> `None`
  (no env var set).

## Estimated diff size

- `_content_ops_reasoning.py`: ~95 LOC.
- `api/__init__.py` delta: ~5 LOC.
- Tests: ~180 LOC.
- Plan doc: ~165 LOC.

Total: ~445 LOC. Marginally over the 400 LOC soft cap.
Indivisible -- the factory needs the route-mount kwarg to
have an effect; tests need both. The ~80 LOC overage is
all plan doc and test scaffolding; production code is
~100 LOC total.
