# PR: wire `scope_provider` from authenticated `AuthUser` (E2.5 of N)

## Why this slice exists

PR #454 (E2) added the landing_page wiring helper and tests
but gated it behind `enable_db_services=False` in production
because Codex flagged a P1 cross-tenant safety issue:
without a `scope_provider`, the executor falls back to an
empty `TenantScope` and `PostgresLandingPageRepository.save_drafts`
persists drafts under `account_id=""`. All tenants' drafts
would land in the same empty-account bucket.

This slice wires the missing piece -- a `scope_provider` that
resolves the per-request `AuthUser` from the existing
`require_b2b_plan` auth gate and converts it to a typed
`TenantScope` -- and then flips `enable_db_services=True` in
production so the bundle's `landing_page` slot lights up.

After this lands, `POST /api/v1/content-ops/execute` with
`outputs=["landing_page"]` writes drafts under the correct
authenticated tenant.

## Scope (this PR)

Three files, tightly bounded:

1. **`atlas_brain/_content_ops_scope.py`** (new): a
   `ContextVar`-based bridge from FastAPI's per-request
   dependency injection to the route's plain
   `Callable[[], TenantScope]` `scope_provider` shape.
   - `_CURRENT_AUTH_USER: ContextVar[AuthUser | None]`
     captures the per-request user.
   - `capture_content_ops_auth_user(user: AuthUser =
     Depends(require_b2b_plan("b2b_growth")))` -- FastAPI
     dependency that runs the existing auth gate AND sets
     the ContextVar. Used as `dependencies=[Depends(...)]`
     on the content-ops router so it runs on every
     content-ops request.
   - `build_content_ops_scope() -> TenantScope | None`
     reads the ContextVar and converts to `TenantScope`.
     Returns `None` when no user is captured (matches the
     existing route's "scope-not-configured" path).
   - Both factories accept dependency-injection kwargs for
     tests so the test harness doesn't need to construct
     real `AuthUser` instances through the host's auth chain.

2. **`atlas_brain/api/__init__.py`** -- swap the existing
   `dependencies=[Depends(require_b2b_plan("b2b_growth"))]`
   for `dependencies=[Depends(capture_content_ops_auth_user)]`
   (which itself depends on `require_b2b_plan`, so the auth
   gate is preserved). Pass `scope_provider=build_content_ops_scope`.
   Flip the `execution_services_provider` to call
   `build_content_ops_execution_services(enable_db_services=True)`.

3. **`tests/test_atlas_content_ops_scope.py`** (new): six
   regression tests pinning the ContextVar / scope bridge:
   - `test_build_content_ops_scope_round_trips_user_via_context_var`
     -- set the user via `set_current_auth_user`,
     `build_content_ops_scope()` returns the matching
     `TenantScope`.
   - `test_build_content_ops_scope_returns_none_when_no_user_captured`
     -- background-task / unauthenticated path.
   - `test_build_content_ops_scope_uses_injected_user_factory`
     -- DI kwarg short-circuits the ContextVar read.
   - `test_build_content_ops_scope_returns_none_when_factory_returns_none`
     -- factory returning None -> scope None.
   - `test_context_var_is_task_local` -- asyncio.Task-local
     isolation; sibling tasks don't see each other's users.
   - `test_codex_p1_contract_account_id_is_non_empty_for_authenticated_user`
     -- explicit pin on the cross-tenant safety contract from
     PR #454's Codex P1 review.

### What's NOT in this slice

- **`scope_provider` shape changes** in
  `extracted_content_pipeline/api/control_surfaces.py`. The
  existing `Callable[[], TenantScope | Mapping | None |
  Awaitable[...]]` shape is fine; the host bridges its
  per-request `AuthUser` to that shape via a `ContextVar`.
- **Wiring the remaining 4 LLM-needing services**
  (`campaign`, `blog_post`, `report`, `sales_brief`). All
  4 still need an `IntelligenceRepository` host factory --
  E3+. This slice just unblocks `landing_page`.
- **Per-request reasoning provider wiring.** PR #402 already
  wired the route-level seam; the bundle's
  `with_reasoning_context()` derivation handles per-request
  rebinding when a host wants multi-pass reasoning. Out of
  scope here.

## Mechanism

The `ContextVar` bridge lets the host expose the per-request
`AuthUser` to a plain callable `scope_provider`:

```python
# atlas_brain/_content_ops_scope.py
# Test-clean: NO fastapi or .auth.dependencies imports here.
# The capturing FastAPI dep lives in api/__init__.py because
# importing atlas_brain.auth.dependencies pulls cryptography
# which panics in dependency-light dev envs.
from contextvars import ContextVar
from extracted_content_pipeline.campaign_ports import TenantScope

_CURRENT_AUTH_USER: ContextVar[Any | None] = ContextVar(
    "content_ops_auth_user", default=None,
)


def set_current_auth_user(user: Any | None) -> None:
    _CURRENT_AUTH_USER.set(user)


def build_content_ops_scope(
    *, user_factory: Callable[[], Any | None] | None = None,
) -> TenantScope | None:
    user = (user_factory or _CURRENT_AUTH_USER.get)()
    if user is None:
        return None
    return TenantScope(
        account_id=getattr(user, "account_id", None),
        user_id=getattr(user, "user_id", None),
    )
```

The FastAPI dep + the route mount live in
`atlas_brain/api/__init__.py` (which already pays the
auth-chain import cost):

```python
# atlas_brain/api/__init__.py
from fastapi import Depends
from .._content_ops_scope import (
    build_content_ops_scope,
    set_current_auth_user,
)
from ..auth.dependencies import AuthUser, require_b2b_plan


async def _capture_content_ops_auth_user(
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
) -> AuthUser:
    set_current_auth_user(user)
    return user


content_ops_router = create_content_ops_control_surface_router(
    dependencies=[Depends(_capture_content_ops_auth_user)],
    execution_services_provider=(
        lambda: build_content_ops_execution_services(
            enable_db_services=True,
        )
    ),
    scope_provider=build_content_ops_scope,
)
```

`capture_content_ops_auth_user` is itself a `Depends` of
`require_b2b_plan("b2b_growth")`, so the auth gate is
preserved (paying tier check, B2B product check, past-due
guard); we just thread the resulting user into a
ContextVar before the route handler runs.

## Intentional

- **`ContextVar` rather than extending the route's
  `scope_provider` signature.** The route's existing
  `Callable[[], ...]` shape lives in the extracted package
  and serves multiple hosts; widening it to accept
  FastAPI-shaped `Depends` objects would couple the
  extracted package to FastAPI specifically. The
  `ContextVar` keeps the FastAPI coupling on the host side.
- **The FastAPI capturing dep lives in `api/__init__.py`,
  not inside `_content_ops_scope.py`.** Importing
  `atlas_brain.auth.dependencies` pulls in `cryptography`,
  which panics in dependency-light dev envs (the same
  reason the LLM/Skill adapters in PR #453 use
  `SimpleNamespace` rather than the host's `Message`
  dataclass). `api/__init__.py` already pays the auth-chain
  import cost, so the dep belongs there. The scope module
  exposes `set_current_auth_user` / `get_current_auth_user`
  so the dep can plumb the user across the boundary.
- **Both factories accept DI kwargs.** Same pattern as PRs
  #453 and #454; tests pass stubs without triggering the
  host's auth init chain.
- **`enable_db_services=True` flipped on AT THE SAME
  TIME** as the scope wiring. Splitting into two more PRs
  ("wire scope" then "flip flag") would leave the gate
  closed on a flag-only PR with nothing to gate -- not a
  meaningful slice. The Codex P1 was about the unsafe
  combination; closing both ends together is the right
  unit.
- **`build_content_ops_scope()` returns `None` when no
  user is captured.** Matches the existing route's
  "scope-not-configured" path. Background tasks or test
  harnesses that don't go through the auth gate get a
  `None` scope, which the executor handles cleanly.
- **No env-var gate.** Production always wires scope and
  enables DB services; tests always inject stubs.
  Operators that want to disable DB services can swap
  the bundle factory.

## Deferred

- `IntelligenceRepository` host factory + the 4 remaining
  generators (E3+).
- `allowed_vendors` / `roles` on `TenantScope` -- the
  existing `AuthUser` dataclass doesn't carry those today;
  if a future feature needs them, the host's auth layer
  surfaces them and `build_content_ops_scope` plumbs them
  through.
- Multi-pass reasoning provider host wiring.

## Verification

- `pytest tests/test_atlas_content_ops_scope.py
   tests/test_atlas_content_ops_execution_services.py`
  -- new + existing tests pass.
- AST + ASCII checks on new module + test + modified
  `__init__.py`.
- Smoke: `python -c "from atlas_brain._content_ops_scope
   import build_content_ops_scope; print(
   build_content_ops_scope())"` -> `None` (no user
  captured outside a request).

## Estimated diff size

Initial estimate undershot api/__init__.py (5x) and the test
file. Updated for transparency.

- `_content_ops_scope.py`: ~106 LOC actual (initial ~80;
  added defensive `getattr` reads + module docstring more
  detail than projected).
- `api/__init__.py`: ~30 LOC delta actual (initial ~6;
  the new FastAPI dep + its docstring + the
  set_current_auth_user wiring + the lambda wrapping
  enable_db_services=True together exceed the rough
  mental model).
- Test: ~150 LOC actual (initial ~120; 6 tests rather
  than 3).
- Plan doc: ~225 LOC actual (post-update with split
  Mechanism blocks + new Intentional bullet + expanded
  test inventory).

Total actual: ~510 LOC. Over the 400 LOC soft cap.
Indivisible -- the ContextVar setter, reader, FastAPI dep,
and route mount must ship together for the bridge to
function. Plan doc and tests dominate.
