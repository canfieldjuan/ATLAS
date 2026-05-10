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

3. **`tests/test_atlas_content_ops_scope.py`** (new): pin the
   ContextVar / scope round-trip:
   - `capture_content_ops_auth_user` sets the user;
     `build_content_ops_scope()` reads it back as a
     `TenantScope` with `account_id` + `user_id`.
   - `build_content_ops_scope()` returns `None` when no
     user is captured (e.g. background task path).
   - `_CURRENT_AUTH_USER` is request-local: setting it in
     one async task doesn't leak to a sibling task.

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
from contextvars import ContextVar
from fastapi import Depends
from .auth.dependencies import AuthUser, require_b2b_plan
from extracted_content_pipeline.campaign_ports import TenantScope

_CURRENT_AUTH_USER: ContextVar[AuthUser | None] = ContextVar(
    "content_ops_auth_user", default=None,
)


async def capture_content_ops_auth_user(
    user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
) -> AuthUser:
    _CURRENT_AUTH_USER.set(user)
    return user


def build_content_ops_scope() -> TenantScope | None:
    user = _CURRENT_AUTH_USER.get()
    if user is None:
        return None
    return TenantScope(
        account_id=user.account_id,
        user_id=user.user_id,
    )
```

`ContextVar` is asyncio-aware: each request runs in its own
task and gets its own ContextVar context, so concurrent
requests don't see each other's users.

The route mount in `atlas_brain/api/__init__.py` plugs the
two together:

```python
content_ops_router = create_content_ops_control_surface_router(
    dependencies=[Depends(capture_content_ops_auth_user)],
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

- `_content_ops_scope.py`: ~80 LOC.
- `api/__init__.py`: ~6 LOC delta.
- Test: ~120 LOC.
- Plan doc: ~190 LOC.

Total: ~395 LOC. Right at the budget.
