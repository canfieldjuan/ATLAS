"""Per-request scope bridge for Content Ops.

The route's `scope_provider` is a plain
``Callable[[], TenantScope | Mapping | None | Awaitable[...]]``
declared by the extracted package's
`create_content_ops_control_surface_router(...)`. It can't
receive FastAPI dependency-injected objects (the route
handler invokes it as `provider()` with no arguments). To
bridge the per-request `AuthUser` produced by the host's
auth chain into that plain-callable shape, we use an
asyncio-aware `ContextVar`:

1. The route's `dependencies=[Depends(capture_content_ops_auth_user)]`
   (defined in `atlas_brain/api/__init__.py` -- the FastAPI
   dep wiring lives there because importing it in this
   module would pull in the host's heavy auth chain via
   `atlas_brain.auth.dependencies` -> `cryptography` etc.)
   runs the existing `require_b2b_plan("b2b_growth")` gate
   and calls `set_current_auth_user(user)` on this module.

2. When the route handler invokes
   `scope_provider() -> build_content_ops_scope()`, the
   factory reads the `ContextVar` and converts the user to
   a typed `TenantScope`.

`ContextVar` is asyncio-aware: each request runs in its own
task and gets its own ContextVar context, so concurrent
requests don't see each other's users.

See `plans/PR-Content-Ops-Scope-Wire-1.md` for the slice
contract. Closes the Codex P1 deferred from PR #454 (E2):
landing_page drafts now persist under the authenticated
tenant's `account_id` instead of an empty string.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

from extracted_content_pipeline.campaign_ports import TenantScope


# Module-level per-request ContextVar. asyncio.Task-local: each
# inbound request gets its own copy. Default `None` so any code
# path that runs outside a request (background tasks, test harness
# without a fake user) sees the same "scope not configured" signal
# the existing route path already handles.
_CURRENT_AUTH_USER: ContextVar[Any | None] = ContextVar(
    "content_ops_auth_user", default=None,
)


def set_current_auth_user(user: Any | None) -> None:
    """Set the per-request `AuthUser` captured by the host's
    FastAPI dep. Called from `atlas_brain/api/__init__.py`'s
    `capture_content_ops_auth_user` dependency.

    Lives here (rather than inside the FastAPI dep itself) so
    tests can populate the ContextVar without going through
    the auth chain.
    """

    _CURRENT_AUTH_USER.set(user)


def get_current_auth_user() -> Any | None:
    """Test seam + read accessor for the captured user."""

    return _CURRENT_AUTH_USER.get()


def build_content_ops_scope(
    *,
    user_factory: Callable[[], Any | None] | None = None,
) -> TenantScope | None:
    """Resolve the per-request `AuthUser` into a `TenantScope`.

    Production: reads the `ContextVar` populated by
    `capture_content_ops_auth_user` upstream in the request's
    dep chain.

    Tests: pass `user_factory=` to inject a stub. When the
    factory returns `None`, the scope is `None` -- matches the
    existing route's "scope not configured" path so background
    tasks and test runs without a fake user behave predictably.
    """

    if user_factory is not None:
        user = user_factory()
    else:
        user = get_current_auth_user()

    if user is None:
        return None
    return TenantScope(
        account_id=getattr(user, "account_id", None),
        user_id=getattr(user, "user_id", None),
    )


__all__ = [
    "build_content_ops_scope",
    "get_current_auth_user",
    "set_current_auth_user",
]
