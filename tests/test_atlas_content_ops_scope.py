"""Pin the ContextVar -> TenantScope bridge for Content Ops.

`atlas_brain/_content_ops_scope.py` exposes:
- `set_current_auth_user(user)` -- the host's FastAPI dep
  (defined in `atlas_brain/api/__init__.py`) calls this from
  inside the auth gate so the per-request `AuthUser` lands in
  a `ContextVar`.
- `get_current_auth_user()` -- read accessor for tests.
- `build_content_ops_scope(user_factory=None)` -- the route's
  `scope_provider`. Reads the ContextVar (or an injected
  factory in tests) and returns a typed `TenantScope`.

Five regression tests:

1. Round-trip: set a fake user, build_content_ops_scope()
   returns a TenantScope with matching account_id / user_id.
2. Returns None when no user is captured (background-task
   path; matches the route's "scope not configured" branch).
3. user_factory DI kwarg short-circuits the ContextVar read
   so tests don't need to populate it.
4. ContextVar is asyncio-task-local: setting in one task
   doesn't leak into a sibling task.
5. Closes the Codex P1 contract from PR #454: the scope's
   account_id is non-empty when an authenticated user is
   present, so PostgresLandingPageRepository.save_drafts no
   longer falls back to account_id="".

The ContextVar lives at module scope, so tests that mutate
it must reset (a `setUp`/`tearDown` pattern via pytest
fixtures keeps tests independent).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from atlas_brain._content_ops_scope import (
    build_content_ops_scope,
    get_current_auth_user,
    set_current_auth_user,
)
from extracted_content_pipeline.campaign_ports import TenantScope


@pytest.fixture(autouse=True)
def _reset_context_var() -> Any:
    """Reset the ContextVar between tests so leaks don't
    cascade. autouse so every test gets a clean slate."""

    set_current_auth_user(None)
    yield
    set_current_auth_user(None)


def _fake_user(*, account_id: str = "acct-1", user_id: str = "user-1") -> Any:
    return SimpleNamespace(account_id=account_id, user_id=user_id)


def test_build_content_ops_scope_round_trips_user_via_context_var() -> None:
    """Set the ContextVar; build_content_ops_scope() returns a
    TenantScope carrying the same identifiers."""

    user = _fake_user(account_id="acct-42", user_id="user-99")
    set_current_auth_user(user)

    scope = build_content_ops_scope()

    assert isinstance(scope, TenantScope)
    assert scope.account_id == "acct-42"
    assert scope.user_id == "user-99"


def test_build_content_ops_scope_returns_none_when_no_user_captured() -> None:
    """Background tasks / test runs without a fake user see the
    same `None` scope the existing route's
    "scope-not-configured" path handles."""

    # Fixture already reset; no user captured.
    assert get_current_auth_user() is None
    assert build_content_ops_scope() is None


def test_build_content_ops_scope_uses_injected_user_factory() -> None:
    """Tests can pass `user_factory=` to inject a stub without
    populating the ContextVar."""

    fake = _fake_user(account_id="dep-acct", user_id="dep-user")
    scope = build_content_ops_scope(user_factory=lambda: fake)

    assert isinstance(scope, TenantScope)
    assert scope.account_id == "dep-acct"
    assert scope.user_id == "dep-user"


def test_build_content_ops_scope_returns_none_when_factory_returns_none() -> None:
    """Even with the kwarg present, returning None -> None."""

    assert build_content_ops_scope(user_factory=lambda: None) is None


@pytest.mark.asyncio
async def test_context_var_is_task_local() -> None:
    """asyncio.Task-local: setting the user in one task doesn't
    leak into a sibling task. Concurrent requests must not see
    each other's tenants."""

    seen_in_sibling: dict[str, Any] = {}

    async def _set_in_task() -> None:
        set_current_auth_user(_fake_user(account_id="task-A"))
        # Yield control so the sibling can run before we
        # re-check.
        await asyncio.sleep(0)
        seen_in_sibling["after_yield"] = get_current_auth_user()

    async def _sibling_task() -> None:
        # Sibling hasn't set anything; should see None even if
        # the other task set its own.
        await asyncio.sleep(0)
        seen_in_sibling["sibling"] = get_current_auth_user()

    await asyncio.gather(_set_in_task(), _sibling_task())

    # Sibling never set -- must be None despite the other task
    # writing to its own copy.
    assert seen_in_sibling["sibling"] is None
    # The setter task still sees its own user after yielding.
    assert getattr(seen_in_sibling["after_yield"], "account_id", None) == "task-A"


def test_codex_p1_contract_account_id_is_non_empty_for_authenticated_user() -> None:
    """Codex P1 from PR #454 review: without scope wiring,
    PostgresLandingPageRepository.save_drafts persists drafts
    under account_id="" -- cross-tenant leakage. With this
    slice, an authenticated AuthUser flows through the bridge
    and the resulting TenantScope.account_id is the user's
    real account id. Pin the contract."""

    user = _fake_user(account_id="real-tenant-id", user_id="user-x")
    set_current_auth_user(user)

    scope = build_content_ops_scope()

    assert scope is not None
    assert scope.account_id == "real-tenant-id"
    assert scope.account_id != ""
