"""SaaS authentication and authorization for Atlas consumer intelligence."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AuthUser": ".dependencies",
    "require_auth": ".dependencies",
    "optional_auth": ".dependencies",
    "require_plan": ".dependencies",
    "create_access_token": ".jwt",
    "create_refresh_token": ".jwt",
    "decode_token": ".jwt",
    "hash_password": ".passwords",
    "verify_password": ".passwords",
}

__all__ = [
    "AuthUser",
    "require_auth",
    "optional_auth",
    "require_plan",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "verify_password",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve package-level auth exports.

    Config validation imports ``atlas_brain.auth.encryption`` while
    ``settings`` is still being constructed. Eagerly importing auth
    dependencies here re-enters ``atlas_brain.config`` before that module has
    finished initialization, so keep settings-dependent modules behind this
    lazy boundary.
    """

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
