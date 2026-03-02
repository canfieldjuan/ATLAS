"""SaaS authentication and authorization for Atlas consumer intelligence."""

from .dependencies import AuthUser, require_auth, optional_auth, require_plan
from .jwt import create_access_token, create_refresh_token, decode_token
from .passwords import hash_password, verify_password

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
