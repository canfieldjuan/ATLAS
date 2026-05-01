"""Fail-closed auth helpers for standalone competitive intelligence."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    account_id: str
    email: str | None = None
    plan: str | None = None


def require_auth() -> AuthUser:
    raise HTTPException(
        status_code=501,
        detail="Standalone auth adapter is not configured",
    )

