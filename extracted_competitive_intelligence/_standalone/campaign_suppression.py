"""Suppression policy port for standalone competitive intelligence."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class SuppressionPolicyNotConfigured(RuntimeError):
    pass


@runtime_checkable
class SuppressionPolicy(Protocol):
    async def is_suppressed(self, pool: Any, *, email: str) -> dict[str, Any] | None:
        ...


_policy: SuppressionPolicy | None = None


def configure_suppression_policy(policy: SuppressionPolicy | None) -> None:
    global _policy
    _policy = policy


async def is_suppressed(pool: Any, *, email: str) -> dict[str, Any] | None:
    if _policy is None:
        raise SuppressionPolicyNotConfigured(
            "Standalone suppression policy adapter is not configured"
        )
    return await _policy.is_suppressed(pool, email=email)

