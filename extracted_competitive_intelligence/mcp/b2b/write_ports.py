"""Host integration ports for competitive-intelligence write tools."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol


class WriteIntelligencePortNotConfigured(RuntimeError):
    """Raised when a host-only write-intelligence dependency is unavailable."""


class ChallengerBriefBuilder(Protocol):
    async def __call__(
        self,
        pool: Any,
        *,
        incumbent: str,
        challenger: str,
        persist: bool,
        max_target_accounts: int,
    ) -> dict[str, Any]:
        """Build a challenger brief for a vendor pair."""


class AccountsInMotionBuilder(Protocol):
    async def __call__(
        self,
        pool: Any,
        *,
        vendor_name: str,
        persist: bool,
        min_urgency: float,
        max_accounts: int,
    ) -> dict[str, Any]:
        """Build an accounts-in-motion report for a vendor."""


BlogMatcher = Callable[..., Awaitable[list[dict[str, Any]]]]
ReasoningViewLoader = Callable[..., Awaitable[Any]]


_challenger_brief_builder: ChallengerBriefBuilder | None = None
_accounts_in_motion_builder: AccountsInMotionBuilder | None = None
_blog_matcher: BlogMatcher | None = None
_reasoning_view_loader: ReasoningViewLoader | None = None


def configure_challenger_brief_builder(builder: ChallengerBriefBuilder) -> None:
    """Register the host adapter used by build_challenger_brief."""
    global _challenger_brief_builder
    _challenger_brief_builder = builder


def configure_accounts_in_motion_builder(builder: AccountsInMotionBuilder) -> None:
    """Register the host adapter used by build_accounts_in_motion."""
    global _accounts_in_motion_builder
    _accounts_in_motion_builder = builder


def configure_blog_matcher(matcher: BlogMatcher | None) -> None:
    """Register optional blog-context enrichment for campaign drafts."""
    global _blog_matcher
    _blog_matcher = matcher


def configure_reasoning_view_loader(loader: ReasoningViewLoader | None) -> None:
    """Register optional reasoning metadata enrichment for campaign drafts."""
    global _reasoning_view_loader
    _reasoning_view_loader = loader


def get_challenger_brief_builder() -> ChallengerBriefBuilder:
    if _challenger_brief_builder is None:
        raise WriteIntelligencePortNotConfigured(
            "No challenger brief builder has been configured"
        )
    return _challenger_brief_builder


def get_accounts_in_motion_builder() -> AccountsInMotionBuilder:
    if _accounts_in_motion_builder is None:
        raise WriteIntelligencePortNotConfigured(
            "No accounts-in-motion builder has been configured"
        )
    return _accounts_in_motion_builder


def get_blog_matcher() -> BlogMatcher | None:
    return _blog_matcher


def get_reasoning_view_loader() -> ReasoningViewLoader | None:
    return _reasoning_view_loader
