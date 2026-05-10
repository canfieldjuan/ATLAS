"""Fan an autonomous-task blueprint out to per-tenant subscribers.

After the autonomous blog-post task generates and stores a
finished blog draft, this helper looks up active
``BlogPostSubscription`` rows (PR #460) and writes a per-account
copy of the blueprint into ``blog_blueprints`` (PR #458) so the
Content Ops ``/api/v1/content-ops/execute`` route (PR #459) can
read them under each tenant's authenticated scope.

Per-account ``save_blueprints`` failures are logged but don't
abort the remaining fanout: the autonomous task's draft has
already been stored, so partial fanout is strictly better than
no fanout.

Lives at ``atlas_brain/`` root for the same reason as
``_blog_post_subscriptions.py`` and
``_content_ops_infrastructure.py`` -- importing
``atlas_brain.services`` triggers an eager torch / ollama load
that panics in dependency-light dev envs.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Sequence

from extracted_content_pipeline.blog_blueprint_postgres import (
    BlogBlueprint,
    PostgresBlogBlueprintRepository,
)
from extracted_content_pipeline.campaign_ports import TenantScope

from ._blog_post_subscriptions import (
    BlogPostSubscription,
    list_active_blog_post_subscriptions,
)


logger = logging.getLogger(__name__)


# Fixed target_mode for autonomous-task blueprints. The
# autonomous task generates one shape of B2B retention blog
# post regardless of topic_type; subscriptions filter by
# topic_type for variety. target_mode is the top-level
# grouping for future fan-in (e.g. an "executive brief"
# generator emitting target_mode="executive_brief").
DEFAULT_TARGET_MODE = "b2b_blog_post"


SubscriptionsFactory = Callable[[], Awaitable[Sequence[BlogPostSubscription]]]
RepoFactory = Callable[[Any], Any]


async def fanout_blueprint(
    pool: Any,
    blueprint: Any,
    *,
    target_mode: str = DEFAULT_TARGET_MODE,
    subscriptions_factory: SubscriptionsFactory | None = None,
    repo_factory: RepoFactory | None = None,
) -> int:
    """Save a per-account copy of ``blueprint`` for every
    matching subscription. Returns the count of fanout writes.

    ``blueprint`` is duck-typed: any object exposing
    ``topic_type`` / ``slug`` / ``suggested_title`` / ``sections``
    / ``charts`` / ``tags`` / ``data_context`` /
    ``quotable_phrases`` / ``cta`` is acceptable. The autonomous
    task passes a ``PostBlueprint`` dataclass.

    DI kwargs let tests stub the subscription reader and repo
    constructor without populating the host's full init chain
    (asyncpg / saas_accounts / blog_blueprints migration).
    """

    subs = await _read_subscriptions(pool, subscriptions_factory)
    if not subs:
        return 0

    matching = [
        sub
        for sub in subs
        if sub.matches(
            topic_type=getattr(blueprint, "topic_type", ""),
            target_mode=target_mode,
        )
    ]
    if not matching:
        return 0

    repo = (
        repo_factory(pool)
        if repo_factory is not None
        else PostgresBlogBlueprintRepository(pool=pool)
    )

    payload = _serialize_payload(blueprint)
    saved = 0
    for sub in matching:
        record = BlogBlueprint(
            target_mode=target_mode,
            topic_type=str(getattr(blueprint, "topic_type", "")),
            slug=str(getattr(blueprint, "slug", "")),
            suggested_title=str(getattr(blueprint, "suggested_title", "")),
            payload=payload,
        )
        try:
            await repo.save_blueprints(
                [record],
                scope=TenantScope(account_id=sub.account_id),
            )
            saved += 1
        except Exception as exc:
            logger.warning(
                "Blog blueprint fanout failed for account_id=%s slug=%s: %s",
                sub.account_id,
                getattr(blueprint, "slug", "?"),
                exc,
            )
    return saved


async def _read_subscriptions(
    pool: Any,
    factory: SubscriptionsFactory | None,
) -> Sequence[BlogPostSubscription]:
    if factory is not None:
        return await factory()
    return await list_active_blog_post_subscriptions(pool)


def _serialize_payload(blueprint: Any) -> dict[str, Any]:
    """Convert a PostBlueprint (or duck-typed analog) into a
    JSONB-safe dict. Nested dataclasses (SectionSpec /
    ChartSpec) round-trip through ``asdict``."""

    return {
        "sections": [
            _to_dict(item) for item in (getattr(blueprint, "sections", None) or [])
        ],
        "charts": [
            _to_dict(item) for item in (getattr(blueprint, "charts", None) or [])
        ],
        "tags": list(getattr(blueprint, "tags", None) or []),
        "data_context": dict(getattr(blueprint, "data_context", None) or {}),
        "quotable_phrases": [
            _to_dict(item)
            for item in (getattr(blueprint, "quotable_phrases", None) or [])
        ],
        "cta": _to_dict(getattr(blueprint, "cta", None))
        if getattr(blueprint, "cta", None)
        else None,
    }


def _to_dict(value: Any) -> Any:
    """Coerce a dataclass / mapping into a plain dict."""

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value


__all__ = [
    "DEFAULT_TARGET_MODE",
    "fanout_blueprint",
]
