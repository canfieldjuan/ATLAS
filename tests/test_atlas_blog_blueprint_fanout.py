"""Pin the blog-blueprint fanout helper.

`atlas_brain/_blog_blueprint_fanout.py` is the bridge between
the autonomous task's in-memory `PostBlueprint` and the
per-tenant `blog_blueprints` table. PR-Subscriptions-2 hooks
this into the autonomous task's two `_assemble_and_store`
callsites; the tests here pin the helper's contract:

1. Empty subscription list short-circuits to 0 (no repo
   construction, no save).
2. No-matching-subscriber short-circuits (subs exist but
   all reject the blueprint's topic_type / target_mode).
3. Multi-subscriber fanout returns the count of saves.
4. Per-account save failure is logged but doesn't abort the
   remaining fanout (partial fanout is better than none).
5. Payload serialization handles SectionSpec / ChartSpec
   nested dataclasses.
6. `target_mode` default is the package-level constant.
7. `target_mode` override is honored.
8. `topic_types` / `target_modes` filters integrate
   correctly via `matches()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from types import SimpleNamespace

import pytest

from atlas_brain._blog_blueprint_fanout import (
    DEFAULT_TARGET_MODE,
    fanout_blueprint,
)
from atlas_brain._blog_post_subscriptions import BlogPostSubscription


@dataclass
class _ChartStub:
    chart_id: str
    chart_type: str
    title: str
    data: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SectionStub:
    id: str
    heading: str
    goal: str
    key_stats: dict[str, Any] = field(default_factory=dict)


def _blueprint(
    *,
    topic_type: str = "vendor_alternative",
    slug: str = "acme-pricing",
) -> Any:
    """Duck-type the host's PostBlueprint without importing
    the 9000-line autonomous task."""

    return SimpleNamespace(
        topic_type=topic_type,
        slug=slug,
        suggested_title="Acme Pricing Pressure",
        tags=["pricing", "retention"],
        data_context={"vendor": "Acme", "report_date": "2026-05-09"},
        sections=[_SectionStub(id="intro", heading="Why Acme", goal="Frame")],
        charts=[_ChartStub(chart_id="px", chart_type="bar", title="Mentions")],
        quotable_phrases=[{"text": "moving away from Acme", "vendor": "Acme"}],
        cta={"label": "See alternatives", "href": "/alts"},
    )


class _RecordingRepo:
    def __init__(self, *, fail_for_account: str | None = None) -> None:
        self.saved: list[tuple[str, str]] = []
        self.fail_for_account = fail_for_account

    async def save_blueprints(self, blueprints, *, scope):
        if self.fail_for_account and scope.account_id == self.fail_for_account:
            raise RuntimeError("simulated DB failure")
        for bp in blueprints:
            self.saved.append((scope.account_id, bp.slug))


@pytest.mark.asyncio
async def test_empty_subscription_list_short_circuits() -> None:
    repo = _RecordingRepo()

    async def _no_subs() -> list[BlogPostSubscription]:
        return []

    saved = await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(),
        subscriptions_factory=_no_subs,
        repo_factory=lambda _pool: repo,
    )

    assert saved == 0
    assert repo.saved == []


@pytest.mark.asyncio
async def test_no_matching_subscriber_short_circuits() -> None:
    """All subs reject the blueprint's topic_type so the
    fanout writes nothing -- distinct from the empty-list
    canary because subs exist; just none match."""

    repo = _RecordingRepo()

    async def _subs() -> list[BlogPostSubscription]:
        return [
            BlogPostSubscription(
                account_id="acct-1",
                topic_types=("churn_report",),
            ),
        ]

    saved = await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(topic_type="vendor_alternative"),
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: repo,
    )
    assert saved == 0
    assert repo.saved == []


@pytest.mark.asyncio
async def test_multi_subscriber_fanout_returns_count() -> None:
    repo = _RecordingRepo()

    async def _subs() -> list[BlogPostSubscription]:
        return [
            BlogPostSubscription(account_id="acct-1"),
            BlogPostSubscription(account_id="acct-2"),
            BlogPostSubscription(account_id="acct-3"),
        ]

    saved = await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(slug="acme-pricing"),
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: repo,
    )
    assert saved == 3
    assert sorted(repo.saved) == [
        ("acct-1", "acme-pricing"),
        ("acct-2", "acme-pricing"),
        ("acct-3", "acme-pricing"),
    ]


@pytest.mark.asyncio
async def test_per_account_save_failure_does_not_abort_others() -> None:
    """The autonomous task's draft is already stored when
    fanout runs; one bad subscription must not lose the
    remaining writes."""

    repo = _RecordingRepo(fail_for_account="acct-2")

    async def _subs() -> list[BlogPostSubscription]:
        return [
            BlogPostSubscription(account_id="acct-1"),
            BlogPostSubscription(account_id="acct-2"),  # fails
            BlogPostSubscription(account_id="acct-3"),
        ]

    saved = await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(),
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: repo,
    )
    assert saved == 2
    assert sorted(repo.saved) == [
        ("acct-1", "acme-pricing"),
        ("acct-3", "acme-pricing"),
    ]


@pytest.mark.asyncio
async def test_payload_serializes_nested_dataclasses() -> None:
    """SectionSpec / ChartSpec are dataclasses; the helper
    converts them via asdict so JSONB serialization in the
    Postgres adapter (PR #458) sees plain dicts."""

    captured: dict[str, Any] = {}

    class _PayloadCapturingRepo:
        async def save_blueprints(self, blueprints, *, scope):
            captured["payload"] = blueprints[0].payload

    async def _subs() -> list[BlogPostSubscription]:
        return [BlogPostSubscription(account_id="acct-1")]

    await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(),
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: _PayloadCapturingRepo(),
    )

    payload = captured["payload"]
    assert payload["sections"] == [
        {"id": "intro", "heading": "Why Acme", "goal": "Frame", "key_stats": {}},
    ]
    assert payload["charts"] == [
        {
            "chart_id": "px",
            "chart_type": "bar",
            "title": "Mentions",
            "data": [],
            "config": {},
        },
    ]
    assert payload["tags"] == ["pricing", "retention"]
    assert payload["data_context"]["vendor"] == "Acme"
    assert payload["quotable_phrases"][0]["vendor"] == "Acme"
    assert payload["cta"]["label"] == "See alternatives"


def test_default_target_mode_constant_is_b2b_blog_post() -> None:
    """The autonomous task fans out under a single target_mode
    so subscriptions can fan in by topic_type alone. Pin the
    constant so a renaming refactor surfaces here."""

    assert DEFAULT_TARGET_MODE == "b2b_blog_post"


@pytest.mark.asyncio
async def test_target_mode_override_is_honored() -> None:
    captured: dict[str, Any] = {}

    class _Repo:
        async def save_blueprints(self, blueprints, *, scope):
            captured["target_mode"] = blueprints[0].target_mode

    async def _subs() -> list[BlogPostSubscription]:
        return [BlogPostSubscription(account_id="acct-1")]

    await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(),
        target_mode="executive_brief",
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: _Repo(),
    )
    assert captured["target_mode"] == "executive_brief"


@pytest.mark.asyncio
async def test_target_modes_filter_rejects_default_target_mode() -> None:
    """A subscription with target_modes=("executive_brief",)
    rejects the default b2b_blog_post target_mode and the
    fanout writes nothing for that account. Confirms the
    helper passes target_mode through matches() correctly."""

    repo = _RecordingRepo()

    async def _subs() -> list[BlogPostSubscription]:
        return [
            BlogPostSubscription(
                account_id="acct-only-exec",
                target_modes=("executive_brief",),
            ),
            BlogPostSubscription(
                account_id="acct-firehose",
            ),
        ]

    saved = await fanout_blueprint(
        pool=SimpleNamespace(),
        blueprint=_blueprint(),
        subscriptions_factory=_subs,
        repo_factory=lambda _pool: repo,
    )
    # Only the firehose subscription matches the default
    # target_mode; the exec-brief-only subscription is
    # filtered out.
    assert saved == 1
    assert repo.saved == [("acct-firehose", "acme-pricing")]
